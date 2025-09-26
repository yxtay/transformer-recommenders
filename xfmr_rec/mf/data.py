from collections.abc import Callable

import datasets
import lightning as lp
import numpy as np
import pandas as pd
import polars as pl
import pyarrow.compute as pc
import pydantic
import torch
import torch.utils.data as torch_data
from loguru import logger

from xfmr_rec.data import download_unpack_data, prepare_movielens
from xfmr_rec.params import DATA_DIR, ITEMS_PARQUET, MOVIELENS_1M_URL, USERS_PARQUET


class MFDatasetConfig(pydantic.BaseModel):
    query_sampling_prob: float = 0.5


class MFDataModuleConfig(MFDatasetConfig):
    data_dir: str = DATA_DIR
    items_parquet: str = ITEMS_PARQUET
    users_parquet: str = USERS_PARQUET

    batch_size: int = 1024
    num_workers: int = 1


class MFDataset(torch_data.Dataset[dict[str, str]]):
    def __init__(
        self,
        config: MFDatasetConfig,
        *,
        items_dataset: datasets.Dataset,
        users_dataset: datasets.Dataset,
    ) -> None:
        self.config = MFDatasetConfig.model_validate(config)
        self.rng = np.random.default_rng()

        self.id2idx = pd.Series(
            pd.RangeIndex(len(items_dataset)),
            index=items_dataset.with_format("pandas")["item_id"].array,
        )
        self.all_idx = set(self.id2idx)
        self.item_text: datasets.Column = items_dataset["item_text"]

        self.events_dataset = self.process_events(users_dataset)

        logger.info(f"num_rows: {len(self)}, num_items: {len(self.id2idx)}")

    def duplicate_rows(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        history_item_idx = batch["history.item_id"]
        num_copies = [len(seq) for seq in history_item_idx]
        return {key: batch[key].repeat(num_copies) for key in batch}

    def process_events(self, users_dataset: datasets.Dataset) -> datasets.Dataset:
        return (
            users_dataset.flatten()
            .select_columns(["user_text", "history.item_id", "history.label"])
            .with_format("numpy")
            .map(self.duplicate_rows, batched=True)
        )

    def __len__(self) -> int:
        return len(self.events_dataset)

    def map_id2idx(
        self,
        item_ids: np.typing.NDArray[np.str_],
        labels: np.typing.NDArray[np.bool],
    ) -> list[int]:
        mask = [item_id in self.id2idx.index for item_id in item_ids]
        item_idx = self.id2idx[item_ids[mask]].to_numpy()
        return item_idx[labels[mask]].tolist()

    def sample_positive_idx(self, history_item_idx: list[int]) -> int:
        indices = range(len(history_item_idx))
        return self.rng.choice(indices).item()

    def sample_negative(self, history_item_idx: list[int]) -> int:
        neg_candidates = list(self.all_idx - set(history_item_idx))
        if len(neg_candidates) == 0:
            neg_candidates = list(self.all_idx)
        return self.rng.choice(neg_candidates).item()

    def sample_query_text(
        self,
        user_text: str,
        pos_idx: int,
        history_item_idx: list[int],
    ) -> str:
        query_candidates = history_item_idx[:pos_idx]
        if (
            self.rng.random() > self.config.query_sampling_prob
            or len(query_candidates) == 0
        ):
            return user_text

        query_idx = self.rng.choice(query_candidates)
        return self.item_text[query_idx]

    def __getitem__(self, idx: int) -> dict[str, str]:
        row = self.events_dataset[idx]
        history_item_idx = self.map_id2idx(row["history.item_id"], row["history.label"])

        pos_idx = self.sample_positive_idx(history_item_idx=history_item_idx)
        neg_item_idx = self.sample_negative(history_item_idx=history_item_idx)
        query_text = self.sample_query_text(
            user_text=row["user_text"],
            pos_idx=pos_idx,
            history_item_idx=history_item_idx,
        )
        return {
            "pos_text": self.item_text[history_item_idx[pos_idx]],
            "neg_text": self.item_text[neg_item_idx],
            "query_text": query_text,
        }


class MFDataModule(lp.LightningDataModule):
    def __init__(self, config: MFDataModuleConfig) -> None:
        super().__init__()
        self.config = MFDataModuleConfig.model_validate(config)
        self.save_hyperparameters(self.config.model_dump())

        self.items_dataset: datasets.Dataset | None = None
        self.users_dataset: datasets.Dataset | None = None
        self.train_dataset: MFDataset | None = None
        self.val_dataset: datasets.Dataset | None = None
        self.test_dataset: datasets.Dataset | None = None
        self.predict_dataset: datasets.Dataset | None = None

    def prepare_data(self, *, overwrite: bool = False) -> pl.LazyFrame:
        from filelock import FileLock

        data_dir = self.config.data_dir
        with FileLock(f"{data_dir}.lock"):
            download_unpack_data(MOVIELENS_1M_URL, data_dir, overwrite=overwrite)
            return prepare_movielens(data_dir, overwrite=overwrite)

    def setup(self, stage: str | None = None) -> None:
        if self.items_dataset is None:
            self.items_dataset = datasets.Dataset.from_parquet(
                self.config.items_parquet
            )

        if self.users_dataset is None:
            self.users_dataset = datasets.Dataset.from_parquet(
                self.config.users_parquet
            )

        if self.train_dataset is None:
            train_dataset = datasets.Dataset.from_parquet(
                self.config.users_parquet, filters=pc.field("is_train")
            )
            self.train_dataset = MFDataset(
                self.config,
                items_dataset=self.items_dataset,
                users_dataset=train_dataset,
            )

        if self.val_dataset is None and stage in {"fit", "validate", None}:
            self.val_dataset = datasets.Dataset.from_parquet(
                self.config.users_parquet, filters=pc.field("is_val")
            ).with_format("numpy")

        if self.test_dataset is None and stage in {"test", None}:
            self.test_dataset = datasets.Dataset.from_parquet(
                self.config.users_parquet, filters=pc.field("is_test")
            ).with_format("numpy")

        if self.predict_dataset is None:
            self.predict_dataset = datasets.Dataset.from_parquet(
                self.config.users_parquet, filters=pc.field("is_predict")
            ).with_format("numpy")

    def get_dataloader(
        self,
        dataset: datasets.Dataset,
        *,
        shuffle: bool = False,
        batch_size: int | None = None,
        collate_fn: Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]
        | None = None,
    ) -> torch_data.DataLoader:
        return torch_data.DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=self.config.num_workers,
            multiprocessing_context="spawn" if self.config.num_workers > 0 else None,
            persistent_workers=self.config.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
        )

    def train_dataloader(self) -> torch_data.DataLoader:
        return self.get_dataloader(
            self.train_dataset, shuffle=True, batch_size=self.config.batch_size
        )

    def val_dataloader(self) -> torch_data.DataLoader:
        return self.get_dataloader(self.val_dataset)

    def test_dataloader(self) -> torch_data.DataLoader:
        return self.get_dataloader(self.test_dataset)

    def predict_dataloader(self) -> torch_data.DataLoader:
        return self.get_dataloader(self.predict_dataset)


if __name__ == "__main__":
    import rich

    config = MFDataModuleConfig()
    datamodule = MFDataModule(config)
    datamodule.prepare_data()
    datamodule.setup()
    print(datamodule)

    dataloaders = [
        datamodule.items_dataset,
        datamodule.users_dataset,
        datamodule.train_dataset,
        datamodule.train_dataloader(),
        datamodule.val_dataset,
        datamodule.val_dataloader(),
        datamodule.test_dataset,
        datamodule.test_dataloader(),
    ]
    for dataloader in dataloaders:
        batch = next(iter(dataloader))
        rich.print(batch)
        shapes = {
            key: value.shape
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }
        rich.print(shapes)
