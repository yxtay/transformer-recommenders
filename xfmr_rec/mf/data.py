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


class MFDataModuleConfig(pydantic.BaseModel):
    data_dir: str = DATA_DIR
    items_parquet: str = ITEMS_PARQUET
    users_parquet: str = USERS_PARQUET

    batch_size: int = 1024
    num_workers: int = 1


class MFDataset(torch_data.Dataset[dict[str, str]]):
    def __init__(
        self,
        items_dataset: datasets.Dataset,
        users_dataset: datasets.Dataset,
    ) -> None:
        self.rng = np.random.default_rng()

        self.id2idx = pd.Series({k: i for i, k in enumerate(items_dataset["item_id"])})
        self.all_idx = set(self.id2idx)
        self.item_text: list[str] = items_dataset["item_text"]

        self.users_dataset = self.process_events(users_dataset)

        logger.info(f"num_rows: {len(self)}, num_items: {len(self.id2idx)}")

    def process_events(self, users_dataset: datasets.Dataset) -> datasets.Dataset:
        def map_item_idx(example: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            item_ids = example["history.item_id"]
            labels = example["history.label"]

            mask = [item_id in self.id2idx.index for item_id in item_ids]
            item_idx = self.id2idx[item_ids[mask]].to_numpy()
            return {"history_item_idx": item_idx[labels[mask]]}

        def duplicate_rows(
            batch: dict[str, list[np.ndarray]],
        ) -> dict[str, list[np.ndarray]]:
            history_item_idx = batch["history_item_idx"]
            num_copies = [len(seq) for seq in history_item_idx]
            return {key: np.repeat(batch[key], num_copies) for key in batch}

        return (
            users_dataset.flatten()
            .select_columns(["user_text", "history.item_id", "history.label"])
            .with_format("numpy")
            .map(map_item_idx)
            .map(duplicate_rows, batched=True)
            .with_format("torch")
        )

    def sample_positive(self, history_item_idx: torch.Tensor) -> torch.Tensor:
        pos_candidates = history_item_idx
        return self.rng.choice(pos_candidates)

    def sample_negative(self, history_item_idx: torch.Tensor) -> torch.Tensor:
        neg_candidates = list(self.all_idx - set(history_item_idx.tolist()))
        if len(neg_candidates) == 0:
            neg_candidates = list(self.all_idx)
        return self.rng.choice(neg_candidates)

    def __len__(self) -> int:
        return len(self.users_dataset)

    def __getitem__(self, idx: int) -> dict[str, str]:
        row = self.users_dataset[idx]
        user_text = row["user_text"]
        history_item_idx = row["history_item_idx"]

        pos_item_idx = self.sample_positive(history_item_idx=history_item_idx)
        neg_item_idx = self.sample_negative(history_item_idx=history_item_idx)
        return {
            "user_text": user_text,
            "pos_item_text": self.item_text[pos_item_idx],
            "neg_item_text": self.item_text[neg_item_idx],
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
                items_dataset=self.items_dataset, users_dataset=train_dataset
            )

        if self.val_dataset is None and stage in {"fit", "validate", None}:
            self.val_dataset = datasets.Dataset.from_parquet(
                self.config.users_parquet, filters=pc.field("is_val")
            )

        if self.test_dataset is None and stage in {"test", None}:
            self.test_dataset = datasets.Dataset.from_parquet(
                self.config.users_parquet, filters=pc.field("is_test")
            )

        if self.predict_dataset is None:
            self.predict_dataset = datasets.Dataset.from_parquet(
                self.config.users_parquet, filters=pc.field("is_predict")
            )

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
        # datamodule.val_dataset,
        # datamodule.val_dataloader(),
        # datamodule.test_dataset,
        # datamodule.test_dataloader(),
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
