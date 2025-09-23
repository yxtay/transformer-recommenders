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


class SeqDataConfig(pydantic.BaseModel):
    max_seq_length: int = 32
    pos_lookahead: int = 0


class SeqDataModuleConfig(SeqDataConfig):
    data_dir: str = DATA_DIR
    items_parquet: str = ITEMS_PARQUET
    users_parquet: str = USERS_PARQUET

    batch_size: int = 32
    num_workers: int = 1


class SeqDataset(torch_data.Dataset[dict[str, list[str]]]):
    def __init__(
        self,
        config: SeqDataConfig,
        *,
        items_dataset: datasets.Dataset,
        users_dataset: datasets.Dataset,
    ) -> None:
        self.config = config
        self.rng = np.random.default_rng()

        self.id2idx = pd.Series({k: i for i, k in enumerate(items_dataset["item_id"])})
        self.all_item_idx = set(self.id2idx)
        self.item_text: list[str] = items_dataset["item_text"]

        self.users_dataset = self.process_events(users_dataset)

        logger.info(repr(self.config))
        logger.info(f"num_rows: {len(self)}, num_items: {len(self.id2idx)}")

    def process_events(self, users_dataset: datasets.Dataset) -> datasets.Dataset:
        def map_item_idx(example: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            item_ids = example["history.item_id"]
            labels = example["history.label"]

            mask = [item_id in self.id2idx.index for item_id in item_ids]
            item_idx = self.id2idx[item_ids[mask]].to_numpy()
            return {
                "history_item_idx": item_idx,
                "history_label": labels[mask],
            }

        def duplicate_rows(
            batch: dict[str, list[np.ndarray]],
        ) -> dict[str, list[np.ndarray]]:
            history_item_idx = batch["history_item_idx"]
            num_copies = [
                ((len(seq) - 1) // self.config.max_seq_length + 1)
                for seq in history_item_idx
            ]
            return {
                key: [
                    el
                    for n, el in zip(num_copies, batch[key], strict=True)
                    for _ in range(n)
                ]
                for key in batch
            }

        return (
            users_dataset.flatten()
            .select_columns(["history.item_id", "history.label"])
            .with_format("numpy")
            .map(map_item_idx)
            .map(duplicate_rows, batched=True)
            .with_format("torch")
        )

    def sample_sequence(self, history_item_idx: torch.Tensor) -> torch.Tensor:
        indices = torch.arange(len(history_item_idx) - 1)

        max_seq_length = self.config.max_seq_length
        if len(indices) <= max_seq_length:
            return indices

        return torch.as_tensor(
            np.sort(self.rng.choice(indices, size=max_seq_length, replace=False))
        )

    def sample_positive(
        self,
        history_item_idx: torch.Tensor,
        history_label: torch.Tensor,
        sampled_indices: torch.Tensor,
    ) -> torch.Tensor:
        positives = torch.zeros_like(sampled_indices)
        pos_lookahead = self.config.pos_lookahead

        for i, idx in enumerate(sampled_indices):
            start_idx = idx + 1
            end_idx = start_idx + pos_lookahead if pos_lookahead > 0 else None
            pos_candidates = history_item_idx[start_idx:end_idx]
            pos_candidates = pos_candidates[history_label[start_idx:end_idx]]

            if len(pos_candidates) > 0:
                positives[i] = self.rng.choice(pos_candidates)
        return positives

    def sample_negative(
        self,
        history_item_idx: torch.Tensor,
        sampled_indices: torch.Tensor,
    ) -> torch.Tensor:
        seq_len = len(sampled_indices)
        neg_candidates = list(self.all_item_idx - set(history_item_idx.tolist()))
        if len(neg_candidates) == 0:
            neg_candidates = list(self.all_item_idx)

        sampled_negatives = self.rng.choice(
            neg_candidates, seq_len, replace=len(neg_candidates) < seq_len
        )
        return torch.as_tensor(sampled_negatives)

    def __len__(self) -> int:
        return len(self.users_dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.users_dataset[idx]
        history_item_idx = row["history_item_idx"]
        history_label = row["history.label"]

        sampled_indices = self.sample_sequence(history_item_idx)
        pos_item_idx = self.sample_positive(
            history_item_idx=history_item_idx,
            history_label=history_label,
            sampled_indices=sampled_indices,
        )
        neg_item_idx = self.sample_negative(
            history_item_idx=history_item_idx,
            sampled_indices=sampled_indices,
        )
        return {
            "history_item_text": self.item_text[history_item_idx[sampled_indices]],
            "pos_item_text": self.item_text[pos_item_idx],
            "neg_item_text": self.item_text[neg_item_idx],
        }

    def collate(self, batch: list[dict[str, list[str]]]) -> dict[str, list[list[str]]]:
        return {col: [example[col] for example in batch] for col in batch[0]}


class SeqDataModule(lp.LightningDataModule):
    def __init__(self, config: SeqDataModuleConfig) -> None:
        super().__init__()
        self.config = SeqDataModuleConfig.model_validate(config)
        self.save_hyperparameters(self.config.model_dump())

        self.items_dataset: datasets.Dataset | None = None
        self.users_dataset: datasets.Dataset | None = None
        self.train_dataset: SeqDataset | None = None
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
            self.train_dataset = SeqDataset(
                config=self.config,
                items_dataset=self.items_dataset,
                users_dataset=train_dataset,
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
            self.train_dataset,
            shuffle=True,
            batch_size=self.config.batch_size,
            collate_fn=self.train_dataset.collate,
        )

    def val_dataloader(self) -> torch_data.DataLoader:
        return self.get_dataloader(self.val_dataset)

    def test_dataloader(self) -> torch_data.DataLoader:
        return self.get_dataloader(self.test_dataset)

    def predict_dataloader(self) -> torch_data.DataLoader:
        return self.get_dataloader(self.predict_dataset)


if __name__ == "__main__":
    import rich

    datamodule = SeqDataModule(SeqDataModuleConfig())
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
