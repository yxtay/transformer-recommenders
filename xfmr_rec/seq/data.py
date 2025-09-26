from collections.abc import Callable
from typing import TypedDict

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
from sentence_transformers import SentenceTransformer
from torch.nn.utils.rnn import pad_sequence

from xfmr_rec.data import download_unpack_data, prepare_movielens
from xfmr_rec.params import (
    DATA_DIR,
    ITEMS_PARQUET,
    MOVIELENS_1M_URL,
    PRETRAINED_MODEL_NAME,
    USERS_PARQUET,
)

NumpyStrArray = np.typing.NDArray[str]
NumpyIntArray = np.typing.NDArray[int]
NumpyBoolArray = np.typing.NDArray[bool]


class SeqExample(TypedDict):
    history_item_idx: torch.Tensor
    history_item_text: list[str]
    pos_item_idx: torch.Tensor
    pos_item_text: list[str]
    neg_item_idx: torch.Tensor
    neg_item_text: list[str]


class SeqBatch(TypedDict):
    history_item_idx: torch.Tensor
    history_item_text: list[list[str]]
    pos_item_idx: torch.Tensor
    pos_item_text: list[list[str]]
    neg_item_idx: torch.Tensor
    neg_item_text: list[list[str]]


class SeqDataConfig(pydantic.BaseModel):
    max_seq_length: int = 32
    pos_lookahead: int = 0
    num_negatives: int = 3


class SeqDataModuleConfig(SeqDataConfig):
    data_dir: str = DATA_DIR
    items_parquet: str = ITEMS_PARQUET
    users_parquet: str = USERS_PARQUET

    pretrained_model_name: str = PRETRAINED_MODEL_NAME
    batch_size: int = 32
    num_workers: int = 1


class SeqDataset(torch_data.Dataset[SeqExample]):
    def __init__(
        self,
        config: SeqDataConfig,
        *,
        items_dataset: datasets.Dataset,
        events_dataset: datasets.Dataset,
    ) -> None:
        self.config = config
        self.rng = np.random.default_rng()

        # idx 0 for padding
        self.id2idx = pd.Series(
            pd.RangeIndex(len(items_dataset)) + 1,
            index=items_dataset.with_format("pandas")["item_id"].array,
        )
        self.all_idx = set(self.id2idx)
        self.item_texts: datasets.Column = items_dataset["item_text"]

        self.events_dataset = self.process_events(events_dataset)

        logger.info(repr(self.config))
        logger.info(f"num_rows: {len(self)}, num_items: {len(self.id2idx)}")

    def duplicate_rows(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        history_item_idx = batch["history.item_id"]
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

    def process_events(self, events_dataset: datasets.Dataset) -> datasets.Dataset:
        return (
            events_dataset.flatten()
            .select_columns(["history.item_id", "history.label"])
            .with_format("numpy")
            .map(self.duplicate_rows, batched=True)
        )

    def __len__(self) -> int:
        return len(self.events_dataset)

    def map_id2idx(
        self,
        item_ids: NumpyStrArray,
        labels: NumpyBoolArray,
    ) -> tuple[NumpyIntArray, NumpyBoolArray]:
        mask = [item_id in self.id2idx.index for item_id in item_ids]
        item_idx = self.id2idx[item_ids[mask]].to_numpy()
        return item_idx, labels[mask]

    def sample_sequence(self, history_item_idx: NumpyIntArray) -> NumpyIntArray:
        indices = np.arange(len(history_item_idx) - 1)
        max_seq_length = self.config.max_seq_length
        if len(indices) <= max_seq_length:
            return indices

        return np.sort(self.rng.choice(indices, size=max_seq_length, replace=False))

    def sample_positives(
        self,
        history_item_idx: NumpyIntArray,
        history_label: NumpyBoolArray,
        sampled_indices: NumpyIntArray,
    ) -> NumpyIntArray:
        positives = np.zeros_like(sampled_indices)
        pos_lookahead = self.config.pos_lookahead

        for i, idx in enumerate(sampled_indices):
            start_idx = idx + 1
            end_idx = start_idx + pos_lookahead if pos_lookahead > 0 else None
            pos_candidates = history_item_idx[start_idx:end_idx]
            pos_candidates = pos_candidates[history_label[start_idx:end_idx]]

            if len(pos_candidates) > 0:
                positives[i] = self.rng.choice(pos_candidates)
        return positives

    def sample_negatives(
        self,
        history_item_idx: NumpyIntArray,
        sampled_indices: NumpyIntArray,
    ) -> NumpyIntArray:
        seq_len = len(sampled_indices)
        neg_candidates = list(self.all_idx - set(history_item_idx.tolist()))
        if len(neg_candidates) == 0:
            neg_candidates = list(self.all_idx)

        return self.rng.choice(
            neg_candidates, seq_len, replace=len(neg_candidates) < seq_len
        )

    def __getitem__(self, idx: int) -> SeqExample:
        row = self.events_dataset[idx]
        history_item_idx, history_label = self.map_id2idx(
            row["history.item_id"], row["history.label"]
        )

        sampled_indices = self.sample_sequence(history_item_idx)
        pos_item_idx = self.sample_positives(
            history_item_idx=history_item_idx,
            history_label=history_label,
            sampled_indices=sampled_indices,
        )
        neg_item_idx = self.sample_negatives(
            history_item_idx=history_item_idx,
            sampled_indices=sampled_indices,
        )
        return {
            "history_item_idx": torch.as_tensor(history_item_idx[sampled_indices]),
            "history_item_text": self.item_texts[history_item_idx[sampled_indices] - 1],
            "pos_item_idx": torch.as_tensor(pos_item_idx),
            "pos_item_text": self.item_texts[pos_item_idx - 1],
            "neg_item_idx": torch.as_tensor(neg_item_idx),
            "neg_item_text": self.item_texts[neg_item_idx - 1],
        }

    def collate(self, batch: list[SeqExample]) -> SeqBatch:
        collated = {key: [example[key] for example in batch] for key in batch[0]}
        return {
            key: pad_sequence(value, batch_first=True)
            if isinstance(value[0], torch.Tensor)
            else value
            for key, value in collated.items()
        }


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
            model = SentenceTransformer(self.config.pretrained_model_name)
            self.items_dataset = datasets.Dataset.from_parquet(
                self.config.items_parquet
            ).map(
                lambda batch: {"embedding": model.encode(batch["item_text"])},
                batched=True,
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
            if isinstance(value, (torch.Tensor, np.ndarray))
        }
        rich.print(shapes)
