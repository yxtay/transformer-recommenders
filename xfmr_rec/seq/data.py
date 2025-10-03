from collections.abc import Callable
from typing import TypedDict

import datasets
import lightning as lp
import numpy as np
import pandas as pd
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

NumpyStrArray = np.typing.NDArray[np.str_]
NumpyIntArray = np.typing.NDArray[np.int_]
NumpyBoolArray = np.typing.NDArray[np.bool_]


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
        """Initialize the sequential dataset.

        Args:
            config: Configuration controlling sequence length and lookahead.
            items_dataset: Dataset of items providing ``item_id`` -> ``item_text`` mapping.
            events_dataset: Per-user events dataset used to build history sequences.
        """
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

    def map_id2idx(
        self,
        example: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Map per-user history item ids to internal indices.

        Designed to be used with :meth:`datasets.Dataset.map`; filters out any
        item ids that are not present in the current item vocabulary and
        returns numpy arrays for indices and labels.

        Args:
            example: A mapping containing ``history.item_id`` and ``history.label``.

        Returns:
            A dict with keys ``history_item_idx`` and ``history_label`` containing
            numpy arrays.
        """
        item_ids = example["history.item_id"]
        labels = example["history.label"]
        mask = [item_id in self.id2idx.index for item_id in item_ids]
        item_idx = self.id2idx[item_ids[mask]].to_numpy()
        return {"history_item_idx": item_idx, "history_label": labels[mask]}

    def duplicate_rows(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Duplicate per-user rows so long histories can be chunked.

        For users with histories longer than ``max_seq_length`` this function
        creates multiple rows so each produced row can be sampled into a
        fixed-size sequence during training.

        Args:
            batch: A batch mapping produced during dataset mapping.

        Returns:
            The expanded batch with duplicated rows.
        """
        history_item_idx = batch["history_item_idx"]
        num_copies = [
            ((len(history) - 1) // self.config.max_seq_length + 1)
            for history in history_item_idx
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
        """Preprocess and expand per-user event records for sequential sampling.

        Flattens each user's history, maps item ids to indices and duplicates
        rows for long histories so downstream sampling can produce fixed-size
        sequences.

        Args:
            events_dataset: Dataset containing per-user interactions.

        Returns:
            Dataset expanded for sequential sampling.
        """
        return (
            events_dataset.flatten()
            .select_columns(["history.item_id", "history.label"])
            .with_format("numpy")
            .map(self.map_id2idx)
            .map(self.duplicate_rows, batched=True)
        )

    def __len__(self) -> int:
        """Number of examples (rows) in the processed events dataset.

        Returns:
            The number of rows in the processed events dataset.
        """
        return len(self.events_dataset)

    def sample_sequence(self, history_item_idx: NumpyIntArray) -> NumpyIntArray:
        """Sample a set of positions from a user's history for training.

        Returns a sorted array of indices (positions) selected from the
        history excluding the final item which serves as a candidate positive.
        If the history is shorter than ``max_seq_length`` all available
        positions are returned.

        Args:
            history_item_idx: Array of item indices in the user's history.

        Returns:
            Sorted array of sampled history positions.
        """
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
        """For each sampled position pick a positive item from future interactions.

        If ``pos_lookahead`` is zero the positive is sampled from any later
        interaction; otherwise the choice is restricted to the next
        ``pos_lookahead`` positions.

        Args:
            history_item_idx: Array of item indices in the user's history.
            history_label: Boolean array indicating positive interactions.
            sampled_indices: Indices sampled from the history to produce positives for.

        Returns:
            Array of positive item indices aligned with ``sampled_indices``.
        """
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
        """Sample negative item indices not present in the user's history.

        Returns ``seq_len`` negative indices sampled uniformly without
        replacement where possible.

        Args:
            history_item_idx: Array of item indices in the user's history.
            sampled_indices: Sampled positions used to determine sequence length.

        Returns:
            Array of negative item indices.
        """
        seq_len = len(sampled_indices)
        neg_candidates = list(self.all_idx - set(history_item_idx.tolist()))
        if len(neg_candidates) == 0:
            neg_candidates = list(self.all_idx)

        return self.rng.choice(
            neg_candidates, seq_len, replace=len(neg_candidates) < seq_len
        )

    def __getitem__(self, idx: int) -> SeqExample:
        """Return a training example containing history, positives and negatives.

        The returned :class:`SeqExample` contains tensors for indices and
        lists of item texts for history/positive/negative items aligned by
        position.

        Args:
            idx: Index of the example to retrieve.

        Returns:
            A :class:`SeqExample` mapping field names to tensors and lists.
        """
        row = self.events_dataset[idx]
        history_item_idx = row["history_item_idx"]
        history_label = row["history_label"]

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
        """Collate a list of examples into a batched SeqBatch.

        Pads tensor sequences and leaves lists of texts as-is so they can be
        encoded downstream.

        Args:
            batch: List of ``SeqExample`` items to collate.

        Returns:
            A ``SeqBatch`` with padded tensors and lists of texts.
        """
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

    def prepare_data(self, *, overwrite: bool = False) -> None:
        from filelock import FileLock

        """Download and prepare MovieLens artifacts for the sequence data module.

        This method acquires a file lock to avoid concurrent downloads and
        calls :func:`xfmr_rec.data.prepare_movielens` to create the processed
        parquet artifacts.

        Args:
            overwrite: If True, force re-download and re-processing of the dataset.
        """
        data_dir = self.config.data_dir
        with FileLock(f"{data_dir}.lock"):
            download_unpack_data(MOVIELENS_1M_URL, data_dir, overwrite=overwrite)
            prepare_movielens(data_dir, overwrite=overwrite)

    def setup(self, stage: str | None = None) -> None:
        """Prepare datasets for the specified stage.

        This method loads item and user parquet files, computes embeddings for
        items if necessary, and constructs train/val/test datasets as
        :class:`SeqDataset` or :class:`datasets.Dataset` depending on the split.

        Args:
            stage: Optional stage indicator such as 'fit', 'validate', 'test',
                or 'predict'. If None, prepares all datasets.
        """
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
                events_dataset=train_dataset,
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
    ) -> torch_data.DataLoader[dict[str, torch.Tensor]]:
        """Create a PyTorch DataLoader from a HuggingFace dataset.

        Args:
            dataset: The dataset to wrap.
            shuffle: Whether to shuffle the dataset each epoch.
            batch_size: Batch size to use for the DataLoader. If None, uses
                the configured batch size.
            collate_fn: Optional collate function applied to each batch.

        Returns:
            A configured :class:`torch.utils.data.DataLoader` instance.
        """
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

    def train_dataloader(self) -> torch_data.DataLoader[SeqBatch]:
        """Return the training DataLoader.

        Returns:
            DataLoader for the training dataset.
        """
        return self.get_dataloader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.config.batch_size,
            collate_fn=self.train_dataset.collate,
        )

    def val_dataloader(self) -> torch_data.DataLoader[dict[str, torch.Tensor]]:
        """Return the validation DataLoader.

        Returns:
            DataLoader for the validation dataset.
        """
        return self.get_dataloader(self.val_dataset)

    def test_dataloader(self) -> torch_data.DataLoader[dict[str, torch.Tensor]]:
        """Return the test DataLoader.

        Returns:
            DataLoader for the test dataset.
        """
        return self.get_dataloader(self.test_dataset)

    def predict_dataloader(self) -> torch_data.DataLoader[dict[str, torch.Tensor]]:
        """Return the prediction DataLoader.

        Returns:
            DataLoader for the prediction dataset.
        """
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
