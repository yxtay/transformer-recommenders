from collections.abc import Callable

import datasets
import lightning as lp
import numpy as np
import pandas as pd
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
        events_dataset: datasets.Dataset,
    ) -> None:
        """Create an MFDataset for contrastive training of matching models.

        Args:
            config: Configuration object controlling sampling behavior.
            items_dataset: Datasets ``Dataset`` containing item metadata
                (``item_id`` and ``item_text``).
            events_dataset: Per-user events dataset used to build
                query/positive/negative triplets.
        """
        self.config = MFDatasetConfig.model_validate(config)
        self.rng = np.random.default_rng()

        self.id2idx = pd.Series(
            pd.RangeIndex(len(items_dataset)),
            index=items_dataset.with_format("pandas")["item_id"].array,
        )
        self.all_idx = set(self.id2idx)
        self.item_text: datasets.Column = items_dataset["item_text"]

        self.events_dataset = self.process_events(events_dataset)

        logger.info(f"num_rows: {len(self)}, num_items: {len(self.id2idx)}")

    def map_id2idx(
        self,
        example: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Map raw item ids in a user's history to integer indices.

        This helper is intended to be used with :meth:`datasets.Dataset.map` and
        converts arrays of item ids and labels into numpy arrays of internal
        indices and masked labels for items that exist in the item vocabulary.

        Args:
            example: A batch/example dictionary with keys ``history.item_id`` and
                ``history.label``.

        Returns:
            Dictionary containing ``history_item_idx`` and ``history_label``
            numpy arrays.
        """
        item_ids = example["history.item_id"]
        labels = example["history.label"]
        mask = [item_id in self.id2idx.index for item_id in item_ids]
        item_idx = self.id2idx[item_ids[mask]].to_numpy()
        return {"history_item_idx": item_idx, "history_label": labels[mask]}

    def duplicate_rows(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Duplicate rows so each training query is a separate example.

        For each user example this function calculates how many training
        queries can be sampled from the history and repeats the arrays
        accordingly so every resulting row corresponds to a single query
        position.

        Args:
            batch: A batch from the dataset mapping stage containing
                ``history_item_idx`` and ``history_label``.

        Returns:
            The transformed batch with rows duplicated per query.
        """
        history_item_idx = batch["history_item_idx"]
        num_copies = [len(history) - 1 for history in history_item_idx]
        return {key: batch[key].repeat(num_copies) for key in batch}

    def process_events(self, events_dataset: datasets.Dataset) -> datasets.Dataset:
        """Preprocess events dataset into per-query training rows.

        Flattens the per-user nested history into rows suitable for
        sampling query/positive/negative examples and maps item ids to
        internal indices.

        Args:
            events_dataset: A ``datasets.Dataset`` containing per-user interaction records.

        Returns:
            A dataset where each row corresponds to a potential training query.
        """
        return (
            events_dataset.flatten()
            .select_columns(["user_text", "history.item_id", "history.label"])
            .with_format("numpy")
            .map(self.map_id2idx)
            .map(self.duplicate_rows, batched=True)
        )

    def __len__(self) -> int:
        """Return the number of training rows available.

        Returns:
            Number of rows in the processed events dataset.
        """
        return len(self.events_dataset)

    def sample_query_idx(self, history_item_idx: np.ndarray) -> int:
        """Randomly sample a query (position) index from history.

        The query is selected from positions that have at least one later
        item to serve as a positive.

        Args:
            history_item_idx: Array of history item indices for the user.

        Returns:
            The sampled query index.
        """
        indices = range(len(history_item_idx) - 1)
        return self.rng.choice(indices).item()

    def sample_positive(
        self, history_item_idx: np.ndarray, history_label: np.ndarray, query_idx: int
    ) -> int:
        """Sample a positive item index for a given query position.

        Only items after the query position that have a positive label are
        considered.

        Args:
            history_item_idx: Array of history item indices.
            history_label: Boolean array of labels for each history position.
            query_idx: Position index to sample a positive for.

        Returns:
            The sampled positive item index.
        """
        pos_candidates = history_item_idx[query_idx + 1 :]
        pos_candidates = pos_candidates[history_label[query_idx + 1 :]]
        return self.rng.choice(pos_candidates)

    def sample_negative(self, history_item_idx: list[int]) -> int:
        """Sample a negative (non-interacted) item index.

        Picks uniformly from the set of items the user has not interacted
        with; falls back to the full item set if necessary.

        Args:
            history_item_idx: List of item indices the user has interacted with.

        Returns:
            The sampled negative item index.
        """
        neg_candidates = list(self.all_idx - set(history_item_idx))
        if len(neg_candidates) == 0:
            neg_candidates = list(self.all_idx)
        return self.rng.choice(neg_candidates).item()

    def sample_query_text(self, user_text: str, item_text: str) -> str:
        """Return either the item text or the user text to use as the query.

        Chooses between using the last interacted item's text or the user's
        text with probability controlled by ``query_sampling_prob``.

        Args:
            user_text: The user's text representation.
            item_text: The last interacted item's text.

        Returns:
            The text to use as the query.
        """
        return (
            item_text
            if self.rng.random() < self.config.query_sampling_prob
            else user_text
        )

    def __getitem__(self, idx: int) -> dict[str, str]:
        """Return a single training example (query, positive, negative).

        The returned dictionary contains ``query_text``, ``pos_text``, and
        ``neg_text`` fields used by downstream collate functions or
        encoders.

        Args:
            idx: Index of the example to retrieve.

        Returns:
            A dict with keys ``query_text``, ``pos_text``, and ``neg_text``.
        """
        row = self.events_dataset[idx]
        history_item_idx = row["history_item_idx"]
        history_label = row["history_label"]

        query_idx = self.sample_query_idx(history_item_idx=history_item_idx)
        pos_item_idx = self.sample_positive(
            history_item_idx=history_item_idx,
            history_label=history_label,
            query_idx=query_idx,
        )
        neg_item_idx = self.sample_negative(history_item_idx=history_item_idx)
        query_text = self.sample_query_text(
            user_text=row["user_text"],
            item_text=self.item_text[history_item_idx[query_idx]],
        )
        return {
            "query_text": query_text,
            "pos_text": self.item_text[pos_item_idx],
            "neg_text": self.item_text[neg_item_idx],
        }


class MFDataModule(lp.LightningDataModule):
    def __init__(self, config: MFDataModuleConfig) -> None:
        """Initialize the MF PyTorch Lightning DataModule.

        Args:
            config: Data module configuration controlling paths and loader params.
        """
        super().__init__()
        self.config = MFDataModuleConfig.model_validate(config)
        self.save_hyperparameters(self.config.model_dump())

        self.items_dataset: datasets.Dataset | None = None
        self.users_dataset: datasets.Dataset | None = None
        self.train_dataset: MFDataset | None = None
        self.val_dataset: datasets.Dataset | None = None
        self.test_dataset: datasets.Dataset | None = None
        self.predict_dataset: datasets.Dataset | None = None

    def prepare_data(self, *, overwrite: bool = False) -> None:
        from filelock import FileLock

        """Download and prepare MovieLens artifacts for the data module.

        Uses a file lock to avoid concurrent downloads and calls the
        high-level :func:`xfmr_rec.data.prepare_movielens` helper.

        Args:
            overwrite: If True, force re-download and re-processing of the dataset.
        """
        data_dir = self.config.data_dir
        with FileLock(f"{data_dir}.lock"):
            download_unpack_data(MOVIELENS_1M_URL, data_dir, overwrite=overwrite)
            prepare_movielens(data_dir, overwrite=overwrite)

    def setup(self, stage: str | None = None) -> None:
        """Load datasets for the given stage into memory or lazy handles.

        This method populates `items_dataset`, `users_dataset`, and split
        partitions (train/val/test/predict). It is safe to call multiple
        times and will skip loading if datasets are already available.

        Args:
            stage: Optional stage hint such as 'fit', 'validate', 'test', or
                'predict'. When None, all splits are prepared.
        """
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

    def train_dataloader(self) -> torch_data.DataLoader[dict[str, torch.Tensor]]:
        """Return the training DataLoader.

        Returns:
            DataLoader for the training dataset.
        """
        return self.get_dataloader(
            self.train_dataset, shuffle=True, batch_size=self.config.batch_size
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
