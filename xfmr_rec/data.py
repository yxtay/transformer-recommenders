from __future__ import annotations

import pathlib
import shutil
import tempfile
from collections.abc import Callable
from typing import TYPE_CHECKING, TypedDict

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

from xfmr_rec.params import (
    DATA_DIR,
    ITEMS_PARQUET,
    MOVIELENS_1M_URL,
    PRETRAINED_MODEL_NAME,
    USERS_PARQUET,
)

if TYPE_CHECKING:
    import numpy as np

###
# download data
###


def download_data(
    url: str = MOVIELENS_1M_URL, dest_dir: str = DATA_DIR, *, overwrite: bool = False
) -> pathlib.Path:
    """Download the MovieLens dataset to a local directory.

    Downloads the file at ``url`` into ``dest_dir`` and returns the full
    path to the downloaded archive. If the destination file already exists
    the download is skipped unless ``overwrite`` is True.

    Args:
        url: URL of the MovieLens archive to download.
        dest_dir: Directory where the archive will be saved.
        overwrite: If True, overwrite any existing file at the destination.

    Returns:
        pathlib.Path: Path to the downloaded archive file.

    Raises:
        httpx.HTTPError: If the HTTP request fails while downloading.
    """
    import httpx

    # prepare destination
    dest = pathlib.Path(dest_dir, pathlib.Path(url).name)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # download zip
    if not dest.exists() or overwrite:
        logger.info("downloading data: {}", url)
        # download to temp file, then move to xfmr_rec/data.py
        with httpx.stream("GET", url) as resp, tempfile.NamedTemporaryFile() as f:
            resp.raise_for_status()
            for chunk in resp.iter_bytes():
                f.write(chunk)
            pathlib.Path(f.name).rename(dest)

    logger.info("data downloaded: {}", dest)
    return dest


def unpack_data(
    archive_file: str | pathlib.Path, *, overwrite: bool = False
) -> list[str]:
    """Unpack a downloaded MovieLens archive.

    Extracts the archive into a directory next to the archive file. If the
    destination directory already exists the extraction is skipped unless
    ``overwrite`` is True.

    Args:
        archive_file: Path to the archive file to unpack.
        overwrite: If True, re-extract even if the destination directory exists.

    Returns:
        list[str]: Filenames contained in the unpacked directory.

    Raises:
        shutil.ReadError: If the archive cannot be unpacked.
    """
    archive_file = pathlib.Path(archive_file)
    dest_dir = archive_file.parent / archive_file.stem

    if not dest_dir.exists() or overwrite:
        shutil.unpack_archive(archive_file, dest_dir.parent)

    unpacked_files = [file.name for file in dest_dir.iterdir()]
    logger.info("data unpacked: {}", unpacked_files)
    return unpacked_files


def download_unpack_data(
    url: str = MOVIELENS_1M_URL, dest_dir: str = DATA_DIR, *, overwrite: bool = False
) -> list[str]:
    """Download and unpack MovieLens in a single step.

    Combine :func:`download_data` and :func:`unpack_data` to fetch and
    extract the archive, returning the list of unpacked filenames.

    Args:
        url: URL of the archive to download.
        dest_dir: Directory where the archive will be saved and unpacked.
        overwrite: If True, re-download and/or re-unpack even if files exist.

    Returns:
        list[str]: Filenames that were unpacked.
    """
    archive_file = download_data(url=url, dest_dir=dest_dir, overwrite=overwrite)
    return unpack_data(archive_file, overwrite=overwrite)


###
# load data
###


def load_items(src_dir: str = DATA_DIR) -> pl.LazyFrame:
    """Read raw MovieLens movie records and return a Polars LazyFrame.

    Read the ``movies.dat`` file from ``src_dir`` and return a
    :class:`polars.LazyFrame` where movie metadata is transformed into an
    ``item_text`` JSON string containing the title and genres.

    Args:
        src_dir: Root directory containing the unzipped MovieLens data
            (expects a ``ml-1m`` subdirectory).

    Returns:
        pl.LazyFrame: LazyFrame of items with columns ``item_id`` and
        ``item_text``.
    """
    items_dat = pathlib.Path(src_dir, "ml-1m", "movies.dat")
    dtype = {"movie_id": "str", "title": "str", "genres": "str"}
    items = (
        pd.read_csv(
            items_dat,
            sep="::",
            header=None,
            names=list(dtype.keys()),
            dtype=dtype,
            engine="python",
            encoding="iso-8859-1",
        )
        .pipe(pl.from_pandas)
        .rename({"movie_id": "item_id"})
        .with_columns(genres=pl.col("genres").str.split("|"))
        .with_columns(item_text=pl.struct("title", "genres").struct.json_encode())
        .drop("title", "genres")
    )
    logger.info("items loaded: {}, shape: {}", items_dat, items.shape)
    return items.lazy()


def load_users(src_dir: str = DATA_DIR) -> pl.LazyFrame:
    """Read raw MovieLens user records and return a Polars LazyFrame.

    Read ``users.dat`` and construct a JSON ``user_text`` field containing
    demographic information suitable for text-based models.

    Args:
        src_dir: Root directory containing the unzipped MovieLens data
            (expects a ``ml-1m`` subdirectory).

    Returns:
        pl.LazyFrame: LazyFrame of users containing ``user_id`` and
        ``user_text`` columns.
    """
    users_dat = pathlib.Path(src_dir, "ml-1m", "users.dat")
    dtype = {
        "user_id": "str",
        "gender": "str",
        "age": "int32",
        "occupation": "int32",
        "zipcode": "str",
    }

    users = (
        pd.read_csv(
            users_dat,
            sep="::",
            header=None,
            names=list(dtype.keys()),
            dtype=dtype,
            engine="python",
        )
        .pipe(pl.from_pandas)
        .with_columns(
            user_text=pl.struct(
                "gender", "age", "occupation", "zipcode"
            ).struct.json_encode()
        )
        .drop("gender", "age", "occupation", "zipcode")
    )
    logger.info("users loaded: {}, shape: {}", users_dat, users.shape)
    return users.lazy()


def load_events(src_dir: str = DATA_DIR) -> pl.LazyFrame:
    """Read MovieLens rating events and return a Polars LazyFrame.

    Read ``ratings.dat`` and convert it to a :class:`polars.LazyFrame` with
    columns ``user_id``, ``item_id``, ``event_value``, ``datetime``,
    ``event_name`` and a boolean ``label`` column indicating a positive
    interaction.

    Args:
        src_dir: Root directory containing the unzipped MovieLens data
            (expects a ``ml-1m`` subdirectory).

    Returns:
        pl.LazyFrame: LazyFrame of event records.
    """
    events_dat = pathlib.Path(src_dir, "ml-1m", "ratings.dat")
    dtype = {
        "user_id": "str",
        "movie_id": "str",
        "rating": "int32",
        "timestamp": "int32",
    }
    events = (
        pd.read_csv(
            events_dat,
            sep="::",
            header=None,
            names=list(dtype.keys()),
            dtype=dtype,
            engine="python",
        )
        .pipe(pl.from_pandas)
        .rename({"movie_id": "item_id", "rating": "event_value"})
        .with_columns(
            datetime=pl.from_epoch("timestamp"),
            event_name=pl.lit("rating"),
            label=pl.lit(value=True),
        )
    )
    logger.info("events loaded: {}, shape: {}", events_dat, events.shape)
    return events.lazy()


###
# process data
###


def train_test_split(
    events: pl.LazyFrame,
    *,
    group_col: str = "user_id",
    order_col: str = "datetime",
    train_prop: float = 0.8,
    val_prop: float = 0.2,
) -> pl.LazyFrame:
    """Split events into train / validation / test per user by time.

    Mark interactions as train/validation/test per user based on chronological
    ordering. For each user the earliest ``train_prop`` fraction of events is
    marked as training; the remaining events are split between validation and
    test according to ``val_prop`` (users with more leftover events are more
    likely to contribute validation examples).

    Args:
        events: LazyFrame of event records containing the grouping and
            ordering columns specified by ``group_col`` and ``order_col``.
        group_col: Column used to group events (typically ``user_id``).
        order_col: Column used to order events within each group
            (typically ``datetime``).
        train_prop: Fraction of each user's earliest interactions reserved
            for training (0 < train_prop < 1).
        val_prop: Fraction of the non-training interactions to mark as
            validation (0 <= val_prop <= 1).

    Returns:
        pl.LazyFrame: The input LazyFrame augmented with boolean columns
        ``is_train``, ``is_val``, ``is_test``, and ``is_predict``.

    Raises:
        ValueError: If proportions are out of bounds.
    """
    events = (
        events.lazy()
        .with_columns(
            p=((pl.col(order_col).rank("min") - 1) / pl.count(order_col)).over(
                group_col
            ),
        )
        # first train_prop will be train set
        .with_columns(is_train=pl.col("p") < train_prop)
        .drop("p")
    )
    users_split = (
        events.filter(~pl.col("is_train"))
        .group_by(group_col)
        .len()
        .with_columns(p=((pl.col("len").rank("min") - 1) / pl.count("len")))
        # largest val_prop by count will be val set
        .with_columns(is_val=pl.col("p") >= 1 - val_prop)
        .drop("len", "p")
    )
    return events.join(
        users_split, on=group_col, how="left", validate="m:1"
    ).with_columns(
        is_val=~pl.col("is_train") & pl.col("is_val"),
        is_test=~pl.col("is_train") & ~pl.col("is_val"),
        is_predict=True,
    )


def process_events(
    events: pl.LazyFrame,
    items: pl.LazyFrame,
    users: pl.LazyFrame,
    *,
    src_dir: str = DATA_DIR,
    overwrite: bool = False,
) -> pl.LazyFrame:
    """Join events with item and user metadata and persist the result as Parquet.

    If a processed Parquet file already exists at ``src_dir/ml-1m/events.parquet``
    and ``overwrite`` is False the function will return a LazyFrame that
    scans the existing file. Otherwise the joins are computed, written to
    disk, and a scanning LazyFrame is returned.

    Args:
        events: LazyFrame of raw events to be joined.
        items: LazyFrame of item metadata to join on ``item_id``.
        users: LazyFrame of user metadata to join on ``user_id``.
        src_dir: Directory containing the MovieLens dataset and where
            processed Parquet files will be written.
        overwrite: If True, recompute and overwrite any existing parquet files.

    Returns:
        pl.LazyFrame: LazyFrame that scans the processed events parquet file.

    Raises:
        IOError: If writing the parquet file fails.
    """
    events_parquet = pathlib.Path(src_dir, "ml-1m", "events.parquet")
    if events_parquet.exists() and not overwrite:
        events_processed = pl.scan_parquet(events_parquet)
        logger.info("events loaded: {}", events_parquet)
        return events_processed

    events_processed = (
        events.lazy()
        .join(items.lazy(), on="item_id", how="left", validate="m:1")
        .join(users.lazy(), on="user_id", how="left", validate="m:1")
        .collect()
    )

    events_processed.write_parquet(events_parquet)
    logger.info("events saved: {}, shape: {}", events_parquet, events_processed.shape)
    return pl.scan_parquet(events_parquet)


def process_items(
    items: pl.LazyFrame,
    events: pl.LazyFrame,
    *,
    src_dir: str = DATA_DIR,
    overwrite: bool = False,
) -> pl.LazyFrame:
    """Process item metadata and persist as Parquet.

    Joins item metadata with information about whether the item appears in the
    training split and writes the result to ``items.parquet`` under ``src_dir``.

    Args:
        items: LazyFrame of raw item metadata.
        events: LazyFrame of processed events (used to determine training presence).
        src_dir: Directory where the processed Parquet should be written.
        overwrite: If True, recompute and overwrite the parquet file.

    Returns:
        pl.LazyFrame: LazyFrame that scans the processed items parquet file.

    Raises:
        IOError: If writing the parquet file fails.
    """
    items_parquet = pathlib.Path(src_dir, "ml-1m", "items.parquet")
    if items_parquet.exists() and not overwrite:
        items_processed = pl.scan_parquet(items_parquet)
        logger.info("items loaded: {}", items_parquet)
        return items_processed

    items_train = events.lazy().group_by("item_id").agg(pl.any("is_train"))
    items_processed = (
        items.lazy()
        .join(items_train, on="item_id", how="left", validate="1:1")
        .with_columns(is_val=True, is_test=True, is_predict=True)
        .collect()
    )

    items_processed.write_parquet(items_parquet)
    logger.info("items saved: {}, shape: {}", items_parquet, items_processed.shape)
    return pl.scan_parquet(items_parquet)


def process_users(
    users: pl.LazyFrame,
    events: pl.LazyFrame,
    *,
    src_dir: str = DATA_DIR,
    overwrite: bool = False,
) -> pl.LazyFrame:
    """Aggregate user interactions and persist user-level Parquet.

    Build per-user history and target lists from the events LazyFrame and
    join this information with the user metadata. The output is written to
    ``users.parquet`` unless it already exists and ``overwrite`` is False.

    Args:
        users: LazyFrame of raw user metadata.
        events: LazyFrame of processed events with split information.
        src_dir: Destination directory for saved Parquet.
        overwrite: If True, recompute and overwrite the parquet file.

    Returns:
        pl.LazyFrame: LazyFrame that scans the processed users parquet file.

    Raises:
        IOError: If writing the parquet file fails.
    """
    users_parquet = pathlib.Path(src_dir, "ml-1m", "users.parquet")
    if users_parquet.exists() and not overwrite:
        users_processed = pl.scan_parquet(users_parquet)
        logger.info("users loaded: {}", users_parquet)
        return users_processed

    activity_cols = [
        "datetime",
        "event_name",
        "event_value",
        "label",
        "item_id",
        "item_text",
    ]
    users_interactions = (
        events.lazy()
        .group_by("user_id")
        .agg(
            history=pl.struct(*activity_cols).filter("is_train"),
            target=pl.struct(*activity_cols).filter(~pl.col("is_train")),
            is_train=pl.any("is_train"),
            is_val=pl.any("is_val"),
            is_test=pl.any("is_test"),
            is_predict=pl.any("is_predict"),
        )
        .with_columns(
            pl.col(col).list.sort().alias(col) for col in ["history", "target"]
        )
        .with_columns(
            pl.struct(
                pl.col(col)
                .list.eval(  # devskim: ignore DS189424
                    pl.element().struct.field(field)
                )
                .alias(field)
                for field in activity_cols
            ).alias(col)
            for col in ["history", "target"]
        )
    )
    users_processed = (
        users.lazy()
        .join(users_interactions, on="user_id", how="left", validate="1:1")
        .collect()
    )

    users_processed.write_parquet(users_parquet)
    logger.info("users saved: {}, shape: {}", users_parquet, users_processed.shape)
    return pl.scan_parquet(users_parquet)


def prepare_movielens(
    src_dir: str = DATA_DIR, *, overwrite: bool = False
) -> pl.LazyFrame:
    """High-level helper to prepare MovieLens artifacts.

    Loads raw files, computes train/val/test splits, processes joins, and
    writes/returns the processed Parquet-backed LazyFrames for events, items,
    and users.

    Args:
        src_dir: Directory containing or to contain the MovieLens data.
        overwrite: If True, force recomputation and overwrite any existing outputs.

    Returns:
        pl.LazyFrame: The processed events LazyFrame (scanning the saved Parquet file).
    """
    items = load_items(src_dir)
    users = load_users(src_dir)
    events = load_events(src_dir).pipe(train_test_split)

    events = process_events(events, items, users, src_dir=src_dir, overwrite=overwrite)
    items = process_items(items, events, src_dir=src_dir, overwrite=overwrite)
    users = process_users(users, events, src_dir=src_dir, overwrite=overwrite)
    return events


###
# Sequential Dataset and DataModule
###

NumpyBoolArray = np.typing.NDArray[np.bool_]
NumpyIntArray = np.typing.NDArray[np.int_]
NumpyStrArray = np.typing.NDArray[np.str_]


class SeqExample(TypedDict):
    history_item_idx: torch.Tensor
    pos_item_idx: torch.Tensor
    neg_item_idx: torch.Tensor
    history_item_text: list[str]
    pos_item_text: list[str]
    neg_item_text: list[str]


class SeqBatch(TypedDict):
    history_item_idx: torch.Tensor
    pos_item_idx: torch.Tensor
    neg_item_idx: torch.Tensor
    history_item_text: list[list[str]]
    pos_item_text: list[list[str]]
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
        labels = labels[mask]
        # trim away events after the last positive label
        # for history with no positive, empty array is returned and filtered away later
        max_idx = np.flatnonzero(labels).max(initial=-1) + 1
        return {
            "history_item_idx": item_idx[:max_idx],
            "history_label": labels[:max_idx],
        }

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
        return {key: batch[key].repeat(num_copies) for key in batch}

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
            .filter(lambda row: len(row["history_item_idx"]) > 0)
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
        row: dict[str, np.ndarray] = self.events_dataset[idx]
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
            "pos_item_idx": torch.as_tensor(pos_item_idx),
            "neg_item_idx": torch.as_tensor(neg_item_idx),
            "history_item_text": self.item_texts[
                (history_item_idx[sampled_indices] - 1).tolist()
            ],
            "pos_item_text": self.item_texts[(pos_item_idx - 1).tolist()],
            "neg_item_text": self.item_texts[(neg_item_idx - 1).tolist()],
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
            assert self.items_dataset is not None
            train_dataset: datasets.Dataset = datasets.Dataset.from_parquet(
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
        dataset: datasets.Dataset | SeqDataset,
        *,
        shuffle: bool = False,
        batch_size: int | None = None,
        collate_fn: Callable[[list[SeqExample]], SeqBatch] | None = None,
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
        assert self.train_dataset is not None
        return self.get_dataloader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.config.batch_size,
            collate_fn=self.train_dataset.collate,
        )

    def val_dataloader(self) -> torch_data.DataLoader[dict[str, torch.Tensor]]:
        assert self.val_dataset is not None
        return self.get_dataloader(self.val_dataset)

    def test_dataloader(self) -> torch_data.DataLoader[dict[str, torch.Tensor]]:
        assert self.test_dataset is not None
        return self.get_dataloader(self.test_dataset)

    def predict_dataloader(self) -> torch_data.DataLoader[dict[str, torch.Tensor]]:
        assert self.predict_dataset is not None
        return self.get_dataloader(self.predict_dataset)


def main(data_dir: str = DATA_DIR, *, overwrite: bool = True) -> None:
    """Command-line entrypoint to prepare the MovieLens dataset.

    Downloads and unpacks the dataset, runs the full preprocessing pipeline,
    and prints a small summary of the processed events to stdout. This
    function is intended to be used via the module's CLI invocation.

    Args:
        data_dir: Directory to download/unpack and store processed outputs.
        overwrite: If True, force re-download and re-processing of the dataset.
    """
    download_unpack_data(overwrite=overwrite)
    with pl.StringCache():
        prepare_movielens(data_dir, overwrite=overwrite).head().collect().glimpse()


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main, as_positional=False)
