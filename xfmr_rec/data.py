from __future__ import annotations

import pathlib
import shutil
import tempfile

import pandas as pd
import polars as pl
from loguru import logger

from xfmr_rec.params import DATA_DIR, MOVIELENS_1M_URL

###
# download data
###


def download_data(
    url: str = MOVIELENS_1M_URL, dest_dir: str = DATA_DIR, *, overwrite: bool = False
) -> pathlib.Path:
    """Download the MovieLens dataset to a local directory.

    Downloads the file at ``url`` into ``dest_dir`` and returns the full path
    to the downloaded archive. The download is skipped if the destination
    file already exists unless ``overwrite`` is True.

    Args:
        url: The URL of the MovieLens archive to download.
        dest_dir: Directory to save the downloaded archive into.
        overwrite: If True, overwrite any existing file at the destination.

    Returns:
        Path to the downloaded archive file.
    """
    import httpx

    # prepare destination
    dest = pathlib.Path(dest_dir, pathlib.Path(url).name)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # download zip
    if not dest.exists() or overwrite:
        logger.info("downloading data: {}", url)
        # download to temp file, then move to dest
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
    target directory already exists the extraction is skipped unless
    ``overwrite`` is True.

    Args:
        archive_file: Path to the archive file to unpack.
        overwrite: If True, re-extract even if the destination directory exists.

    Returns:
        Names of files contained in the unpacked directory.
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

    Combines :func:`download_data` and :func:`unpack_data` to fetch and
    extract the archive, returning the list of unpacked files.

    Args:
        url: URL of the archive to download.
        dest_dir: Directory where the archive will be saved and unpacked.
        overwrite: If True, re-download and/or re-unpack even if files exist.

    Returns:
        Filenames that were unpacked.
    """
    archive_file = download_data(url=url, dest_dir=dest_dir, overwrite=overwrite)
    return unpack_data(archive_file, overwrite=overwrite)


###
# load data
###


def load_items(src_dir: str = DATA_DIR) -> pl.LazyFrame:
    """Read raw MovieLens movie records and return a Polars LazyFrame.

    The function reads the ``movies.dat`` file from ``src_dir`` and returns a
    :class:`polars.LazyFrame` where movie metadata has been transformed into
    an ``item_text`` JSON string containing title and genres.

    Args:
        src_dir: Root directory containing the unzipped MovieLens data (``ml-1m``).

    Returns:
        LazyFrame of items with columns ``item_id`` and ``item_text``.
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

    Reads ``users.dat`` and constructs a JSON ``user_text`` field containing
    demographic information suitable for text-based models.

    Args:
        src_dir: Root directory containing the unzipped MovieLens data (``ml-1m``).

    Returns:
        LazyFrame of users with a ``user_text`` column.
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

    The function reads ``ratings.dat`` and converts it to a LazyFrame with
    columns ``user_id``, ``item_id``, ``event_value``, ``datetime``, and a
    boolean ``label`` column indicating a positive interaction.

    Args:
        src_dir: Root directory containing the unzipped MovieLens data (``ml-1m``).

    Returns:
        LazyFrame of event records.
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

    This function marks each interaction as part of the training set based
    on its relative position in a user's timeline. The remaining interactions
    are split across validation and test such that users with more leftover
    interactions are more likely to have validation examples.

    Args:
        events: LazyFrame of event records containing the grouping and ordering
            columns specified by ``group_col`` and ``order_col``.
        group_col: Column used to group events (typically ``user_id``).
        order_col: Column used to order events within each group
            (typically ``datetime``).
        train_prop: Fraction of each user's earliest interactions reserved for training.
        val_prop: Fraction of the non-training interactions to mark as validation.

    Returns:
        The same events LazyFrame augmented with boolean columns ``is_train``,
        ``is_val``, ``is_test``, and ``is_predict``.
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
    """Join events with item and user metadata and persist as Parquet.

    If a processed Parquet file already exists at ``src_dir/ml-1m/events.parquet``
    the function will return a scanned LazyFrame pointing at it (unless
    ``overwrite`` is True). Otherwise it computes the joins, writes the
    Parquet file, and returns a LazyFrame that scans the saved file.

    Args:
        events: LazyFrame of raw events to be joined.
        items: LazyFrame of item metadata to join on ``item_id``.
        users: LazyFrame of user metadata to join on ``user_id``.
        src_dir: Directory containing the MovieLens dataset and where processed
            Parquet files will be written.
        overwrite: If True, recompute and overwrite any existing parquet files.

    Returns:
        LazyFrame pointing at the processed events parquet file.
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
        LazyFrame pointing to the processed items parquet file.
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

    Builds per-user history and target lists from the events LazyFrame and
    joins this information with the user metadata. The output is written to
    ``users.parquet`` unless it already exists and ``overwrite`` is False.

    Args:
        users: LazyFrame of raw user metadata.
        events: LazyFrame of processed events with split information.
        src_dir: Destination directory for saved Parquet.
        overwrite: If True, recompute and overwrite the parquet file.

    Returns:
        LazyFrame pointing to the processed users parquet file.
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
        The processed events LazyFrame (scanning the saved Parquet file).
    """
    items = load_items(src_dir)
    users = load_users(src_dir)
    events = load_events(src_dir).pipe(train_test_split)

    events = process_events(events, items, users, src_dir=src_dir, overwrite=overwrite)
    items = process_items(items, events, src_dir=src_dir, overwrite=overwrite)
    users = process_users(users, events, src_dir=src_dir, overwrite=overwrite)
    return events


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
    from jsonargparse import CLI

    CLI(main, as_positional=False)
