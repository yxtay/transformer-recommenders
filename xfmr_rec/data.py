from __future__ import annotations

import pathlib

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
    import tempfile

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
    import shutil

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
    archive_file = download_data(url=url, dest_dir=dest_dir, overwrite=overwrite)
    return unpack_data(archive_file, overwrite=overwrite)


###
# load data
###


def load_items(src_dir: str = DATA_DIR) -> pl.LazyFrame:
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
        .with_row_index("item_rn", offset=1)
        .with_columns(genres=pl.col("genres").str.split("|"))
        .with_columns(item_text=pl.struct("title", "genres").struct.json_encode())
        .drop("title", "genres")
    )
    logger.info("items loaded: {}, shape: {}", items_dat, items.shape)
    return items.lazy()


def load_users(src_dir: str = DATA_DIR) -> pl.LazyFrame:
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
        .with_row_index("user_rn", offset=1)
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
        .with_columns(datetime=pl.from_epoch("timestamp"))
        .with_columns(event_name=pl.lit("rating"))
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
    from concurrent.futures import ThreadPoolExecutor

    events_parquet = pathlib.Path(src_dir, "ml-1m", "events.parquet")
    if events_parquet.exists() and not overwrite:
        events_processed = pl.scan_parquet(events_parquet)
        logger.info("events loaded: {}", events_parquet)
        return events_processed

    events_processed = (
        events.lazy()
        .join(items.lazy(), on="item_id", how="left", validate="m:1")
        .join(users.lazy(), on="user_id", how="left", validate="m:1")
        .sort(["user_id", "datetime"])
    )

    with ThreadPoolExecutor() as executor:
        for _, df in events_processed.collect().group_by("user_id"):
            executor.submit(gather_history, events=df.lazy(), path=events_parquet)

    events_processed = pl.scan_parquet(events_parquet)
    n_row = events_processed.select(pl.len()).collect().item()
    n_col = events_processed.collect_schema().len()
    logger.info("events saved: {}, shape: {}", events_parquet, (n_row, n_col))
    return events_processed


def gather_history(events: pl.LazyFrame, *, path: pathlib.Path) -> pl.LazyFrame:
    activity_cols = [
        "datetime",
        "event_name",
        "event_value",
        "item_rn",
        "item_id",
        "item_text",
    ]
    events_history = (
        events.rolling("datetime", period="4w", closed="none", group_by="user_id")
        .agg(history=pl.struct(*activity_cols))
        .with_columns(history=pl.col("history").list.sort())
        .with_columns(
            history=pl.struct(
                pl.col("history")
                .list.eval(  # devskim: ignore DS189424
                    pl.element().struct.field(col)
                )
                .alias(col)
                for col in activity_cols
            )
        )
        .unique(["user_id", "datetime"])
    )
    events_target = (
        events.group_by("user_id", "is_train")
        .agg(target=pl.struct(*activity_cols))
        .with_columns(target=pl.col("target").list.sort())
        .with_columns(
            target=pl.struct(
                pl.col("target")
                .list.eval(  # devskim: ignore DS189424
                    pl.element().struct.field(col)
                )
                .alias(col)
                for col in activity_cols
            )
        )
    )
    events_history = events.join(
        events_history, on=["user_id", "datetime"], validate="m:1"
    ).join(events_target, on=["user_id", "is_train"], validate="m:1")
    events_history.collect().write_parquet(path, partition_by="user_id")
    return events_history


def process_items(
    items: pl.LazyFrame,
    events: pl.LazyFrame,
    *,
    src_dir: str = DATA_DIR,
    overwrite: bool = False,
) -> pl.LazyFrame:
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
    users_parquet = pathlib.Path(src_dir, "ml-1m", "users.parquet")
    if users_parquet.exists() and not overwrite:
        users_processed = pl.scan_parquet(users_parquet)
        logger.info("users loaded: {}", users_parquet)
        return users_processed

    activity_cols = [
        "datetime",
        "event_name",
        "event_value",
        "item_rn",
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
    items = load_items(src_dir)
    users = load_users(src_dir)
    events = load_events(src_dir).pipe(train_test_split)

    events = process_events(events, items, users, src_dir=src_dir, overwrite=overwrite)
    items = process_items(items, events, src_dir=src_dir, overwrite=overwrite)
    users = process_users(users, events, src_dir=src_dir, overwrite=overwrite)
    return events


def main(data_dir: str = DATA_DIR, *, overwrite: bool = True) -> None:
    download_unpack_data(overwrite=overwrite)
    with pl.StringCache():
        prepare_movielens(data_dir, overwrite=overwrite).head().collect().glimpse()


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main, as_positional=False)
