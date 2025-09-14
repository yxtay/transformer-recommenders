from __future__ import annotations

import datetime
import functools
import pathlib
from typing import TYPE_CHECKING, TypedDict, TypeVar

import pydantic
import torch
import torch.utils.data as torch_data
from lightning import LightningDataModule

from xfmr_rec.data.load import embed_example, select_fields, torch_collate
from xfmr_rec.params import (
    BATCH_SIZE,
    DATA_DIR,
    ITEM_ID_COL,
    ITEM_IDX_COL,
    ITEM_TEXT_COL,
    ITEMS_TABLE_NAME,
    LANCE_DB_PATH,
    MOVIELENS_1M_URL,
    TARGET_COL,
    TOP_K,
    USER_ID_COL,
    USER_IDX_COL,
    USER_TEXT_COL,
    USERS_TABLE_NAME,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    import lancedb
    import numpy as np
    import numpy.typing as npt
    import pandas as pd
    import polars as pl

FT = TypeVar("FT")
BT = TypeVar("BT")


class ItemFeaturesType(TypedDict):
    idx: int
    text: str


class ItemBatchType(TypedDict):
    idx: torch.Tensor
    text: list[str]


class UserFeaturesType(TypedDict):
    pos_idx: torch.Tensor
    text: str


class UserBatchType(TypedDict):
    pos_idx: torch.Tensor
    text: list[str]


class InteractionFeaturesType(TypedDict):
    target: int
    user: UserFeaturesType
    item: ItemFeaturesType
    neg_item: ItemFeaturesType


class InteractionBatchType(TypedDict):
    target: torch.Tensor
    user: UserBatchType
    item: ItemBatchType
    neg_item: ItemBatchType


class FeaturesProcessor[FT, BT](pydantic.BaseModel):
    batch_size: int = BATCH_SIZE
    data_dir: str = DATA_DIR

    @property
    def data_path(self) -> str:
        raise NotImplementedError

    @property
    def fields(self) -> list[str]:
        raise NotImplementedError

    def collate(self, batch: list[FT]) -> BT:
        return torch_collate.default_collate(batch)

    def get_data(self, subset: str, cycle: int = 1) -> torch_data.IterDataPipe[FT]:
        import pyarrow.dataset as ds

        from xfmr_rec.data.load import ParquetDictLoaderIterDataPipe

        valid_subset = {"train", "val", "test", "predict"}
        if subset not in valid_subset:
            msg = f"`{subset}` is not one of `{valid_subset}`"
            raise ValueError(msg)

        filter_expr = ds.field(f"is_{subset}")
        return (
            ParquetDictLoaderIterDataPipe([self.data_path], filter_expr=filter_expr)
            .cycle(count=cycle)
            .shuffle()  # devskim: ignore DS148264
            .sharding_filter()
        )

    def get_processed_data(
        self, subset: str, cycle: int = 1
    ) -> torch_data.IterDataPipe[FT]:
        return self.get_data(subset, cycle=cycle).map(self.process)

    def get_batch_data(
        self, subset: str, cycle: int = 1
    ) -> torch_data.IterDataPipe[BT]:
        return (
            self.get_processed_data(subset, cycle=cycle)
            .map(functools.partial(select_fields, fields=self.fields))
            .batch(self.batch_size)
            .collate(collate_fn=self.collate)
        )


class LanceDbProcessor(pydantic.BaseModel):
    idx_col: str
    id_col: str
    text_col: str
    lance_table_name: str
    lance_db_path: str = LANCE_DB_PATH

    @property
    def lance_db(self) -> lancedb.DBConnection:
        import lancedb

        return lancedb.connect(self.lance_db_path)

    @property
    def lance_table(self) -> lancedb.table.Table:
        return self.lance_db.open_table(self.lance_table_name)

    def get_id(self, id_val: str | None) -> dict[str, Any]:
        if id_val is None:
            return {}
        result = (
            self.lance_table.search().where(f"{self.id_col} = '{id_val}'").to_list()
        )
        if len(result) == 0:
            return {}
        return result[0]


class ItemProcessor(
    FeaturesProcessor[ItemFeaturesType, ItemBatchType], LanceDbProcessor
):
    idx_col: str = ITEM_IDX_COL
    id_col: str = ITEM_ID_COL
    text_col: str = ITEM_TEXT_COL
    lance_table_name: str = ITEMS_TABLE_NAME

    num_partitions: int | None = None
    num_sub_vectors: int | None = None
    num_probes: int = 8
    refine_factor: int = 4

    @property
    def data_path(self) -> str:
        return pathlib.Path(self.data_dir, "ml-1m", "items.parquet").as_posix()

    def process(self, example: dict[str, Any]) -> ItemFeaturesType:
        return {
            **example,
            "text": example[self.text_col],
            "idx": example[self.idx_col],
        }

    @property
    def fields(self) -> list[str]:
        return ["text", "idx"]

    def get_index(
        self, model: torch.nn.Module, subset: str = "predict"
    ) -> lancedb.table.Table:
        import math

        import pyarrow as pa

        fields = [self.idx_col, self.id_col, self.text_col, "embedding"]
        dp = (
            self.get_processed_data(subset)
            .map(functools.partial(torch.inference_mode(embed_example), model=model))
            .map(functools.partial(select_fields, fields=fields))
            .batch(self.batch_size)
        )

        batch = next(iter(dp))
        num_items = len(dp) * len(batch)
        example = batch[0]
        (embedding_dim,) = example["embedding"].shape

        # rule of thumb: nlist ~= 4 * sqrt(n_vectors)
        num_partitions = self.num_partitions or 2 ** int(math.log2(num_items) / 2)
        num_sub_vectors = self.num_sub_vectors or embedding_dim // 8

        accelerator = "cuda" if torch.cuda.is_available() else None

        schema = pa.RecordBatch.from_pylist(batch).schema
        schema = schema.set(
            schema.get_field_index("embedding"),
            pa.field("embedding", pa.list_(pa.float32(), embedding_dim)),
        )

        table = self.lance_db.create_table(
            self.lance_table_name,
            data=iter(dp.map(pa.RecordBatch.from_pylist)),
            schema=schema,
            mode="overwrite",
        )
        table.create_scalar_index(self.id_col)
        table.create_fts_index(self.text_col)
        table.create_index(
            vector_column_name="embedding",
            metric="cosine",
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors,
            index_type="IVF_HNSW_PQ",
            accelerator=accelerator,
        )
        table.optimize(
            cleanup_older_than=datetime.timedelta(days=0),
            delete_unverified=True,
            retrain=True,
        )
        return table

    def search(
        self,
        embedding: npt.NDArray[np.float64],
        exclude_item_ids: list[str] | None = None,
        top_k: int = TOP_K,
    ) -> pd.DataFrame:
        if self.lance_table is None:
            msg = "`index` must be intialised first"
            raise ValueError(msg)

        exclude_item_ids = exclude_item_ids or ["0"]
        exclude_filter = ", ".join(f"'{item}'" for item in exclude_item_ids)
        exclude_filter = f"{self.id_col} NOT IN ({exclude_filter})"
        return (
            self.lance_table.search(embedding)
            .where(exclude_filter, prefilter=True)
            .nprobes(self.num_probes)
            .refine_factor(self.refine_factor)
            .limit(top_k)
            .to_pandas()
            .assign(score=lambda df: 1 - df["_distance"])
            .drop(columns="_distance")
        )


class UserProcessor(
    FeaturesProcessor[UserFeaturesType, UserBatchType], LanceDbProcessor
):
    idx_col: str = USER_IDX_COL
    id_col: str = USER_ID_COL
    text_col: str = USER_TEXT_COL
    lance_table_name: str = USERS_TABLE_NAME

    @property
    def data_path(self) -> str:
        return pathlib.Path(self.data_dir, "ml-1m", "users.parquet").as_posix()

    def process(self, example: dict[str, Any]) -> UserFeaturesType:
        return {
            **example,
            "text": example[self.text_col],
            "pos_idx": torch.as_tensor(
                example["target"][ITEM_IDX_COL], dtype=torch.int64
            ),
        }

    @property
    def fields(self) -> list[str]:
        return ["text", "pos_idx"]

    def get_index(self, subset: str = "predict") -> lancedb.table.Table:
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq

        columns = [self.idx_col, self.id_col, self.text_col, "history", "target"]
        filters = ds.field(f"is_{subset}")
        pa_table = pq.read_table(self.data_path, columns=columns, filters=filters)

        table = self.lance_db.create_table(
            self.lance_table_name, data=pa_table, mode="overwrite"
        )
        table.create_scalar_index(self.id_col)
        table.create_fts_index(self.text_col)
        table.optimize(
            cleanup_older_than=datetime.timedelta(days=0),
            delete_unverified=True,
            retrain=True,
        )
        return table

    def get_activity(self, id_val: str | None, activity_name: str) -> dict[str, int]:
        activity = self.get_id(id_val).get(activity_name, {})
        return dict(zip(activity[ITEM_ID_COL], activity[TARGET_COL], strict=False))


class InteractionProcessor(
    FeaturesProcessor[InteractionFeaturesType, InteractionBatchType]
):
    item_processor: ItemProcessor
    user_processor: UserProcessor

    target_col: str = TARGET_COL
    batch_size: int = BATCH_SIZE
    data_dir: str = DATA_DIR

    @property
    def data_path(self) -> str:
        return pathlib.Path(self.data_dir, "ml-1m", "events.parquet").as_posix()

    def process(self, example: dict[str, Any]) -> InteractionFeaturesType:
        item_features = select_fields(
            self.item_processor.process(example), fields=self.item_processor.fields
        )
        user_features = select_fields(
            self.user_processor.process(example), fields=self.user_processor.fields
        )
        target = example[self.target_col]
        return {"target": target, "user": user_features, "item": item_features}

    def collate(self, batch: list[InteractionFeaturesType]) -> InteractionBatchType:
        target = torch_collate.default_collate([example["target"] for example in batch])
        item = self.item_processor.collate([example["item"] for example in batch])
        user = self.user_processor.collate([example["user"] for example in batch])
        neg_item = self.item_processor.collate(
            [example["neg_item"] for example in batch]
        )
        return {"target": target, "item": item, "user": user, "neg_item": neg_item}

    def get_processed_data(
        self, subset: str
    ) -> torch_data.IterDataPipe[InteractionFeaturesType]:
        from xfmr_rec.data.load import merge_examples, nest_example

        neg_item_dp = (
            self.item_processor.get_processed_data(subset, cycle=0)
            .map(functools.partial(select_fields, fields=self.item_processor.fields))
            .map(functools.partial(nest_example, key="neg_item"))
        )
        return super().get_processed_data(subset).zip(neg_item_dp).map(merge_examples)

    def get_batch_data(
        self, subset: str
    ) -> torch_data.IterDataPipe[InteractionBatchType]:
        return (
            self.get_processed_data(subset)
            .batch(self.batch_size)
            .collate(collate_fn=self.collate)
        )


class MatrixFactorizationDataConfig(pydantic.BaseModel):
    data_dir: str = DATA_DIR
    batch_size: int = BATCH_SIZE
    num_partitions: int | None = None
    num_sub_vectors: int | None = None
    num_probes: int = 8
    refine_factor: int = 4
    num_workers: int | None = 1


class MatrixFactorizationDataModule(LightningDataModule):
    def __init__(
        self,
        config: MatrixFactorizationDataConfig,
    ) -> None:
        super().__init__()
        self.config = MatrixFactorizationDataConfig.model_validate(config)
        self.save_hyperparameters(self.config.model_dump())

        self.item_processor = ItemProcessor(
            batch_size=self.config.batch_size,
            num_partitions=self.config.num_partitions,
            num_sub_vectors=self.config.num_sub_vectors,
            num_probes=self.config.num_probes,
            refine_factor=self.config.refine_factor,
            data_dir=self.config.data_dir,
        )
        self.user_processor = UserProcessor(
            batch_size=self.config.batch_size,
            data_dir=self.config.data_dir,
        )
        self.interaction_processor = InteractionProcessor(
            item_processor=self.item_processor,
            user_processor=self.user_processor,
            batch_size=self.config.batch_size,
            data_dir=self.config.data_dir,
        )

    def prepare_data(self, *, overwrite: bool = False) -> pl.LazyFrame:
        from filelock import FileLock

        from xfmr_rec.data.prepare import download_unpack_data, prepare_movielens

        data_dir = self.config.data_dir
        with FileLock(f"{data_dir}.lock"):
            download_unpack_data(MOVIELENS_1M_URL, data_dir, overwrite=overwrite)
            return prepare_movielens(data_dir, overwrite=overwrite)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_data = self.interaction_processor.get_processed_data("train")

        if stage in {"fit", "validate"}:
            self.val_data = self.user_processor.get_processed_data("val")

        if stage == "test":
            self.test_data = self.user_processor.get_processed_data("test")

        if stage == "predict":
            self.predict_data = self.user_processor.get_processed_data("predict")

    def get_dataloader(
        self,
        dataset: torch_data.Dataset[FT],
        *,
        batch_size: int | None = None,
        collate_fn: Callable[[list[FT]], BT] | None = None,
        shuffle: bool = False,
    ) -> torch_data.DataLoader[BT]:
        import os

        num_workers = self.config.num_workers

        if num_workers is None:
            num_workers = os.cpu_count() or 1

        multiprocessing_context = "spawn" if num_workers > 0 else None

        return torch_data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            multiprocessing_context=multiprocessing_context,
            persistent_workers=num_workers > 0,
        )

    def train_dataloader(self) -> torch_data.DataLoader[InteractionBatchType]:
        batch_size = self.config.batch_size
        return self.get_dataloader(
            self.train_data,
            batch_size=batch_size,
            collate_fn=self.interaction_processor.collate,
            shuffle=True,
        )

    def val_dataloader(self) -> torch_data.DataLoader[ItemFeaturesType]:
        return self.get_dataloader(self.val_data)

    def test_dataloader(self) -> torch_data.DataLoader[ItemFeaturesType]:
        return self.get_dataloader(self.test_data)

    def predict_dataloader(self) -> torch_data.DataLoader[ItemFeaturesType]:
        return self.get_dataloader(self.predict_data)


if __name__ == "__main__":
    import rich

    dm = MatrixFactorizationDataModule(MatrixFactorizationDataConfig())
    dm.prepare_data().head().collect().glimpse()
    dm.setup("fit")

    dataloaders = [
        dm.item_processor.get_batch_data("train"),
        dm.user_processor.get_batch_data("train"),
        dm.interaction_processor.get_processed_data("train"),
        dm.interaction_processor.get_batch_data("train"),
        dm.train_dataloader(),
        dm.val_dataloader(),
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

    dm.user_processor.get_index().search().to_polars().glimpse()
    rich.print(dm.user_processor.get_id("1"))
    rich.print(dm.user_processor.get_activity("1", "history"))
    rich.print(dm.user_processor.get_activity("1", "target"))

    dm.item_processor.get_index(
        lambda _: torch.rand(1, 32)  # devskim: ignore DS148264
    ).search().to_polars().glimpse()
    rich.print(dm.item_processor.get_id("1"))
    query_vector = torch.rand(1, 32).numpy()  # devskim: ignore DS148264
    rich.print(dm.item_processor.search(query_vector, top_k=5))
