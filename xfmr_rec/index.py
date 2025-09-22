from __future__ import annotations

import datetime
import math
import pathlib
from typing import TYPE_CHECKING, Any

import datasets
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pydantic
from loguru import logger

from xfmr_rec.params import ITEMS_TABLE_NAME, LANCE_DB_PATH

if TYPE_CHECKING:
    import lancedb
    import numpy as np


class IndexConfig(pydantic.BaseModel):
    id_col: str = "item_id"
    embedding_col: str | None = None


class LanceIndexConfig(IndexConfig):
    lancedb_path: str = LANCE_DB_PATH
    table_name: str = ITEMS_TABLE_NAME
    text_col: str = "item_text"


class LanceIndex:
    def __init__(
        self, config: LanceIndexConfig, table: lancedb.table.Table | None = None
    ) -> None:
        super().__init__()
        self.config = config
        self.table = table

    @classmethod
    def load(cls, config: LanceIndexConfig) -> LanceIndex:
        self = cls(config)
        self.open_table()

        for index in self.table.list_indices():
            match index.index_type:
                case "IvfHnswPq":
                    self.config.embedding_col = index.columns[0]
                case "FTS":
                    self.config.text_col = index.columns[0]
                case "BTree":
                    self.config.id_col = index.columns[0]
        return self

    def open_table(self) -> lancedb.table.Table:
        import lancedb

        db = lancedb.connect(self.config.lancedb_path)
        self.table = db.open_table(self.config.table_name)

        logger.info(f"{self.__class__.__name__}: {self.table}")
        logger.info(
            f"num_items: {self.table.count_rows()}, columns: {self.table.schema.names}"
        )
        return self.table

    def index_data(
        self, dataset: datasets.Dataset, *, overwrite: bool = False
    ) -> lancedb.table.Table:
        if self.table is not None and not overwrite:
            return self.table

        import lancedb

        schema = dataset.data.schema
        schema = schema.set(
            # scalar index does not work on large_string
            schema.get_field_index(self.config.id_col),
            pa.field(self.config.id_col, pa.string()),
        )

        if self.config.embedding_col:
            embedding_dim = len(dataset[self.config.embedding_col][0])
            schema = schema.set(
                # embedding column must be fixed size float array
                schema.get_field_index(self.config.embedding_col),
                pa.field(
                    self.config.embedding_col, pa.list_(pa.float32(), embedding_dim)
                ),
            )

        db = lancedb.connect(self.config.lancedb_path)
        self.table = db.create_table(
            self.config.table_name,
            data=dataset.data.to_batches(max_chunksize=1024),
            schema=schema,
            mode="overwrite",
        )
        self.table.create_scalar_index(self.config.id_col)
        self.table.create_fts_index(self.config.text_col)

        if self.config.embedding_col:
            num_items = len(dataset)
            embedding_dim = len(dataset[self.config.embedding_col][0])
            # rule of thumb: nlist ~= 4 * sqrt(n_vectors)
            num_partitions = 2 ** int(math.log2(num_items) / 2)
            num_sub_vectors = embedding_dim // 8

            self.table.create_index(
                vector_column_name=self.config.embedding_col,
                metric="cosine",
                num_partitions=num_partitions,
                num_sub_vectors=num_sub_vectors,
                index_type="IVF_HNSW_PQ",
            )

        self.table.optimize(
            cleanup_older_than=datetime.timedelta(days=0),
            delete_unverified=True,
            retrain=True,
        )

        logger.info(f"{self.__class__.__name__}: {self.table}")
        logger.info(
            f"num_items: {self.table.count_rows()}, columns: {self.table.schema.names}"
        )
        return self.table

    def search(
        self,
        embedding: np.ndarray,
        exclude_item_ids: list[str] | None = None,
        top_k: int = 20,
    ) -> datasets.Dataset:
        exclude_item_ids = exclude_item_ids or [""]
        exclude_filter = ", ".join(
            f"'{str(item).replace("'", "''")}'" for item in exclude_item_ids
        )
        exclude_filter = f"{self.config.id_col} NOT IN ({exclude_filter})"
        rec_table = (
            self.table.search(embedding)
            .where(exclude_filter, prefilter=True)
            .nprobes(8)
            .refine_factor(4)
            .limit(top_k)
            .to_arrow()
        )
        rec_table = rec_table.append_column(
            "score", pc.subtract(1, rec_table["_distance"])
        )
        return datasets.Dataset(rec_table)

    def get_ids(self, ids: list[str]) -> datasets.Dataset:
        ids_filter = ", ".join(f"'{str(id_val).replace("'", "''")}'" for id_val in ids)
        result = (
            self.table.search()
            .where(f"{self.config.id_col} IN ({ids_filter})")
            .to_arrow()
        )
        return datasets.Dataset(result)

    def get_id(self, id_val: str | None) -> dict[str, Any]:
        if id_val is None:
            return {}

        result = self.get_ids([id_val])
        if len(result) == 0:
            return {}
        return result[0]


class FaissIndex:
    def __init__(
        self, config: IndexConfig, index: datasets.Dataset | None = None
    ) -> None:
        super().__init__()
        self.config = config
        self.index = index
        self.id2idx: pd.Series | None = None

    def save(self, path: str) -> None:
        index_name = self.index.list_indexes()[0]
        self.index.to_parquet(pathlib.Path(path, "data.parquet"))
        self.index.save_faiss_index(index_name, pathlib.Path(path, "index.faiss"))

    @classmethod
    def load(cls, config: IndexConfig, path: str) -> FaissIndex:
        index: datasets.Dataset = datasets.Dataset.from_parquet(
            pathlib.Path(path, "data.parquet").as_posix()
        )
        index.load_faiss_index("embedding_idx", pathlib.Path(path, "index.faiss"))

        required_cols = {config.id_col}
        if config.embedding_col is not None:
            required_cols.add(config.embedding_col)

        missing_cols = required_cols - set(index.column_names)
        if len(missing_cols) > 0:
            msg = f"index is missing required columns: {missing_cols}"
            raise ValueError(msg)

        logger.info(f"{cls.__name__}: {index}")
        logger.info(f"num_items: {len(index)}, columns: {index.column_names}")
        return cls(config, index).configure_id2idx(overwrite=True)

    def configure_id2idx(self, *, overwrite: bool = False) -> FaissIndex:
        if self.id2idx is not None and not overwrite:
            return self

        if self.index is None:
            msg = "index is not initialised"
            raise RuntimeError(msg)

        self.id2idx = pd.Series(
            {k: i for i, k in enumerate(self.index[self.config.id_col])}
        )
        return self

    def index_data(
        self, dataset: datasets.Dataset, *, overwrite: bool = False
    ) -> datasets.Dataset:
        if self.index is not None and not overwrite:
            return self.index

        import faiss

        self.index = dataset
        self.configure_id2idx(overwrite=True)
        if self.config.embedding_col is not None:
            # rule of thumb: nlist ~= 4 * sqrt(n_vectors)
            num_items = len(dataset)
            embedding_dim = len(dataset[self.config.embedding_col][0])

            nlist = 2 ** int(math.log2(num_items) / 2)
            m = embedding_dim // 8
            string_factory = (
                f"L2norm,OPQ{m},IVF{nlist}_HNSW32,PQ{m},Refine(L2norm,Flat)"
            )

            self.index.add_faiss_index(
                column=self.config.embedding_col,
                index_name="embedding_idx",
                string_factory=string_factory,
                metric_type=faiss.METRIC_INNER_PRODUCT,
                train_size=num_items,
            )
            faiss_index = self.index.get_index("embedding_idx").faiss_index
            faiss.extract_index_ivf(faiss_index).nprobe = 8

        logger.info(f"{self.__class__.__name__}: {self.index}")
        logger.info(f"num_items: {len(self.index)}, columns: {self.index.column_names}")
        return self.index

    def search(
        self,
        embedding: np.ndarray,
        exclude_item_ids: list[str] | None = None,
        top_k: int = 20,
    ) -> datasets.Dataset:
        exclude_set = set(exclude_item_ids or [""])
        # we take (2 * (top_k + len(exclude_set))) nearest items to ensure sufficient for post-filtering
        index_name = self.index.list_indexes()[0]
        scores, results = self.index.get_nearest_examples(
            index_name=index_name, query=embedding, k=2 * (top_k + len(exclude_set))
        )

        return (
            datasets.Dataset.from_dict(results)
            .add_column("score", scores)
            .filter(lambda example: example[self.config.id_col] not in exclude_set)
            .take(top_k)
        )

    def get_ids(self, ids: list[str]) -> datasets.Dataset:
        if self.id2idx is None:
            msg = "id2idx is not initialised"
            raise RuntimeError(msg)

        idx = [self.id2idx[id_] for id_ in ids if id_ in self.id2idx]
        return self.index.select(idx)

    def get_id(self, id_: str | None) -> dict[str, Any]:
        if id_ is None:
            return {}

        result = self.get_ids([id_])
        if len(result) == 0:
            return {}
        return result[0]
