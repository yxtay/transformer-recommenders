from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any

import pydantic
from loguru import logger

from xfmr_rec.params import ITEMS_TABLE_NAME, LANCE_DB_PATH

if TYPE_CHECKING:
    import datasets
    import lancedb
    import numpy as np


class LanceIndexConfig(pydantic.BaseModel):
    lancedb_path: str = LANCE_DB_PATH
    table_name: str = ITEMS_TABLE_NAME
    id_col: str = "item_id"
    text_col: str = "item_text"
    embedding_col: str | None = None


class LanceIndex:
    def __init__(self, config: LanceIndexConfig) -> None:
        super().__init__()
        self.config = config
        self.table: lancedb.table.Table | None = None

    @classmethod
    def load(cls, config: LanceIndexConfig) -> LanceIndex:
        self = cls(config)
        table = self.open_table()

        for index in table.list_indices():
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

    def index_data(self, dataset: datasets.Dataset) -> lancedb.table.Table:
        import math

        import lancedb
        import pyarrow as pa

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

    def get_ids(self, ids: list[str]) -> datasets.Dataset:
        import datasets

        if not ids:
            return datasets.Dataset.from_dict({})

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

    def search(
        self,
        embedding: np.ndarray,
        exclude_item_ids: list[str] | None = None,
        top_k: int = 20,
    ) -> datasets.Dataset:
        import datasets
        import pyarrow.compute as pc

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
            .select([self.config.id_col, self.config.text_col, "_distance"])
            .to_arrow()
        )
        rec_table = rec_table.append_column(
            "score", pc.subtract(1, rec_table["_distance"])
        )
        return datasets.Dataset(rec_table)
