from __future__ import annotations

import datetime
import math
import pathlib
import shutil
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
    """
    Configuration for index classes specifying ID and embedding columns.
    """

    id_col: str = "item_id"
    embedding_col: str | None = None


class LanceIndexConfig(IndexConfig):
    """
    Configuration for LanceDB index, including paths and text column.
    """

    lancedb_path: str = LANCE_DB_PATH
    table_name: str = ITEMS_TABLE_NAME
    text_col: str = "item_text"


class LanceIndex:
    """
    Index implementation using LanceDB for fast vector and text search.
    """

    def __init__(
        self, config: LanceIndexConfig, table: lancedb.table.Table | None = None
    ) -> None:
        """Initialize LanceIndex with configuration and optional table.

        Args:
            config (LanceIndexConfig): Configuration specifying paths and
                column names used by the index.
            table (lancedb.table.Table | None): Optional pre-opened
                LanceDB table. If not provided the table will be opened
                later via :meth:`open_table`.
        """
        super().__init__()
        self.config = config
        self.table = table

    def save(self, path: str) -> None:
        """Copy the underlying LanceDB store to a new path.

        This is a convenience that copies the directory backing the
        LanceDB instance to ``path`` using ``shutil.copytree``. No
        validation is performed on the target location.

        Args:
            path (str): Destination path where the LanceDB store will be
                copied.

        Returns:
            None: The function copies files on disk and does not return a value.
        """
        shutil.copytree(self.config.lancedb_path, path)

    @classmethod
    def load(cls, config: LanceIndexConfig) -> LanceIndex:
        """Load a LanceIndex from disk and infer column names from indices.

        This classmethod opens the LanceDB table configured in ``config``
        and inspects any indices created on the table to populate the
        configuration fields ``embedding_col``, ``text_col`` and
        ``id_col`` when possible.

        Args:
            config (LanceIndexConfig): Configuration pointing to the
                LanceDB store to load.

        Returns:
            LanceIndex: Configured LanceIndex with an opened table.
        """
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
        """Open and return the LanceDB table specified by the config.

        This method connects to the LanceDB store at ``config.lancedb_path``
        and opens the table ``config.table_name``. The opened table is
        stored on the instance as ``self.table``.

        Returns:
            lancedb.table.Table: Opened LanceDB table object.
        """
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
        """Create and index data in LanceDB from a HuggingFace Dataset.

        The provided dataset is used to create a LanceDB table; scalar
        and full-text-search (FTS) indices are created for the configured
        ID and text columns. If an embedding column is configured a
        vector index (IVF_HNSW_PQ) will be created using heuristics for
        partitioning and sub-vector size.

        Args:
            dataset (datasets.Dataset): HuggingFace Dataset containing
                the data to index.
            overwrite (bool): If ``True`` an existing table will be
                replaced. Defaults to ``False``.

        Returns:
            lancedb.table.Table: The created or existing LanceDB table.
        """
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
        embedding: np.typing.NDArray[np.float32],
        exclude_item_ids: list[str] | None = None,
        top_k: int = 20,
    ) -> datasets.Dataset:
        """Search the LanceDB vector index for the nearest items.

        The method performs a vector search with optional prefiltering to
        exclude specific item IDs. Returned results are converted into a
        HuggingFace ``datasets.Dataset`` and a cosine-like ``score`` is
        appended (computed as 1 - distance).

        Args:
            embedding (np.typing.NDArray[np.float32]): Query vector.
            exclude_item_ids (list[str] | None): Optional list of item
                IDs to exclude from results.
            top_k (int): Number of top results to return.

        Returns:
            datasets.Dataset: Dataset containing the search results with an
            additional ``score`` column. The ``score`` is computed as
            ``1 - _distance`` to resemble cosine similarity.
        """
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
        """Fetch rows from the LanceDB table matching the provided IDs.

        Args:
            ids (list[str]): List of item identifiers to fetch.

        Returns:
            datasets.Dataset: Dataset containing the matching rows.
        """
        ids_filter = ", ".join(f"'{str(id_val).replace("'", "''")}'" for id_val in ids)
        result = (
            self.table.search()
            .where(f"{self.config.id_col} IN ({ids_filter})")
            .to_arrow()
        )
        return datasets.Dataset(result)

    def get_id(self, id_val: str | None) -> dict[str, Any]:
        """Retrieve a single item from LanceDB by its ID.

        Args:
            id_val (str | None): Item ID to fetch. If ``None`` an empty
                dictionary is returned.

        Returns:
            dict[str, Any]: The first matching row as a dictionary or an
            empty dictionary if no match is found.
        """
        if id_val is None:
            return {}

        result = self.get_ids([id_val])
        if len(result) == 0:
            return {}
        return result[0]


class FaissIndex:
    """
    Index implementation using Faiss for fast vector search.
    """

    def __init__(
        self, config: IndexConfig, index: datasets.Dataset | None = None
    ) -> None:
        """Initialize FaissIndex with configuration and optional dataset.

        Args:
            config (IndexConfig): Configuration specifying id and
                embedding column names.
            index (datasets.Dataset | None): Optional dataset already
                prepared for Faiss indexing.
        """
        super().__init__()
        self.config = config
        self.index = index
        self.id2idx: pd.Series | None = None

    def save(self, path: str) -> None:
        """Persist the dataset and Faiss index to disk.

        The dataset is written to ``<path>/data.parquet`` and the Faiss
        index is saved to ``<path>/index.faiss``.

        Args:
            path (str): Directory where data and index files will be
                written.

        Returns:
            None: The function writes files to disk and does not return a value.
        """
        index_name = self.index.list_indexes()[0]
        self.index.to_parquet(pathlib.Path(path, "data.parquet"))
        self.index.save_faiss_index(index_name, pathlib.Path(path, "index.faiss"))

    @classmethod
    def load(cls, config: IndexConfig, path: str) -> FaissIndex:
        """Load a Faiss index and associated data from disk.

        The parquet dataset and Faiss index file are loaded and the method
        validates that required columns (ID and optionally embedding)
        exist in the dataset.

        Args:
            config (IndexConfig): Expected index configuration.
            path (str): Path containing ``data.parquet`` and
                ``index.faiss``.

        Returns:
            FaissIndex: Initialised FaissIndex with loaded dataset and a
            configured ``id2idx`` mapping.
        """
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
        """Create or refresh a pandas Series mapping item IDs to row
        indices in the dataset.

        Args:
            overwrite (bool): If ``True`` rebuild the mapping even if it
                already exists.

        Returns:
            FaissIndex: Self for chaining.
        """
        if self.id2idx is not None and not overwrite:
            return self

        if self.index is None:
            msg = "index is not initialised"
            raise RuntimeError(msg)

        self.id2idx = pd.Series(
            pd.RangeIndex(len(self.index)),
            index=self.index.with_format("pandas")[self.config.id_col].array,
        )
        return self

    def index_data(
        self, dataset: datasets.Dataset, *, overwrite: bool = False
    ) -> datasets.Dataset:
        """Index a datasets.Dataset with Faiss for efficient nearest
        neighbour search.

        If an embedding column is present the method constructs a
        composite index (OPQ + IVF + HNSW + PQ) and trains it using the
        dataset vectors.

        Args:
            dataset (datasets.Dataset): Dataset containing at least the
                embedding column if vector indexing is desired.
            overwrite (bool): If ``True`` replace an existing index.

        Returns:
            datasets.Dataset: The dataset registered with Faiss indices.
        """
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
        embedding: np.typing.NDArray[np.float32],
        exclude_item_ids: list[str] | None = None,
        top_k: int = 20,
    ) -> datasets.Dataset:
        """Search the Faiss index and return the top-k results.

        The method queries the Faiss-backed index and post-filters any
        excluded IDs. It ensures enough neighbors are retrieved to allow
        for filtering and returns a HuggingFace Dataset with a ``score``
        column added.

        Args:
            embedding (np.typing.NDArray[np.float32]): Query vector.
            exclude_item_ids (list[str] | None): Item IDs to exclude.
            top_k (int): Number of results to return after filtering.

        Returns:
            datasets.Dataset: Top-k search results as a Dataset. Scores are
            returned in the same order as the rows.
        """
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
        """Return rows from the dataset that match the provided IDs.

        Args:
            ids (list[str]): List of item IDs to retrieve.

        Returns:
            datasets.Dataset: Dataset containing the matching rows.
        """
        if self.id2idx is None:
            msg = "id2idx is not initialised"
            raise RuntimeError(msg)

        idx = [self.id2idx[id_] for id_ in ids if id_ in self.id2idx]
        return self.index.select(idx)

    def get_id(self, id_: str | None) -> dict[str, Any]:
        """Retrieve a single row from the dataset matching ``id_``.

        Args:
            id_ (str | None): Item ID to fetch. If ``None`` an empty
                dictionary is returned.

        Returns:
            dict[str, Any]: The row as a dictionary or an empty dict if
                not found.
        """
        if id_ is None:
            return {}

        result = self.get_ids([id_])
        if len(result) == 0:
            return {}
        return result[0]
