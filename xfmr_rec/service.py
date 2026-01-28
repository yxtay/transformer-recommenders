from __future__ import annotations

from typing import Annotated, Any

import bentoml
import numpy as np
import pydantic
from bentoml.exceptions import NotFound
from bentoml.validators import DType
from loguru import logger

from xfmr_rec.index import LanceIndex, LanceIndexConfig
from xfmr_rec.params import ITEMS_TABLE_NAME, LANCE_DB_PATH, TOP_K, USERS_TABLE_NAME

NumpyArrayType = Annotated[np.typing.NDArray[np.float32], DType("float32")]


class Activity(pydantic.BaseModel):
    item_id: list[str]
    item_text: list[str]


class BaseQuery(bentoml.IODescriptor):
    """Base query object containing embedding and search parameters.

    Attributes:
        embedding: The computed embedding for the query (optional).
        exclude_item_ids: List of item ids to exclude from results.
        top_k: Maximum number of results to return.
    """

    embedding: NumpyArrayType | None = None
    exclude_item_ids: list[str] | None = None
    top_k: int = TOP_K


class UserQuery(pydantic.BaseModel):
    user_id: str = "0"
    user_text: str = ""
    history: Activity | None = None
    target: Activity | None = None


class ItemQuery(bentoml.IODescriptor):
    item_id: str = "0"
    item_text: str = ""
    embedding: NumpyArrayType | None = None


class ItemCandidate(pydantic.BaseModel):
    item_id: str
    item_text: str
    score: float


EXAMPLE_ITEM = ItemQuery(
    item_id="1",
    item_text='{"title":"Toy Story (1995)","genres":["Animation","Children\'s","Comedy"]}',
)

EXAMPLE_USER = UserQuery(
    user_id="1",
    user_text='{"gender":"F","age":1,"occupation":10,"zipcode":"48067"}',
)

packages = [
    "--extra-index-url https://download.pytorch.org/whl/cpu",
    "datasets",
    "lancedb",
    "loguru",
    "pylance",
    "sentence-transformers[onnx]",
]
IMAGE = bentoml.images.Image().python_packages(*packages)
ENVS = [{"name": "UV_NO_CACHE", "value": "1"}]


class BaseItemIndex:
    model_ref: bentoml.models.BentoModel

    @logger.catch(reraise=True)
    def __init__(self) -> None:
        """Initialize the item index dependency.

        Loads a `LanceIndex` configured to point at the items table inside the
        BentoModel's LANCE_DB_PATH artifact. This is used by the item index
        service to perform vector search and lookups.
        """
        lance_db_path = self.model_ref.path_of(LANCE_DB_PATH)
        config = LanceIndexConfig(
            lancedb_path=lance_db_path, table_name=ITEMS_TABLE_NAME
        )
        self.index = LanceIndex.load(config)

    @bentoml.api()
    @logger.catch(reraise=True)
    def search(self, query: BaseQuery) -> list[ItemCandidate]:
        """Search for item candidates by embedding.

        Args:
            query (BaseQuery): Object containing an `embedding` array to search
                with, along with `exclude_item_ids` and `top_k` parameters.

        Returns:
            list[ItemCandidate]: A list of item candidate objects sorted by
                descending score.
        """
        assert query.embedding is not None
        results = self.index.search(
            query.embedding,
            exclude_item_ids=query.exclude_item_ids,
            top_k=query.top_k,
        )
        return pydantic.TypeAdapter(list[ItemCandidate]).validate_python(
            results.to_list()
        )

    @bentoml.api()
    @logger.catch(reraise=True)
    def get_id(self, item_id: str) -> ItemQuery:
        """Retrieve a single item by id from the index.

        Args:
            item_id (str): The string identifier of the item to fetch.

        Returns:
            ItemQuery: A pydantic model representing the item.

        Raises:
            bentoml.exceptions.NotFound: If the item does not exist in the
                index.
        """
        result = self.index.get_id(item_id)
        if len(result) == 0:
            msg = f"item not found: {item_id = }"
            raise NotFound(msg)
        return ItemQuery.model_validate(result)

    @bentoml.api()
    @logger.catch(reraise=True)
    def get_ids(self, item_ids: list[str]) -> dict[str, ItemQuery]:
        """Retrieve multiple items by id from the index.

        Args:
            item_ids (list[str]): List of item ids to fetch.

        Returns:
            dict[str, ItemQuery]: Mapping from item id to `ItemQuery` models for
                all found items. Missing ids are omitted from the result.
        """
        results = self.index.get_ids(item_ids)
        results = pydantic.TypeAdapter(list[ItemQuery]).validate_python(
            results.to_list()
        )
        return {item.item_id: item for item in results}


class BaseUserIndex:
    model_ref: bentoml.models.BentoModel

    @logger.catch(reraise=True)
    def __init__(self) -> None:
        """Initialize the user index dependency.

        Loads a `LanceIndex` configured to point at the users table inside the
        BentoModel's LANCE_DB_PATH artifact. This provides user lookups by id.
        """
        lance_db_path = self.model_ref.path_of(LANCE_DB_PATH)
        config = LanceIndexConfig(
            lancedb_path=lance_db_path, table_name=USERS_TABLE_NAME
        )
        self.index = LanceIndex.load(config)

    @bentoml.api()
    @logger.catch(reraise=True)
    def get_id(self, user_id: str) -> UserQuery:
        """Retrieve a single user by id from the index.

        Args:
            user_id (str): The string identifier of the user to fetch.

        Returns:
            UserQuery: A pydantic model representing the user.

        Raises:
            bentoml.exceptions.NotFound: If the user does not exist in the
                index.
        """
        result = self.index.get_id(user_id)
        if len(result) == 0:
            msg = f"user not found: {user_id = }"
            raise NotFound(msg)
        return UserQuery.model_validate(result)


class BaseService:
    model_ref: bentoml.models.BentoModel
    item_index: bentoml.Dependency[Any]
    user_index: bentoml.Dependency[Any]

    @bentoml.api()
    @logger.catch(reraise=True)
    async def search_items(self, query: BaseQuery) -> list[ItemCandidate]:
        """Asynchronously search for items using the item index dependency.

        This method delegates to the item index service's async `search`
        implementation. It ensures `exclude_item_ids` is a list before
        forwarding the call.

        Args:
            query (BaseQuery): Query object containing an `embedding` to search
                with.
            exclude_item_ids (list[str] | None): Optional list of item ids to
                exclude from the results.
            top_k (int): Number of candidates to return.

        Returns:
            list[ItemCandidate]: The search results returned by the index.
        """
        if query.embedding is None:
            return []

        query.exclude_item_ids = query.exclude_item_ids or []
        return await self.item_index.to_async.search(query)

    @bentoml.api()
    @logger.catch(reraise=True)
    async def item_id(self, item_id: str) -> ItemQuery:
        """Asynchronously retrieve an item by id using the item index.

        Args:
            item_id (str): The id of the item to fetch.

        Returns:
            ItemQuery: The item model returned by the index.
        """
        return await self.item_index.to_async.get_id(item_id)

    @bentoml.api()
    @logger.catch(reraise=True)
    async def user_id(self, user_id: str) -> UserQuery:
        """Asynchronously retrieve a user by id using the user index.

        Args:
            user_id (str): The id of the user to fetch.

        Returns:
            UserQuery: The user model returned by the index.
        """
        return await self.user_index.to_async.get_id(user_id)

    @bentoml.api()
    @logger.catch(reraise=True)
    async def model_version(self) -> str:
        """Return the BentoModel's version tag as a string.

        Returns:
            str: The model version from the BentoModel tag.
        """
        return self.model_ref.tag.version

    @bentoml.api()
    @logger.catch(reraise=True)
    async def model_name(self) -> str:
        """Return the BentoModel's name tag as a string.

        Returns:
            str: The model name from the BentoModel tag.
        """
        return self.model_ref.tag.name
