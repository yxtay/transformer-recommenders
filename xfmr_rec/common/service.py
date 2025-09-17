from __future__ import annotations

from typing import Annotated

import bentoml
import numpy as np
import numpy.typing as npt
import pydantic
from bentoml.validators import DType
from loguru import logger

from xfmr_rec.params import TOP_K

NumpyArrayType = Annotated[npt.NDArray[np.float32], DType("float32")]


class Activity(pydantic.BaseModel):
    item_id: list[str]
    item_text: list[str]


class UserQuery(pydantic.BaseModel):
    user_id: str = "0"
    user_text: str = ""
    history: Activity | None = None
    target: Activity | None = None


class ItemQuery(bentoml.IODescriptor):
    item_id: str = "0"
    item_text: str = ""
    embedding: NumpyArrayType | None = None


class BaseQuery(bentoml.IODescriptor):
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
    @logger.catch(reraise=True)
    def __init__(self) -> None:
        from xfmr_rec.index import LanceIndex, LanceIndexConfig
        from xfmr_rec.params import ITEMS_TABLE_NAME, LANCE_DB_PATH

        lance_db_path = self.model_ref.path_of(LANCE_DB_PATH)
        config = LanceIndexConfig(
            lancedb_path=lance_db_path, table_name=ITEMS_TABLE_NAME
        )
        self.index = LanceIndex.load(config)

    @bentoml.api()
    @logger.catch(reraise=True)
    def search(
        self, query: BaseQuery, exclude_item_ids: list[str], top_k: int = TOP_K
    ) -> list[ItemCandidate]:
        from pydantic import TypeAdapter

        results = self.index.search(
            query.embedding,
            exclude_item_ids=exclude_item_ids,
            top_k=top_k,
        )
        return TypeAdapter(list[ItemCandidate]).validate_python(results.to_list())

    @bentoml.api()
    @logger.catch(reraise=True)
    def get_id(self, item_id: str) -> ItemQuery:
        from bentoml.exceptions import NotFound

        result = self.index.get_id(item_id)
        if len(result) == 0:
            msg = f"item not found: {item_id = }"
            raise NotFound(msg)
        return ItemQuery.model_validate(result)

    @bentoml.api()
    @logger.catch(reraise=True)
    def get_ids(self, item_ids: list[str]) -> dict[str, ItemQuery]:
        results = self.index.get_ids(item_ids)
        results = pydantic.TypeAdapter(list[ItemQuery]).validate_python(
            results.to_list()
        )
        return {item.item_id: item for item in results}


class BaseUserIndex:
    @logger.catch(reraise=True)
    def __init__(self) -> None:
        from xfmr_rec.index import LanceIndex, LanceIndexConfig
        from xfmr_rec.params import LANCE_DB_PATH, USERS_TABLE_NAME

        lance_db_path = self.model_ref.path_of(LANCE_DB_PATH)
        config = LanceIndexConfig(
            lancedb_path=lance_db_path, table_name=USERS_TABLE_NAME
        )
        self.index = LanceIndex.load(config)

    @bentoml.api()
    @logger.catch(reraise=True)
    def get_id(self, user_id: str) -> UserQuery:
        from bentoml.exceptions import NotFound

        result = self.index.get_id(user_id)
        if len(result) == 0:
            msg = f"user not found: {user_id = }"
            raise NotFound(msg)
        return UserQuery.model_validate(result)


class BaseService:
    @bentoml.api()
    @logger.catch(reraise=True)
    async def search_items(
        self,
        query: BaseQuery,
        exclude_item_ids: list[str] | None = None,
        top_k: int = TOP_K,
    ) -> list[ItemCandidate]:
        exclude_item_ids = exclude_item_ids or []
        return await self.item_index.to_async.search(
            query, exclude_item_ids=exclude_item_ids, top_k=top_k
        )

    @bentoml.api()
    @logger.catch(reraise=True)
    async def item_id(self, item_id: str) -> ItemQuery:
        return await self.item_index.to_async.get_id(item_id)

    @bentoml.api()
    @logger.catch(reraise=True)
    async def user_id(self, user_id: str) -> UserQuery:
        return await self.user_index.to_async.get_id(user_id)

    @bentoml.api()
    @logger.catch(reraise=True)
    async def model_version(self: Service) -> str:
        return self.model_ref.tag.version

    @bentoml.api()
    @logger.catch(reraise=True)
    async def model_name(self: Service) -> str:
        return self.model_ref.tag.name
