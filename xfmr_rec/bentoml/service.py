from __future__ import annotations

import datetime  # noqa: TC003
import json
import pathlib
from typing import Annotated

import bentoml
import numpy as np  # noqa: TC002
import numpy.typing as npt  # noqa: TC002
import pydantic
import torch
from bentoml.validators import DType
from loguru import logger

from xfmr_rec.params import (
    LANCE_DB_PATH,
    MODEL_NAME,
    PROCESSORS_JSON,
    TOP_K,
    TRANSFORMER_PATH,
)


class Activity(pydantic.BaseModel):
    datetime: list[datetime.datetime]
    event_name: list[str]
    event_value: list[int]
    item_rn: list[int]
    item_id: list[str]
    item_text: list[str]


class UserQuery(pydantic.BaseModel):
    user_rn: int = 0
    user_id: str = "0"
    user_text: str = ""
    history: Activity | None = None
    target: Activity | None = None


class ItemQuery(pydantic.BaseModel):
    item_rn: int = 0
    item_id: str = "0"
    item_text: str = ""


class Query(bentoml.IODescriptor):
    text: str = ""
    embedding: Annotated[npt.NDArray[np.float32], DType("float32")] | None = None


class ItemCandidate(pydantic.BaseModel):
    item_id: str
    item_text: str
    score: float


EXAMPLE_ITEM = ItemQuery(
    item_rn=1,
    item_id="1",
    item_text='{"title":"Toy Story (1995)","genres":["Animation","Children\'s","Comedy"]}',
)

EXAMPLE_USER = UserQuery(
    user_rn=1,
    user_id="1",
    user_text='{"gender":"F","age":1,"occupation":10,"zipcode":"48067"}',
)

PACKAGES = [
    "--extra-index-url https://download.pytorch.org/whl/cpu",
    "lancedb",
    "loguru",
    "pandas",
    "pylance",
    "sentence-transformers[onnx]",
]
image = bentoml.images.Image().python_packages(*PACKAGES)
ENVS = [{"name": "UV_NO_CACHE", "value": "1"}]


@bentoml.service()
class Embedder:
    model_ref = bentoml.models.BentoModel(MODEL_NAME)

    @logger.catch(reraise=True)
    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        path = self.model_ref.path_of(TRANSFORMER_PATH)
        self.model = SentenceTransformer(path, local_files_only=True, backend="onnx")
        logger.info("model loaded: {}", path)

    @bentoml.api()
    @logger.catch(reraise=True)
    @torch.inference_mode()
    def embed(self, query: Query) -> Query:
        query.embedding = self.model.encode(query.text)
        return query


@bentoml.service()
class ItemProcessor:
    model_ref = bentoml.models.BentoModel(MODEL_NAME)

    @logger.catch(reraise=True)
    def __init__(self) -> None:
        from xfmr_rec.data.lightning import ItemProcessor

        lance_db_path = self.model_ref.path_of(LANCE_DB_PATH)
        processors_args = json.loads(
            pathlib.Path(self.model_ref.path_of(PROCESSORS_JSON)).read_text()
        )
        processors_args["items"].update({"lance_db_path": lance_db_path})
        self.item_processor = ItemProcessor.model_validate(processors_args["items"])

    @bentoml.api()
    @logger.catch(reraise=True)
    def search(
        self, query: Query, exclude_item_ids: list[str], top_k: int = TOP_K
    ) -> list[ItemCandidate]:
        from pydantic import TypeAdapter

        results_df = self.item_processor.search(
            query.embedding,
            exclude_item_ids=exclude_item_ids,
            top_k=top_k,
        )
        return TypeAdapter(list[ItemCandidate]).validate_python(
            results_df.itertuples(), from_attributes=True
        )

    @bentoml.api()
    @logger.catch(reraise=True)
    def get_id(self, item_id: str) -> ItemQuery:
        from bentoml.exceptions import NotFound

        result = self.item_processor.get_id(item_id)
        if len(result) == 0:
            msg = f"item not found: {item_id = }"
            raise NotFound(msg)
        return ItemQuery.model_validate(result)

    @bentoml.api()
    @logger.catch(reraise=True)
    def process(self, item: ItemQuery) -> Query:
        item_data = item.model_dump()
        return Query.model_validate(self.item_processor.process(item_data))


@bentoml.service()
class UserProcessor:
    model_ref = bentoml.models.BentoModel(MODEL_NAME)

    @logger.catch(reraise=True)
    def __init__(self) -> None:
        from xfmr_rec.data.lightning import UserProcessor

        lance_db_path = self.model_ref.path_of(LANCE_DB_PATH)
        processors_args = json.loads(
            pathlib.Path(self.model_ref.path_of(PROCESSORS_JSON)).read_text()
        )
        processors_args["users"].update({"lance_db_path": lance_db_path})
        self.user_processor = UserProcessor.model_validate(processors_args["users"])

    @bentoml.api()
    @logger.catch(reraise=True)
    def get_id(self, user_id: str) -> UserQuery:
        from bentoml.exceptions import NotFound

        result = self.user_processor.get_id(user_id)
        if len(result) == 0:
            msg = f"user not found: {user_id = }"
            raise NotFound(msg)
        return UserQuery.model_validate(result)

    @bentoml.api()
    @logger.catch(reraise=True)
    def process(self, user: UserQuery) -> Query:
        user_data = user.model_dump()
        return Query.model_validate(self.user_processor.process(user_data))


@bentoml.service(image=image, envs=ENVS, workers="cpu_count")
class Service:
    model_ref = bentoml.models.BentoModel(MODEL_NAME)
    embedder = bentoml.depends(Embedder)
    item_processor = bentoml.depends(ItemProcessor)
    user_processor = bentoml.depends(UserProcessor)

    @bentoml.api()
    @logger.catch(reraise=True)
    async def recommend_with_query(
        self,
        query: Query,
        exclude_item_ids: list[str] | None = None,
        top_k: int = TOP_K,
    ) -> list[ItemCandidate]:
        query = await self.embed_query(query)
        return await self.search_items(
            query, exclude_item_ids=exclude_item_ids, top_k=top_k
        )

    @bentoml.api()
    @logger.catch(reraise=True)
    async def embed_query(self, query: Query) -> Query:
        return await self.embedder.to_async.embed(query)

    @bentoml.api()
    @logger.catch(reraise=True)
    async def search_items(
        self,
        query: Query,
        exclude_item_ids: list[str] | None = None,
        top_k: int = TOP_K,
    ) -> list[ItemCandidate]:
        exclude_item_ids = exclude_item_ids or []
        return await self.item_processor.to_async.search(
            query, exclude_item_ids=exclude_item_ids, top_k=top_k
        )

    @bentoml.api()
    @logger.catch(reraise=True)
    async def recommend_with_item(
        self,
        item: ItemQuery,
        exclude_item_ids: list[str] | None = None,
        top_k: int = TOP_K,
    ) -> list[ItemCandidate]:
        if item.item_id:
            exclude_item_ids = [*(exclude_item_ids or []), item.item_id]

        query = await self.process_item(item)
        return await self.recommend_with_query(
            query, exclude_item_ids=exclude_item_ids, top_k=top_k
        )

    @bentoml.api()
    @logger.catch(reraise=True)
    async def process_item(self, item: ItemQuery) -> Query:
        return await self.item_processor.to_async.process(item)

    @bentoml.api()
    @logger.catch(reraise=True)
    async def recommend_with_item_id(
        self,
        item_id: str,
        exclude_item_ids: list[str] | None = None,
        top_k: int = TOP_K,
    ) -> list[ItemCandidate]:
        item = await self.item_id(item_id)
        return await self.recommend_with_item(
            item, exclude_item_ids=exclude_item_ids, top_k=top_k
        )

    @bentoml.api()
    @logger.catch(reraise=True)
    async def item_id(self, item_id: str) -> ItemQuery:
        return await self.item_processor.to_async.get_id(item_id)

    @bentoml.api()
    @logger.catch(reraise=True)
    async def recommend_with_user(
        self,
        user: UserQuery,
        exclude_item_ids: list[str] | None = None,
        top_k: int = TOP_K,
    ) -> list[ItemCandidate]:
        exclude_item_ids = exclude_item_ids or []
        if user.history:
            exclude_item_ids += user.history.item_id
        if user.target:
            exclude_item_ids += user.target.item_id

        query = await self.process_user(user)
        return await self.recommend_with_query(
            query, exclude_item_ids=exclude_item_ids, top_k=top_k
        )

    @bentoml.api()
    @logger.catch(reraise=True)
    async def process_user(self, user: UserQuery) -> Query:
        return await self.user_processor.to_async.process(user)

    @bentoml.api()
    @logger.catch(reraise=True)
    async def recommend_with_user_id(
        self,
        user_id: str,
        exclude_item_ids: list[str] | None = None,
        top_k: int = TOP_K,
    ) -> list[ItemCandidate]:
        user = await self.user_id(user_id)
        return await self.recommend_with_user(
            user, exclude_item_ids=exclude_item_ids, top_k=top_k
        )

    @bentoml.api()
    @logger.catch(reraise=True)
    async def user_id(self, user_id: str) -> UserQuery:
        return await self.user_processor.to_async.get_id(user_id)

    @bentoml.api()
    @logger.catch(reraise=True)
    async def model_version(self: Service) -> str:
        return self.model_ref.tag.version

    @bentoml.api()
    @logger.catch(reraise=True)
    async def model_name(self: Service) -> str:
        return self.model_ref.tag.name
