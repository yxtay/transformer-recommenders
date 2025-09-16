from __future__ import annotations

from typing import Annotated

import bentoml
import numpy as np  # noqa: TC002
import numpy.typing as npt  # noqa: TC002
import pydantic
import torch
from bentoml.validators import DType
from loguru import logger

from xfmr_rec.params import SEQ_MODEL_NAME, TOP_K


class Activity(pydantic.BaseModel):
    item_id: list[str]
    item_text: list[str]


class UserQuery(pydantic.BaseModel):
    user_id: str = "0"
    user_text: str = ""
    history: Activity | None = None
    target: Activity | None = None


class ItemQuery(pydantic.BaseModel):
    item_id: str = "0"
    item_text: str = ""


class Query(bentoml.IODescriptor):
    texts: list[str]
    embedding: Annotated[npt.NDArray[np.float32], DType("float32")] | None = None


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

PACKAGES = [
    "--extra-index-url https://download.pytorch.org/whl/cpu",
    "datasets",
    "lancedb",
    "loguru",
    "pylance",
    "sentence-transformers[onnx]",
]
image = bentoml.images.Image().python_packages(*PACKAGES)
ENVS = [{"name": "UV_NO_CACHE", "value": "1"}]


@bentoml.service()
class Model:
    model_ref = bentoml.models.BentoModel(SEQ_MODEL_NAME)

    @logger.catch(reraise=True)
    def __init__(self) -> None:
        from xfmr_rec.seq.models import SeqRecModel

        self.model = SeqRecModel.load(self.model_ref.path)
        logger.info("model loaded: {}", self.model_ref.path)

    @bentoml.api(batchable=True)
    @logger.catch(reraise=True)
    @torch.inference_mode()
    def embed(self, queries: list[Query]) -> list[Query]:
        query_texts = [query.texts for query in queries]
        embeddings = self.model.encode(query_texts).numpy(force=True)
        for query, embedding in zip(queries, embeddings, strict=False):
            query.embedding = embedding
        return queries


@bentoml.service()
class ItemIndex:
    model_ref = bentoml.models.BentoModel(SEQ_MODEL_NAME)

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
        self, query: Query, exclude_item_ids: list[str], top_k: int = TOP_K
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


@bentoml.service()
class UserIndex:
    model_ref = bentoml.models.BentoModel(SEQ_MODEL_NAME)

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


@bentoml.service(image=image, envs=ENVS, workers="cpu_count")
class Service:
    model = bentoml.depends(Model)
    item_index = bentoml.depends(ItemIndex)
    user_index = bentoml.depends(UserIndex)

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
        return (await self.model.to_async.embed([query]))[0]

    @bentoml.api()
    @logger.catch(reraise=True)
    async def search_items(
        self,
        query: Query,
        exclude_item_ids: list[str] | None = None,
        top_k: int = TOP_K,
    ) -> list[ItemCandidate]:
        exclude_item_ids = exclude_item_ids or []
        return await self.item_index.to_async.search(
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
        return Query(texts=[item.item_text])

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
        return await self.item_index.to_async.get_id(item_id)

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
        item_texts: list[str] = []
        if user.history:
            item_texts += user.history.item_text
        if user.target:
            item_texts += user.target.item_text
        return Query(texts=item_texts)

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
        return await self.user_index.to_async.get_id(user_id)

    @bentoml.api()
    @logger.catch(reraise=True)
    async def model_version(self: Service) -> str:
        return self.model_ref.tag.version

    @bentoml.api()
    @logger.catch(reraise=True)
    async def model_name(self: Service) -> str:
        return self.model_ref.tag.name
