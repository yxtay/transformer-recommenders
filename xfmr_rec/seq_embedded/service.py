from __future__ import annotations

from typing import Annotated

import bentoml
import numpy as np
import numpy.typing as npt
import pydantic
import torch
from bentoml.validators import DType
from loguru import logger

from xfmr_rec.params import SEQ_EMBEDDED_MODEL_NAME, TOP_K

NUMPY_ARRAY_TYPE = Annotated[npt.NDArray[np.float32], DType("float32")]


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
    embedding: NUMPY_ARRAY_TYPE | None = None


class Query(bentoml.IODescriptor):
    item_ids: list[str]
    input_embeds: NUMPY_ARRAY_TYPE | None = None
    embedding: NUMPY_ARRAY_TYPE | None = None


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
    model_ref = bentoml.models.BentoModel(SEQ_EMBEDDED_MODEL_NAME)

    @logger.catch(reraise=True)
    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        from xfmr_rec.params import TRANSFORMER_PATH

        model_path = self.model_ref.path_of(TRANSFORMER_PATH)
        self.model = SentenceTransformer(model_path)
        logger.info("model loaded: {}", model_path)

    @bentoml.api()
    @logger.catch(reraise=True)
    @torch.inference_mode()
    def encode(self, query: Query) -> Query:
        if query.input_embeds is None or query.input_embeds.size == 0:
            embedding_dim = self.model.get_sentence_embedding_dimension()
            query.embedding = np.zeros((1, embedding_dim), dtype=np.float32)
            return query

        inputs_embeds = torch.as_tensor(query.input_embeds, device=self.model.device)
        query.embedding = self.model(
            {"inputs_embeds": inputs_embeds[None, -self.model.max_seq_length :, :]}
        )["sentence_embedding"].numpy(force=True)
        return query


@bentoml.service()
class ItemIndex:
    model_ref = bentoml.models.BentoModel(SEQ_EMBEDDED_MODEL_NAME)

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
        results = self.index.search(
            query.embedding,
            exclude_item_ids=exclude_item_ids,
            top_k=top_k,
        )
        return pydantic.TypeAdapter(list[ItemCandidate]).validate_python(
            results.to_list()
        )

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


@bentoml.service()
class UserIndex:
    model_ref = bentoml.models.BentoModel(SEQ_EMBEDDED_MODEL_NAME)

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
        query = await self.process_query(query)
        query = await self.encode_query(query)
        exclude_item_ids = [*(exclude_item_ids or []), *query.item_ids]
        return await self.search_items(
            query, exclude_item_ids=exclude_item_ids, top_k=top_k
        )

    async def process_query(self, query: Query) -> Query:
        if query.input_embeds is not None:
            return query

        items = await self.item_index.to_async.get_ids(query.item_ids)
        embeddings = [
            items[item_id].embedding for item_id in query.item_ids if item_id in items
        ]
        query.input_embeds = np.stack(embeddings) if embeddings else None
        return query

    @bentoml.api()
    @logger.catch(reraise=True)
    async def encode_query(self, query: Query) -> Query:
        return await self.model.to_async.encode(query)

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
        query = await self.process_item(item)
        return await self.recommend_with_query(
            query, exclude_item_ids=exclude_item_ids, top_k=top_k
        )

    @bentoml.api()
    @logger.catch(reraise=True)
    async def process_item(self, item: ItemQuery) -> Query:
        return Query(item_ids=[item.item_id], input_embeds=item.embedding[None, :])

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
        query = await self.process_user(user)
        return await self.recommend_with_query(
            query, exclude_item_ids=exclude_item_ids, top_k=top_k
        )

    @bentoml.api()
    @logger.catch(reraise=True)
    async def process_user(self, user: UserQuery) -> Query:
        item_ids: list[str] = []
        if user.history:
            item_ids += user.history.item_id
        if user.target:
            item_ids += user.target.item_id
        return Query(item_ids=item_ids)

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
