from __future__ import annotations

from typing import Annotated

import bentoml
import numpy as np
import pydantic
import torch
from bentoml.exceptions import NotFound
from bentoml.validators import DType
from loguru import logger
from sentence_transformers import SentenceTransformer
from torch.nn.utils.rnn import pad_sequence

from xfmr_rec.index import LanceIndex, LanceIndexConfig
from xfmr_rec.params import (
    ITEMS_TABLE_NAME,
    LANCE_DB_PATH,
    TOP_K,
    TRANSFORMER_PATH,
    USERS_TABLE_NAME,
)

NumpyArrayType = Annotated[np.ndarray[tuple[int], np.dtype[np.float32]], DType("float32")]
MODEL_NAME = "xfmr_rec"


class Activity(pydantic.BaseModel):
    item_id: list[str]
    item_text: list[str]


class Query(bentoml.IODescriptor):
    """Query object containing embedding and search parameters.

    Attributes:
        embedding: The computed embedding for the query.
        item_ids: List of item ids to base recommendations on.
        item_texts: List of item texts to base recommendations on.
        input_embeds: Precomputed item embeddings.
        exclude_item_ids: List of item ids to exclude from results.
        top_k: Maximum number of results to return.
    """

    embedding: NumpyArrayType | None = None
    item_ids: list[str] | None = None
    item_texts: list[str] | None = None
    input_embeds: NumpyArrayType | None = None
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


@bentoml.service()
class Model:
    model_ref = bentoml.models.BentoModel(MODEL_NAME)

    @logger.catch(reraise=True)
    def __init__(self) -> None:
        """Load the SentenceTransformer used for sequence-embedded queries."""
        model_path = self.model_ref.path_of(TRANSFORMER_PATH)
        self.model = SentenceTransformer(model_path).eval()
        self.embed_dim: int = self.model.get_sentence_embedding_dimension()
        logger.info("model loaded: {}", model_path)

    @bentoml.api()
    def max_seq_length(self) -> int:
        return self.model.max_seq_length

    @bentoml.api(batchable=True)
    @logger.catch(reraise=True)
    @torch.inference_mode()
    def embed(self, queries: list[Query]) -> list[Query]:
        """Embed a batch of queries using the SentenceTransformer."""
        inputs_embeds = [
            torch.as_tensor(query.input_embeds[-self.max_seq_length() :])
            if query.input_embeds is not None
            else torch.zeros(1, self.embed_dim)
            for query in queries
        ]
        inputs_embeds = pad_sequence(inputs_embeds, batch_first=True).to(
            self.model.device
        )

        attention_mask = (inputs_embeds != 0).any(-1)
        embeddings = self.model(
            {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}
        )["sentence_embedding"].numpy(force=True)

        for query, embedding in zip(queries, embeddings, strict=True):
            query.embedding = embedding
        return queries


@bentoml.service()
class ItemIndex:
    model_ref = bentoml.models.BentoModel(MODEL_NAME)

    @logger.catch(reraise=True)
    def __init__(self) -> None:
        """Initialize the item index dependency."""
        lance_db_path = self.model_ref.path_of(LANCE_DB_PATH)
        config = LanceIndexConfig(
            lancedb_path=lance_db_path, table_name=ITEMS_TABLE_NAME
        )
        self.index = LanceIndex.load(config)

    @bentoml.api()
    @logger.catch(reraise=True)
    def search(self, query: Query) -> list[ItemCandidate]:
        """Search for item candidates by embedding."""
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
    model_ref = bentoml.models.BentoModel(MODEL_NAME)

    @logger.catch(reraise=True)
    def __init__(self) -> None:
        """Initialize the user index dependency."""
        lance_db_path = self.model_ref.path_of(LANCE_DB_PATH)
        config = LanceIndexConfig(
            lancedb_path=lance_db_path, table_name=USERS_TABLE_NAME
        )
        self.index = LanceIndex.load(config)

    @bentoml.api()
    @logger.catch(reraise=True)
    def get_id(self, user_id: str) -> UserQuery:
        result = self.index.get_id(user_id)
        if len(result) == 0:
            msg = f"user not found: {user_id = }"
            raise NotFound(msg)
        return UserQuery.model_validate(result)


@bentoml.service(image=IMAGE, envs=ENVS, workers="cpu_count")
class Service:
    model_ref = bentoml.models.BentoModel(MODEL_NAME)
    model = bentoml.depends(Model)
    item_index = bentoml.depends(ItemIndex)
    user_index = bentoml.depends(UserIndex)

    @bentoml.api()
    @logger.catch(reraise=True)
    async def recommend_with_query(self, query: Query) -> list[ItemCandidate]:
        query = await self.process_query(query)
        query = await self.embed_query(query)
        query.exclude_item_ids = [
            *(query.exclude_item_ids or []),
            *(query.item_ids or []),
        ]
        if query.embedding is None:
            return []
        return await self.item_index.to_async.search(query)

    async def process_query(self, query: Query) -> Query:
        if query.item_ids is None:
            return query
        if query.input_embeds is not None:
            return query

        items: dict[str, ItemQuery] = await self.item_index.to_async.get_ids(
            query.item_ids
        )
        item_ids = [item_id for item_id in query.item_ids if item_id in items]
        query.item_ids = item_ids[-await self.model.to_async.max_seq_length() :]
        embeddings = [items[item_id].embedding for item_id in query.item_ids]
        query.input_embeds = np.stack(embeddings) if embeddings else None
        return query

    @bentoml.api()
    @logger.catch(reraise=True)
    async def embed_query(self, query: Query) -> Query:
        if query.input_embeds is None:
            return query
        if query.embedding is not None:
            return query

        return (await self.model.to_async.embed([query]))[0]

    @bentoml.api()
    @logger.catch(reraise=True)
    async def recommend_with_item_id(
        self,
        item_id: str,
        exclude_item_ids: list[str] | None = None,
        top_k: int = TOP_K,
    ) -> list[ItemCandidate]:
        item = await self.item_id(item_id)
        query = Query(
            item_ids=[item.item_id],
            item_texts=[item.item_text],
            input_embeds=item.embedding[None, :]
            if item.embedding is not None
            else None,
            exclude_item_ids=exclude_item_ids,
            top_k=top_k,
        )
        return await self.recommend_with_query(query)

    @bentoml.api()
    @logger.catch(reraise=True)
    async def recommend_with_user_id(
        self,
        user_id: str,
        exclude_item_ids: list[str] | None = None,
        top_k: int = TOP_K,
    ) -> list[ItemCandidate]:
        user = await self.user_id(user_id)
        item_ids: list[str] = []
        item_texts: list[str] = []
        if user.history:
            item_ids += user.history.item_id
            item_texts += user.history.item_text
        if user.target:
            item_ids += user.target.item_id
            item_texts += user.target.item_text

        query = Query(
            item_ids=item_ids,
            item_texts=item_texts,
            exclude_item_ids=exclude_item_ids,
            top_k=top_k,
        )
        return await self.recommend_with_query(query)

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
    async def model_version(self) -> str:
        return self.model_ref.tag.version

    @bentoml.api()
    @logger.catch(reraise=True)
    async def model_name(self) -> str:
        return self.model_ref.tag.name
