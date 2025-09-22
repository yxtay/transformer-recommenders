from __future__ import annotations

import bentoml
import torch
from loguru import logger

from xfmr_rec.common.service import (
    ENVS,
    IMAGE,
    BaseItemIndex,
    BaseQuery,
    BaseService,
    BaseUserIndex,
    ItemCandidate,
    ItemQuery,
    UserQuery,
)
from xfmr_rec.params import TOP_K
from xfmr_rec.seq import MODEL_NAME
from xfmr_rec.seq.models import SeqRecModel


class Query(BaseQuery):
    item_ids: list[str] | None = None
    item_texts: list[str] | None = None


@bentoml.service()
class Model:
    model_ref = bentoml.models.BentoModel(MODEL_NAME)

    @logger.catch(reraise=True)
    def __init__(self) -> None:
        self.model = SeqRecModel.load(self.model_ref.path).eval()
        logger.info("model loaded: {}", self.model_ref.path)

    @bentoml.api(batchable=True)
    @logger.catch(reraise=True)
    @torch.inference_mode()
    def embed(self, queries: list[Query]) -> list[Query]:
        item_texts = [query.item_texts for query in queries]
        embeddings = self.model(item_texts)["sentence_embedding"].numpy(force=True)
        for query, embedding in zip(queries, embeddings, strict=False):
            query.embedding = embedding
        return queries


@bentoml.service()
class ItemIndex(BaseItemIndex):
    model_ref = bentoml.models.BentoModel(MODEL_NAME)


@bentoml.service()
class UserIndex(BaseUserIndex):
    model_ref = bentoml.models.BentoModel(MODEL_NAME)


@bentoml.service(image=IMAGE, envs=ENVS, workers="cpu_count")
class Service(BaseService):
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
        query = await self.embed_query(query)
        exclude_item_ids = [*(exclude_item_ids or []), *(query.item_ids or [])]
        return await self.search_items(
            query, exclude_item_ids=exclude_item_ids, top_k=top_k
        )

    async def process_query(self, query: Query) -> Query:
        if query.item_ids is None:
            return query

        if query.item_texts is not None:
            return query

        items = await self.item_index.to_async.get_ids(query.item_ids)
        query.item_texts = [
            items[item_id].item_text for item_id in query.item_ids if item_id in items
        ]
        return query

    @bentoml.api()
    @logger.catch(reraise=True)
    async def embed_query(self, query: Query) -> Query:
        return (await self.model.to_async.embed([query]))[0]

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
        return Query(item_ids=[item.item_id], item_texts=[item.item_text])

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
        item_texts: list[str] = []
        if user.history:
            item_ids += user.history.item_id
            item_texts += user.history.item_text
        if user.target:
            item_ids += user.target.item_id
            item_texts += user.target.item_text
        return Query(item_ids=item_ids, item_texts=item_texts)

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
