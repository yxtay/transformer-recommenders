from __future__ import annotations

import bentoml
import numpy as np
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
    NumpyArrayType,
    UserQuery,
)
from xfmr_rec.params import TOP_K

MODEL_NAME = "xfmr_seq_embedded_rec"


class Query(BaseQuery):
    item_ids: list[str] | None = None
    item_texts: list[str] | None = None
    input_embeds: NumpyArrayType | None = None


@bentoml.service()
class Model:
    model_ref = bentoml.models.BentoModel(MODEL_NAME)

    @logger.catch(reraise=True)
    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        from xfmr_rec.params import TRANSFORMER_PATH

        model_path = self.model_ref.path_of(TRANSFORMER_PATH)
        self.model = SentenceTransformer(model_path).eval()
        logger.info("model loaded: {}", model_path)

    @bentoml.api()
    @logger.catch(reraise=True)
    @torch.inference_mode()
    def embed(self, query: Query) -> Query:
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
    async def embed_query(self, query: Query) -> Query:
        return await self.model.to_async.embed(query)

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
        return Query(
            item_ids=[item.item_id],
            item_texts=[item.item_text],
            input_embeds=item.embedding[None, :],
        )

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
