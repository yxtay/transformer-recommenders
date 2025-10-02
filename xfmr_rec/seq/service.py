from __future__ import annotations

import bentoml
import torch
from loguru import logger

from xfmr_rec.params import TOP_K
from xfmr_rec.seq import MODEL_NAME
from xfmr_rec.seq.models import SeqRecModel
from xfmr_rec.service import (
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


class Query(BaseQuery):
    item_ids: list[str] | None = None
    item_texts: list[str] | None = None


@bentoml.service()
class Model:
    model_ref = bentoml.models.BentoModel(MODEL_NAME)

    @logger.catch(reraise=True)
    def __init__(self) -> None:
        """Load the sequential recommendation model.

        The SeqRecModel is loaded from the BentoModel path and set to eval
        mode. Any loading error from the underlying framework is propagated.
        """
        self.model = SeqRecModel.load(self.model_ref.path).eval()
        logger.info("model loaded: {}", self.model_ref.path)

    @bentoml.api(batchable=True)
    @logger.catch(reraise=True)
    @torch.inference_mode()
    def embed(self, queries: list[Query]) -> list[Query]:
        """Embed a batch of queries using the SeqRecModel.

        Args:
            queries (list[Query]): Batch of queries where `item_texts` is a
                sequence of strings describing items in the sequence.

        Returns:
            list[Query]: The same list of queries where each `Query.embedding`
                is populated with the model-produced sentence embedding.
        """
        item_texts = [query.item_texts for query in queries]
        embeddings = self.model(item_texts)["sentence_embedding"].numpy(force=True)
        for query, embedding in zip(queries, embeddings, strict=True):
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
        """Recommend items for a given sequential query.

        This method ensures the query is expanded with item texts
        (`process_query`), embedded (`embed_query`), merges exclude ids, and
        delegates to `search_items`.

        Args:
            query (Query): The input query with optional `item_ids` or
                `item_texts`.
            exclude_item_ids (list[str] | None): Optional list of item ids to
                exclude.
            top_k (int): Number of results to return.

        Returns:
            list[ItemCandidate]: Recommended item candidates.
        """
        query = await self.process_query(query)
        query = await self.embed_query(query)
        exclude_item_ids = [*(exclude_item_ids or []), *(query.item_ids or [])]
        return await self.search_items(
            query, exclude_item_ids=exclude_item_ids, top_k=top_k
        )

    async def process_query(self, query: Query) -> Query:
        """Populate `item_texts` for a query when only `item_ids` are given.

        If `item_texts` are already present or `item_ids` is None, the query is
        returned unchanged. Otherwise the method fetches the items from the
        item index and constructs `item_texts` in the same order as
        `item_ids`.

        Args:
            query (Query): Query to process.

        Returns:
            Query: Query with `item_texts` populated when possible.
        """
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
        """Ensure a Query object has an embedding by delegating to the model.

        Args:
            query (Query): Query to embed.

        Returns:
            Query: The same query with `embedding` populated.
        """
        if query.embedding is not None:
            return query
        return (await self.model.to_async.embed([query]))[0]

    @bentoml.api()
    @logger.catch(reraise=True)
    async def recommend_with_item(
        self,
        item: ItemQuery,
        exclude_item_ids: list[str] | None = None,
        top_k: int = TOP_K,
    ) -> list[ItemCandidate]:
        """Recommend items given an ItemQuery by converting to Query first.

        Args:
            item (ItemQuery): The item to base recommendations on.
            exclude_item_ids (list[str] | None): Optional ids to exclude.
            top_k (int): Number of results to return.

        Returns:
            list[ItemCandidate]: Recommended items.
        """
        query = await self.process_item(item)
        return await self.recommend_with_query(
            query, exclude_item_ids=exclude_item_ids, top_k=top_k
        )

    @bentoml.api()
    @logger.catch(reraise=True)
    async def process_item(self, item: ItemQuery) -> Query:
        """Convert an ItemQuery into a Query (single-item sequence).

        Args:
            item (ItemQuery): Item to convert.

        Returns:
            Query: Query with a single item id and text.
        """
        return Query(item_ids=[item.item_id], item_texts=[item.item_text])

    @bentoml.api()
    @logger.catch(reraise=True)
    async def recommend_with_item_id(
        self,
        item_id: str,
        exclude_item_ids: list[str] | None = None,
        top_k: int = TOP_K,
    ) -> list[ItemCandidate]:
        """Recommend items for a given item id by fetching the item first.

        Args:
            item_id (str): The id of the item to base recommendations on.
            exclude_item_ids (list[str] | None): Optional ids to exclude.
            top_k (int): Number of results to return.

        Returns:
            list[ItemCandidate]: Recommended items.
        """
        item = await self.item_id(item_id)
        return await self.recommend_with_item(
            item, exclude_item_ids=exclude_item_ids, top_k=top_k
        )

    @bentoml.api()
    @logger.catch(reraise=True)
    async def item_id(self, item_id: str) -> ItemQuery:
        """Asynchronously retrieve an item by id from the item index.

        Args:
            item_id (str): The id of the item to fetch.

        Returns:
            ItemQuery: The item returned by the index.
        """
        return await self.item_index.to_async.get_id(item_id)

    @bentoml.api()
    @logger.catch(reraise=True)
    async def recommend_with_user(
        self,
        user: UserQuery,
        exclude_item_ids: list[str] | None = None,
        top_k: int = TOP_K,
    ) -> list[ItemCandidate]:
        """Recommend items for a given user by converting to a Query then
        delegating to `recommend_with_query`.

        Args:
            user (UserQuery): The user object containing history/target.
            exclude_item_ids (list[str] | None): Optional ids to exclude.
            top_k (int): Number of results to return.

        Returns:
            list[ItemCandidate]: Recommended items for the user.
        """
        query = await self.process_user(user)
        return await self.recommend_with_query(
            query, exclude_item_ids=exclude_item_ids, top_k=top_k
        )

    @bentoml.api()
    @logger.catch(reraise=True)
    async def process_user(self, user: UserQuery) -> Query:
        """Convert a UserQuery into a Query by aggregating history and target
        items.

        Args:
            user (UserQuery): The user to process.

        Returns:
            Query: Aggregated Query with item ids and texts.
        """
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
        """Recommend items for a user id by fetching the UserQuery then
        delegating to `recommend_with_user`.

        Args:
            user_id (str): The id of the user to recommend for.
            exclude_item_ids (list[str] | None): Optional ids to exclude.
            top_k (int): Number of results to return.

        Returns:
            list[ItemCandidate]: Recommended items.
        """
        user = await self.user_id(user_id)
        return await self.recommend_with_user(
            user, exclude_item_ids=exclude_item_ids, top_k=top_k
        )
