from __future__ import annotations

import bentoml
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from xfmr_rec.mf import MODEL_NAME
from xfmr_rec.params import TOP_K, TRANSFORMER_PATH
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
    text: str | None = None


@bentoml.service()
class Model:
    model_ref = bentoml.models.BentoModel(MODEL_NAME)

    @logger.catch(reraise=True)
    def __init__(self) -> None:
        """Load the SentenceTransformer used for embedding queries.

        The model is loaded from the Bento model store using the path defined by
        the `TRANSFORMER_PATH` artifact in the BentoModel. The loaded model is
        placed into evaluation mode.

        Raises:
            Any exception raised by the BentoModel `path_of` call or the
            SentenceTransformer constructor is propagated.
        """
        model_path = self.model_ref.path_of(TRANSFORMER_PATH)
        self.model = SentenceTransformer(
            model_path, local_files_only=True, backend="onnx"
        ).eval()
        logger.info("model loaded: {}", model_path)

    @bentoml.api(batchable=True)
    @logger.catch(reraise=True)
    @torch.inference_mode()
    def embed(self, queries: list[Query]) -> list[Query]:
        """Embed a batch of queries using the transformer model.

        This method expects a list of `Query` objects. It extracts the
        `text` field from each query, computes embeddings with the underlying
        SentenceTransformer, and writes back the resulting embedding into the
        `embedding` attribute of each `Query` object.

        Args:
            queries (list[Query]): A batch of queries to embed. Each query's
                `text` attribute may be ``None``; the transformer will accept
                ``None`` as empty strings depending on its API.

        Returns:
            list[Query]: The same list of queries with `embedding` populated.
        """
        texts = [query.text for query in queries]
        embeddings = self.model.encode(texts)
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
    model_ref = bentoml.models.BentoModel(MODEL_NAME)
    model = bentoml.depends(Model)
    item_index = bentoml.depends(ItemIndex)
    user_index = bentoml.depends(UserIndex)

    @bentoml.api()
    @logger.catch(reraise=True)
    async def recommend_with_query(self, query: Query) -> list[ItemCandidate]:
        """Recommend items for a given textual query.

        This method ensures the query is embedded (calling `embed_query`),
        merges any provided `exclude_item_ids` with item ids present on the
        query itself, and delegates the search to `search_items`.

        Args:
            query (Query): The input query containing optional `text` or
                `item_ids`.
            exclude_item_ids (list[str] | None): Optional list of item ids to
                exclude from the recommendations.
            top_k (int): Number of top candidates to return.

        Returns:
            list[ItemCandidate]: A list of recommended item candidates.
        """
        query = await self.embed_query(query)
        query.exclude_item_ids = [
            *(query.exclude_item_ids or []),
            *(query.item_ids or []),
        ]
        return await self.search_items(query)

    @bentoml.api()
    @logger.catch(reraise=True)
    async def embed_query(self, query: Query) -> Query:
        """Ensure a Query object has an embedding.

        If the `query` already contains an `embedding`, it is returned
        unchanged. Otherwise this method delegates to the Model service's async
        `embed` method to produce an embedding for the query.

        Args:
            query (Query): Query to embed.

        Returns:
            Query: The same query with `embedding` populated.
        """
        if query.text is None:
            return query
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
        """Recommend items given an ItemQuery object.

        Converts the provided `ItemQuery` into an internal `Query` using
        `process_item` and then calls `recommend_with_query`.

        Args:
            item (ItemQuery): The source item to base recommendations on.
            exclude_item_ids (list[str] | None): Optional ids to exclude.
            top_k (int): Number of results to return.

        Returns:
            list[ItemCandidate]: Recommended items.
        """
        query = await self.process_item(
            item, exclude_item_ids=exclude_item_ids, top_k=top_k
        )
        return await self.recommend_with_query(query)

    @bentoml.api()
    @logger.catch(reraise=True)
    async def process_item(
        self,
        item: ItemQuery,
        exclude_item_ids: list[str] | None = None,
        top_k: int = TOP_K,
    ) -> Query:
        """Convert an ItemQuery into an internal Query.

        Args:
            item (ItemQuery): Item to convert.

        Returns:
            Query: A Query containing the single item id and its text.
        """
        return Query(
            item_ids=[item.item_id],
            text=item.item_text,
            exclude_item_ids=exclude_item_ids,
            top_k=top_k,
        )

    @bentoml.api()
    @logger.catch(reraise=True)
    async def recommend_with_item_id(
        self,
        item_id: str,
        exclude_item_ids: list[str] | None = None,
        top_k: int = TOP_K,
    ) -> list[ItemCandidate]:
        """Recommend items for a given item id.

        Fetches the `ItemQuery` for the given `item_id` and forwards to
        `recommend_with_item`.

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
        """Asynchronously retrieve an item by id using the item index.

        This method proxies the call to the item index dependency.

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
        """Recommend items for a given user object.

        Converts the `UserQuery` into an internal `Query` (collecting history
        and target item lists) and delegates to `recommend_with_query`.

        Args:
            user (UserQuery): The user object with optional history/target.
            exclude_item_ids (list[str] | None): Optional ids to exclude.
            top_k (int): Number of results to return.

        Returns:
            list[ItemCandidate]: Recommended items for the user.
        """
        query = await self.process_user(
            user, exclude_item_ids=exclude_item_ids, top_k=top_k
        )
        return await self.recommend_with_query(query)

    @bentoml.api()
    @logger.catch(reraise=True)
    async def process_user(
        self,
        user: UserQuery,
        exclude_item_ids: list[str] | None = None,
        top_k: int = TOP_K,
    ) -> Query:
        """Convert a UserQuery into an internal Query.

        This aggregates item ids and texts from the user's history and target
        lists and includes the user's profile `user_text` as the query text.

        Args:
            user (UserQuery): The user to process.

        Returns:
            Query: Aggregated Query object representing the user's context.
        """
        item_ids: list[str] = []
        item_texts: list[str] = []
        if user.history:
            item_ids += user.history.item_id
            item_texts += user.history.item_text
        if user.target:
            item_ids += user.target.item_id
            item_texts += user.target.item_text

        return Query(
            item_ids=item_ids,
            text=user.user_text,
            exclude_item_ids=exclude_item_ids,
            top_k=top_k,
        )

    @bentoml.api()
    @logger.catch(reraise=True)
    async def recommend_with_user_id(
        self,
        user_id: str,
        exclude_item_ids: list[str] | None = None,
        top_k: int = TOP_K,
    ) -> list[ItemCandidate]:
        """Recommend items for a given user id.

        Fetches the user via `user_id` and delegates to
        `recommend_with_user`.

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
