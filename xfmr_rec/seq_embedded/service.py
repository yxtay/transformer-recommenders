from __future__ import annotations

import bentoml
import numpy as np
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer
from torch.nn.utils.rnn import pad_sequence

from xfmr_rec.params import TOP_K, TRANSFORMER_PATH
from xfmr_rec.seq_embedded import MODEL_NAME
from xfmr_rec.service import (
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


class Query(BaseQuery):
    item_ids: list[str] | None = None
    item_texts: list[str] | None = None
    input_embeds: NumpyArrayType | None = None


@bentoml.service()
class Model:
    model_ref = bentoml.models.BentoModel(MODEL_NAME)

    @logger.catch(reraise=True)
    def __init__(self) -> None:
        """Load the SentenceTransformer used for sequence-embedded queries.

        The transformer model is loaded from the BentoModel's `TRANSFORMER_PATH`
        artifact and set to evaluation mode.
        """
        model_path = self.model_ref.path_of(TRANSFORMER_PATH)
        self.model = SentenceTransformer(model_path).eval()
        self.embed_dim: int = self.model.get_sentence_embedding_dimension()
        logger.info("model loaded: {}", model_path)

    @bentoml.api()
    def max_seq_length(self) -> int:
        """Get the maximum sequence length supported by the transformer model.

        Returns:
            int: The maximum number of tokens the model can process.
        """
        return self.model.max_seq_length

    @bentoml.api(batchable=True)
    @logger.catch(reraise=True)
    @torch.inference_mode()
    def embed(self, queries: list[Query]) -> list[Query]:
        """Embed a batch of queries using the SentenceTransformer.

        For each query, if `input_embeds` is empty or None, returns a zero
        embedding of the expected dimensionality. Otherwise, converts the
        provided numpy embeddings into a torch tensor, truncates to the
        model's maximum sequence length (keeping the last tokens), and
        computes a sentence embedding with the transformer.

        Args:
            queries (list[Query]): Batch of queries containing either
                `input_embeds` or optional `item_texts`.

        Returns:
            list[Query]: The same queries with `embedding` populated as
                numpy arrays of shape (embedding_dim,).
        """
        inputs_embeds = [
            torch.as_tensor(query.input_embeds)
            if query.input_embeds is not None
            else torch.zeros(1, self.embed_dim)
            for query in queries
        ]
        inputs_embeds = pad_sequence(inputs_embeds, batch_first=True).to(
            self.model.device
        )[:, -self.max_seq_length() :, :]

        attention_mask = (inputs_embeds == 0).all(-1).logical_not()
        embeddings = self.model(
            {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}
        )["sentence_embedding"].numpy(force=True)

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
        """Recommend items for a sequence-embedded query.

        This method ensures the query has `input_embeds` populated
        (`process_query`), then embeds it (`embed_query`), merges exclude ids,
        and delegates to `search_items`.

        Args:
            query (Query): The input query containing optional `item_ids`,
                `item_texts`, `input_embeds`, `exclude_item_ids`, and
                `top_k`.

        Returns:
            list[ItemCandidate]: Recommended item candidates.
        """
        query = await self.process_query(query)
        query = await self.embed_query(query)
        query.exclude_item_ids = [
            *(query.exclude_item_ids or []),
            *(query.item_ids or []),
        ]
        return await self.search_items(query)

    async def process_query(self, query: Query) -> Query:
        """Populate `input_embeds` for a query when `item_ids` are provided.

        If `input_embeds` is already present or `item_ids` is None, the query is
        returned unchanged. Otherwise fetches item embeddings from the item
        index and stacks them into a numpy array (or leaves as `None` if none
        found).

        Args:
            query (Query): Query to process.

        Returns:
            Query: Query with `input_embeds` populated when possible.
        """
        if query.item_ids is None:
            return query
        if query.input_embeds is not None:
            return query

        # trim item_ids to valid ones and max_seq_len only
        items: dict[str, ItemQuery] = await self.item_index.to_async.get_ids(
            query.item_ids
        )
        item_ids = [item_id for item_id in query.item_ids if item_id in items]
        query.item_ids = item_ids[-self.model.max_seq_length() :]
        embeddings = [items[item_id].embedding for item_id in query.item_ids]
        query.input_embeds = np.stack(embeddings) if embeddings else None
        return query

    @bentoml.api()
    @logger.catch(reraise=True)
    async def embed_query(self, query: Query) -> Query:
        """Ensure a Query object has an `embedding` by delegating to the model.

        If `query.embedding` is already present it is returned; otherwise this
        method calls the model's async `embed` to compute and return the
        populated Query.

        Args:
            query (Query): Query to embed.

        Returns:
            Query: The same query with `embedding` populated.
        """
        if query.input_embeds is None:
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
        """Recommend items given an ItemQuery by converting to Query first.

        Args:
            item (ItemQuery): The item to base recommendations on.
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
        exclude_item_ids: list[str] | None,
        top_k: int,
    ) -> Query:
        """Convert an ItemQuery into a Query that contains its embedding.

        The returned Query will contain `input_embeds` set to the item's
        embedding expanded with a leading batch axis.

        Args:
            item (ItemQuery): The source item.
            exclude_item_ids (list[str] | None): Optional ids to exclude.
            top_k (int): Number of results to return.

        Returns:
            Query: Query with `item_ids`, `item_texts`, `input_embeds`, and
                any provided exclude_item_ids and top_k parameters.
        """
        assert item.embedding is not None
        return Query(
            item_ids=[item.item_id],
            item_texts=[item.item_text],
            input_embeds=item.embedding[None, :],
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
    async def recommend_with_user(
        self,
        user: UserQuery,
        exclude_item_ids: list[str] | None = None,
        top_k: int = TOP_K,
    ) -> list[ItemCandidate]:
        """Recommend items for a given user by converting to Query then
        delegating to `recommend_with_query`.

        Args:
            user (UserQuery): The user object containing history/target.
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
        exclude_item_ids: list[str] | None,
        top_k: int,
    ) -> Query:
        """Convert a UserQuery into a Query by aggregating history and target
        items.

        Args:
            user (UserQuery): The user to process.
            exclude_item_ids (list[str] | None): Optional ids to exclude.
            top_k (int): Number of results to return.

        Returns:
            Query: Aggregated Query with item ids, texts, and any provided
                exclude_item_ids and top_k parameters.
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
            item_texts=item_texts,
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
