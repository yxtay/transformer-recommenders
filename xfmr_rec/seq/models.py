from __future__ import annotations

import datetime
import pathlib
from typing import TYPE_CHECKING

import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from xfmr_rec.models import ModelConfig, init_bert, to_sentence_transformer

if TYPE_CHECKING:
    import datasets
    import lancedb
    import numpy as np


class SeqRecModelConfig(ModelConfig):
    vocab_size: int | None = 1
    hidden_size: int = 32
    num_hidden_layers: int = 1
    num_attention_heads: int = 4
    intermediate_size: int = 32
    max_position_embeddings: int | None = 32


class SeqRecModel(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.config = config

        embedding_conf = self.config.model_copy(
            update={"vocab_size": None, "max_position_embeddings": None}
        )
        embedder = init_bert(embedding_conf)
        self.embedder = to_sentence_transformer(embedder, device=device)

        encoder_conf = self.config.model_copy(
            update={"is_decoder": True, "pooling_mode": "lasttoken"}
        )
        encoder = init_bert(encoder_conf)
        self.encoder = to_sentence_transformer(
            encoder, pooling_mode="lasttoken", device=device
        )

        logger.info(f"{self.__class__.__name__}: {self.config}")
        logger.info(f"{self}")

    @property
    def device(self) -> torch.device:
        return self.encoder.device

    @property
    def max_seq_length(self) -> int:
        return self.encoder.max_seq_length

    def save(self, path: str) -> None:
        path = pathlib.Path(path)
        self.embedder.save_pretrained(path / "embedder")
        self.encoder.save_pretrained(path / "encoder")

    def load(self, path: str) -> None:
        path = pathlib.Path(path)
        self.embedder = SentenceTransformer(path / "embedder")
        self.encoder = SentenceTransformer(path / "encoder")

    def embed_items(self, item_texts: list[str]) -> torch.Tensor:
        tokenized = self.embedder.tokenize(item_texts)
        tokenized = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in tokenized.items()
        }
        return self.embedder(tokenized)["sentence_embedding"]

    def embed_item_sequence(self, item_sequences: list[list[str]]) -> torch.Tensor:
        import itertools

        from torch.nn.utils.rnn import pad_sequence

        item_sequences = [seq[-self.max_seq_length :] for seq in item_sequences]
        num_items = [len(seq) for seq in item_sequences]
        embeddings = self.embed_items(list(itertools.chain(*item_sequences)))
        return pad_sequence(torch.split(embeddings, num_items), batch_first=True)

    def forward(
        self,
        item_texts: list[list[str]] | None = None,
        item_embeds: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if item_texts is None and item_embeds is None:
            msg = "Either item_texts or item_embeds must be provided."
            raise ValueError(msg)

        if item_embeds is None:
            inputs_embeds = self.embed_item_sequence(item_texts)
        else:
            inputs_embeds = item_embeds[:, -self.max_seq_length :, :]

        attention_mask = (inputs_embeds != 0).any(-1).short()
        features = {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}
        return self.encoder(features)

    def compute_loss(
        self,
        history_item_text: list[list[str]],
        pos_item_text: list[list[str]],
        neg_item_text: list[list[str]],
    ) -> dict[str, torch.Tensor]:
        output = self(history_item_text)
        attention_mask = output["attention_mask"].bool()
        # shape: (batch_size, seq_len)
        output_embeds = output["token_embeddings"]
        # shape: (batch_size, seq_len, hidden_size)
        output_embeds = output_embeds[:, :, None, :][attention_mask]
        # shape: (batch_size * seq_len, 1, hidden_size)

        pos_embeds = self.embed_item_sequence(pos_item_text)[attention_mask]
        # shape: (batch_size * seq_len, hidden_size)
        neg_embeds = self.embed_item_sequence(neg_item_text)[attention_mask]
        # shape: (batch_size * seq_len, hidden_size)
        candidate_embeddings = torch.stack([pos_embeds, neg_embeds], dim=-2)
        # shape: (batch_size * seq_len, 2, hidden_size)

        logits = (output_embeds * candidate_embeddings).sum(dim=-1)
        # shape: (batch_size * seq_len, 2)
        labels_bce = torch.zeros_like(logits)
        # positive item is always at zero index
        labels_bce[:, 0] = 1
        # shape: (batch_size * seq_len, 1 + seq_len * num_neg)
        loss_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels_bce
        )

        # positive item is always at zero index
        labels_ce = torch.zeros_like(logits[:, 0], dtype=torch.long)
        # shape: (batch_size * seq_len)
        loss_ce = torch.nn.functional.cross_entropy(logits, labels_ce)

        batch_size, seq_len = attention_mask.size()
        numel = attention_mask.numel()
        non_zero = (attention_mask != 0).sum().item()
        return {
            "batch/size": batch_size,
            "batch/seq_len": seq_len,
            "batch/numel": numel,
            "batch/non_zero": non_zero,
            "batch/sparsity": non_zero / numel,
            "loss/binary_cross_entropy": loss_bce,
            "loss/binary_cross_entropy_mean": loss_bce / non_zero,
            "loss/cross_entropy": loss_ce,
            "loss/cross_entropy_mean": loss_ce / non_zero,
        }


class ItemsIndex:
    def __init__(
        self,
        items_dataset: datasets.Dataset,
        *,
        lancedb_path: str = "lance_db",
        table_name: str = "items",
    ) -> None:
        super().__init__()
        self.table = self.index_items(
            items_dataset, lancedb_path=lancedb_path, table_name=table_name
        )

        logger.info(f"{self.__class__.__name__}: {self.table}")
        logger.info(
            f"num_items: {self.table.count_rows()}, columns: {self.table.schema.names}"
        )

    def index_items(
        self,
        items_dataset: datasets.Dataset,
        *,
        lancedb_path: str,
        table_name: str,
    ) -> lancedb.table.Table:
        import lancedb
        import numpy as np
        import pyarrow as pa

        num_items = len(items_dataset)
        embedding_dim = len(items_dataset["embedding"][0])

        schema = items_dataset.data.schema
        schema = schema.set(
            # scalar index does not work on large_string
            schema.get_field_index("item_id"),
            pa.field("item_id", pa.string()),
        ).set(
            # embedding column must be fixed size float array
            schema.get_field_index("embedding"),
            pa.field("embedding", pa.list_(pa.float32(), embedding_dim)),
        )

        # rule of thumb: nlist ~= 4 * sqrt(n_vectors)
        num_partitions = 2 ** int(np.log2(num_items) / 2)
        num_sub_vectors = embedding_dim // 8

        db = lancedb.connect(lancedb_path)
        table = db.create_table(
            table_name,
            data=items_dataset.data.to_batches(max_chunksize=1024),
            schema=schema,
            mode="overwrite",
        )
        table.create_scalar_index("item_id")
        table.create_index(
            vector_column_name="embedding",
            metric="cosine",
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors,
            index_type="IVF_HNSW_PQ",
        )
        table.optimize(
            cleanup_older_than=datetime.timedelta(days=0),
            delete_unverified=True,
            retrain=True,
        )
        return table

    def search(
        self,
        embedding: np.ndarray,
        exclude_item_ids: list[str] | None = None,
        top_k: int = 20,
    ) -> datasets.Dataset:
        import datasets
        import pyarrow.compute as pc

        exclude_item_ids = exclude_item_ids or [""]
        exclude_filter = ", ".join(f"'{item}'" for item in exclude_item_ids)
        exclude_filter = f"item_id NOT IN ({exclude_filter})"
        rec_table = (
            self.table.search(embedding)
            .where(exclude_filter, prefilter=True)
            .nprobes(8)
            .refine_factor(4)
            .limit(top_k)
            .select(["item_id", "_distance"])
            .to_arrow()
        )
        rec_table = rec_table.append_column(
            "score", pc.subtract(1, rec_table["_distance"])
        )
        return datasets.Dataset(rec_table)
