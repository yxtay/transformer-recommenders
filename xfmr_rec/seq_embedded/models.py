from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from xfmr_rec.models import ModelConfig, init_sent_transformer

if TYPE_CHECKING:
    import datasets


class SeqEmbeddedModelConfig(ModelConfig):
    vocab_size: int | None = 1
    num_hidden_layers: int | None = 1
    num_attention_heads: int | None = 12
    intermediate_size: int | None = 48
    max_seq_length: int | None = 32
    is_decoder: bool = True


class SeqEmbeddedModel(torch.nn.Module):
    def __init__(
        self,
        config: SeqEmbeddedModelConfig,
        *,
        device: torch.device | str | None = None,
        model: SentenceTransformer | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = model
        self.embeddings: torch.nn.Embedding | None = None
        self.id2idx: pd.Series | None = None

        self.configure_model(device=device)
        logger.info(repr(self.config))
        logger.info(self)

    @property
    def device(self) -> torch.device:
        return self.model.device

    @property
    def max_seq_length(self) -> int:
        return self.model.max_seq_length

    def configure_model(self, device: torch.device | str | None = None) -> None:
        if self.model is None:
            self.model = init_sent_transformer(self.config, device=device)

    def configure_embeddings(self, items_dataset: datasets.Dataset) -> None:
        if self.embeddings is None:
            weights = items_dataset.with_format("torch")["embedding"][:]
            # add idx 0 for padding
            weights = torch.cat([torch.zeros_like(weights[[0], :]), weights])
            self.embeddings = torch.nn.Embedding.from_pretrained(
                weights, freeze=True, padding_idx=0
            ).to(self.device)

        if self.id2idx is None:
            self.id2idx = pd.Series(
                pd.RangeIndex(len(items_dataset)) + 1,
                index=items_dataset.with_format("pandas")["item_id"].array,
            )

    def save(self, path: str) -> None:
        self.model.save(path)
        logger.info(f"model saved: {path}")

    @classmethod
    def load(
        cls, path: str, device: torch.device | str | None = None
    ) -> SeqEmbeddedModel:
        model = SentenceTransformer(path, device=device, local_files_only=True)
        logger.info(f"model loaded: {path}")

        tokenizer_name = model[0].tokenizer.name_or_path
        pooling_mode = model[1].get_pooling_mode_str()
        model_conf = model[0].auto_model.config
        config = SeqEmbeddedModelConfig.model_validate(
            model_conf, from_attributes=True
        ).model_copy(
            update={
                "max_seq_length": model_conf.max_position_embeddings,
                "pretrained_model_name": tokenizer_name,
                "pooling_mode": pooling_mode,
            }
        )
        return cls(config, model=model)

    def forward(
        self,
        item_idx: torch.Tensor | None = None,
        *,
        item_embeds: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if item_embeds is not None:
            input_embeds = item_embeds[:, -self.max_seq_length :, :].to(self.device)
        elif item_idx is not None:
            input_embeds = self.embeddings(
                item_idx[:, -self.max_seq_length :].to(self.device)
            )
        else:
            msg = "either `item_idx` or `item_embeds` must be provided"
            raise ValueError(msg)

        attention_mask = (input_embeds != 0).any(-1).long()
        features = {"inputs_embeds": input_embeds, "attention_mask": attention_mask}
        return self.model(features)

    def encode(self, item_ids: list[str]) -> torch.Tensor:
        item_ids = [item_id for item_id in item_ids if item_id in self.id2idx.index]
        item_idx = torch.as_tensor(self.id2idx[item_ids].to_numpy())
        return self(item_idx[None, :])["sentence_embedding"][0]

    def compute_embeds(
        self,
        history_item_idx: torch.Tensor,
        pos_item_idx: torch.Tensor,
        neg_item_idx: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        output = self(history_item_idx)
        attention_mask = output["attention_mask"].bool()
        # shape: (batch_size, seq_len)
        query_embed = output["token_embeddings"][attention_mask]
        # shape: (batch_size * seq_len, hidden_size)

        pos_embed = self.embeddings(pos_item_idx)[attention_mask]
        # shape: (batch_size * seq_len, hidden_size)
        pos_embed = pos_embed[:, None, :]
        # shape: (batch_size * seq_len, 1, hidden_size)
        neg_embed = self.embeddings(neg_item_idx)[attention_mask]
        # shape: (batch_size * seq_len, hidden_size)
        neg_embed = neg_embed[None, :, :].expand(pos_embed.size(0), -1, -1)
        # shape: (batch_size * seq_len, batch_size * seq_len, hidden_size)
        candidate_embed = torch.cat([pos_embed, neg_embed], dim=1)
        # shape: (batch_size * seq_len, 1 + batch_size * seq_len, hidden_size)
        return {
            "query_embed": query_embed,
            "candidate_embed": candidate_embed,
            "attention_mask": attention_mask,
        }
