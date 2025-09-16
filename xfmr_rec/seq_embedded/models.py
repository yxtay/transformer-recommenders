from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from loguru import logger

from xfmr_rec.models import ModelConfig, init_bert, to_sentence_transformer

if TYPE_CHECKING:
    import datasets


class SeqEmbeddedRecModelConfig(ModelConfig):
    hidden_size: int = 384
    num_hidden_layers: int = 1
    num_attention_heads: int = 12
    intermediate_size: int = 384
    max_seq_length: int = 32


class SeqEmbeddedRecModel(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        items_dataset: datasets.Dataset,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.config = config

        # index 0 is for padding
        self.item_id_map = {k: i + 1 for i, k in enumerate(items_dataset["item_id"])}
        self.item_embeddings = self.load_embeddings(items_dataset)

        bert_model = init_bert(self.config)
        self.model = to_sentence_transformer(bert_model, device=device)
        self.item_embeddings.to(self.device)

        logger.info(f"{self.__class__.__name__}: {self.config}")
        logger.info(f"{self}")

    @property
    def device(self):
        return self.model.device

    def load_embeddings(self, items_dataset: datasets.Dataset) -> torch.nn.Embedding:
        weights = items_dataset.with_format("torch")["embedding"][:]
        # add padding embedding at idx 0
        weights = torch.cat([torch.zeros_like(weights[[0], :]), weights])
        return torch.nn.Embedding.from_pretrained(weights, freeze=True, padding_idx=0)

    def save(self, path: str) -> None:
        self.model.save(path)

    def forward(
        self,
        item_idx: torch.Tensor | None = None,
        item_embeds: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if item_idx is None and item_embeds is None:
            msg = "Either item_idx or item_embeds must be provided."
            raise ValueError(msg)

        if item_embeds is None:
            inputs_embeds = self.item_embeddings(
                item_idx[:, -self.config.max_seq_length :]
            )
            attention_mask = (item_idx != 0).long()
        else:
            inputs_embeds = item_embeds[:, -self.config.max_seq_length :, :]
            attention_mask = (item_embeds != 0).any(-1).long()

        features = {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}
        return self.model(features)

    def encode(self, item_ids: list[str]) -> torch.Tensor:
        item_idx = [
            self.item_id_map[item] for item in item_ids if item in self.item_id_map
        ]
        item_idx = torch.as_tensor(
            [item_idx[-self.config.max_seq_length :]], device=self.model.device
        )
        return self(item_idx)["sentence_embedding"][0]

    def compute_loss(
        self,
        item_idx: torch.Tensor,
        pos_idx: torch.Tensor,
        neg_idx: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        loss_mask = torch.where(pos_idx != 0)
        # shape: (batch_size, seq_len)
        output_embeds = self(item_idx)["token_embeddings"]
        # shape: (batch_size, seq_len, hidden_size)
        output_embeds = output_embeds[:, :, None, :][loss_mask]
        # shape: (batch_size * seq_len, 1, hidden_size)

        candidate_idx = torch.stack([pos_idx, neg_idx], dim=-1)
        # shape: (batch_size, seq_len, 2)
        candidate_idx = candidate_idx[loss_mask]
        # shape: (batch_size * seq_len, 2)
        candidate_embeddings = self.item_embeddings(candidate_idx)
        # shape: (batch_size * seq_len, 2, hidden_size)

        logits = (output_embeds * candidate_embeddings).sum(dim=-1)
        # shape: (batch_size * seq_len, 2)
        labels_bce = torch.zeros_like(logits)
        # positive item is always at zero index
        labels_bce[:, 0] = 1
        # shape: (batch_size * seq_len, 2)
        loss_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels_bce
        )

        # positive item is always at zero index
        labels_ce = torch.zeros_like(candidate_idx[:, 0])
        # shape: (batch_size * seq_len)
        loss_ce = torch.nn.functional.cross_entropy(logits, labels_ce)

        batch_size, seq_len = pos_idx.size()
        numel = pos_idx.numel()
        non_zero = (pos_idx != 0).sum().item()
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
