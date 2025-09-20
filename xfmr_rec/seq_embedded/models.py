from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as torch_fn
from loguru import logger
from sentence_transformers import SentenceTransformer

from xfmr_rec.models import ModelConfig, init_bert, to_sentence_transformer

if TYPE_CHECKING:
    import datasets


class SeqEmbeddedRecModelConfig(ModelConfig):
    vocab_size: int | None = 1
    hidden_size: int = 384
    num_hidden_layers: int = 1
    num_attention_heads: int = 12
    intermediate_size: int = 384
    max_seq_length: int | None = 32


class SeqEmbeddedRecModel(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        device: torch.device | str | None = None,
        model: SentenceTransformer | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = model
        self.embeddings: torch.nn.Embedding | None = None

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
            bert_model = init_bert(self.config)
            self.model = to_sentence_transformer(self.config, bert_model, device=device)

    def configure_embeddings(self, items_dataset: datasets.Dataset) -> None:
        if self.embeddings is None:
            import pandas as pd

            self.id2idx = pd.Series(
                {k: i + 1 for i, k in enumerate(items_dataset["item_id"])}
            )

            weights = items_dataset.with_format("torch")["embedding"][:]
            # add idx 0 for padding
            weights = torch.cat([torch.zeros_like(weights[[0], :]), weights])
            self.embeddings = torch.nn.Embedding.from_pretrained(
                weights, freeze=True, padding_idx=0
            ).to(self.device)

    def save(self, path: str) -> None:
        self.model.save(path)
        logger.info(f"model saved: {path}")

    @classmethod
    def load(
        cls, path: str, device: torch.device | str | None = None
    ) -> SeqEmbeddedRecModel:
        model = SentenceTransformer(path, device=device, local_files_only=True)
        logger.info(f"model loaded: {path}")

        tokenizer_name = model[0].tokenizer.name_or_path
        pooling_mode = model[1].get_pooling_mode_str()
        model_conf = model[0].auto_model.config
        config = SeqEmbeddedRecModelConfig.model_validate(
            model_conf, from_attributes=True
        ).model_copy(
            update={
                "max_seq_length": model_conf.max_position_embeddings,
                "tokenizer_name": tokenizer_name,
                "pooling_mode": pooling_mode,
            }
        )
        return cls(config, model=model)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        input_embeds: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if input_embeds is not None:
            input_embeds = input_embeds[:, -self.max_seq_length :, :].to(self.device)
        elif input_ids is not None:
            input_embeds = self.embeddings(
                input_ids[:, -self.max_seq_length :].to(self.device)
            )
        else:
            msg = "either `input_ids` or `input_embeds` must be provided"
            raise ValueError(msg)

        attention_mask = (input_embeds != 0).any(-1).long()
        features = {"inputs_embeds": input_embeds, "attention_mask": attention_mask}
        return self.model(features)

    def encode(self, item_ids: list[str]) -> torch.Tensor:
        item_ids = [item_id for item_id in item_ids if item_id in self.id2idx.index]
        item_idx = torch.as_tensor(self.id2idx[item_ids].to_numpy())
        return self(item_idx[None, :])["sentence_embedding"][0]

    def compute_loss(
        self,
        history_item_idx: torch.Tensor,
        pos_item_idx: torch.Tensor,
        neg_item_idx: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        output = self(history_item_idx)
        attention_mask = output["attention_mask"].bool()
        # shape: (batch_size, seq_len)
        output_embeds = output["token_embeddings"]
        # shape: (batch_size, seq_len, hidden_size)
        output_embeds = output_embeds[attention_mask][:, None, :]
        # shape: (batch_size * seq_len, 1, hidden_size)
        output_embeds = torch_fn.normalize(output_embeds, dim=-1)
        # shape: (batch_size * seq_len, 1, hidden_size)

        pos_item_idx = pos_item_idx[attention_mask]
        # shape: (batch_size * seq_len)
        neg_item_idx = neg_item_idx[attention_mask]
        # shape: (batch_size * seq_len)
        candidate_idx = torch.stack([pos_item_idx, neg_item_idx], dim=-1)
        # shape: (batch_size * seq_len, 2)
        candidate_embeds = self.embeddings(candidate_idx)
        # shape: (batch_size * seq_len, 2, hidden_size)

        logits = (output_embeds * candidate_embeds).sum(dim=-1)
        # shape: (batch_size * seq_len, 2)
        labels_bce = torch.zeros_like(logits)
        # positive item is always at zero index
        labels_bce[:, 0] = 1
        # shape: (batch_size * seq_len, 2)
        loss_bce = torch_fn.binary_cross_entropy_with_logits(
            logits, labels_bce, reduction="sum"
        )

        # positive item is always at zero index
        labels_ce = torch.zeros_like(logits[:, 0], dtype=torch.long)
        # shape: (batch_size * seq_len)
        loss_ce = torch_fn.cross_entropy(logits, labels_ce, reduction="sum")

        batch_size, seq_len = attention_mask.size()
        numel = attention_mask.numel()
        non_zero = logits.size(0)
        return {
            "batch/size": batch_size,
            "batch/seq_len": seq_len,
            "batch/numel": numel,
            "batch/non_zero": non_zero,
            "batch/sparsity": non_zero / (numel + 1e-10),
            "loss/binary_cross_entropy": loss_bce,
            "loss/binary_cross_entropy_mean": loss_bce / (non_zero + 1e-10),
            "loss/cross_entropy": loss_ce,
            "loss/cross_entropy_mean": loss_ce / (non_zero + 1e-10),
        }
