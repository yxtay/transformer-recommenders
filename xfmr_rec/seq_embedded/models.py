from __future__ import annotations

import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from xfmr_rec.models import ModelConfig, init_bert, to_sentence_transformer


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
        device: torch.device | str | None = None,
        model: SentenceTransformer | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = model

        self.configure_model(device=device)
        logger.info(f"{self.__class__.__name__}: {self.config}")
        logger.info(f"{self}")

    def configure_model(self, device: torch.device | str | None = None) -> None:
        if self.model is None:
            bert_model = init_bert(self.config)
            self.model = to_sentence_transformer(bert_model, device=device)

    @property
    def device(self) -> torch.device:
        return self.model.device

    @property
    def max_seq_length(self) -> int:
        return self.model.max_seq_length

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
        model_config = model[0].auto_model.config
        config = SeqEmbeddedRecModelConfig.model_validate(
            model_config, from_attributes=True
        ).model_copy(
            update={"tokenizer_name": tokenizer_name, "pooling_mode": pooling_mode}
        )
        return cls(config, model=model)

    def forward(self, item_embeds: torch.Tensor) -> dict[str, torch.Tensor]:
        inputs_embeds = item_embeds[:, -self.max_seq_length :, :]
        attention_mask = (inputs_embeds != 0).any(-1).long()
        features = {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}
        return self.model(features)

    def compute_loss(
        self,
        item_embeddings: torch.Tensor,
        pos_embeddings: torch.Tensor,
        neg_embeddings: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        loss_mask = (pos_embeddings != 0).any(-1)
        # shape: (batch_size, seq_len)
        output_embeds = self(item_embeddings)["token_embeddings"]
        # shape: (batch_size, seq_len, hidden_size)
        output_embeds = output_embeds[:, :, None, :][loss_mask]
        # shape: (batch_size * seq_len, 1, hidden_size)

        pos_embeddings = pos_embeddings[loss_mask]
        # shape: (batch_size * seq_len, hidden_size
        neg_embeddings = neg_embeddings[loss_mask]
        # shape: (batch_size * seq_len, hidden_size
        candidate_embeddings = torch.stack([pos_embeddings, neg_embeddings], dim=-2)
        # shape: (batch_size * seq_len, 2, hidden_size

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
        labels_ce = torch.zeros_like(logits[:, 0])
        # shape: (batch_size * seq_len)
        loss_ce = torch.nn.functional.cross_entropy(logits, labels_ce)

        batch_size, seq_len = loss_mask.size()
        numel = loss_mask.numel()
        non_zero = loss_mask.sum().item()
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
