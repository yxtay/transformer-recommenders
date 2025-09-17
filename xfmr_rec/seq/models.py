from __future__ import annotations

import pathlib
from typing import Literal

import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from xfmr_rec.models import ModelConfig, init_bert, to_sentence_transformer
from xfmr_rec.params import EMBEDDER_PATH, ENCODER_PATH


class SeqRecModelConfig(ModelConfig):
    vocab_size: int | None = 1
    hidden_size: int = 32
    num_hidden_layers: int = 1
    num_attention_heads: int = 4
    intermediate_size: int = 32
    max_position_embeddings: int | None = 32

    pooling_mode: Literal["mean", "max", "cls", "lasttoken"] = "lasttoken"


class SeqRecModel(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        device: torch.device | str | None = None,
        embedder: SentenceTransformer | None = None,
        encoder: SentenceTransformer | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.embedder = embedder
        self.encoder = encoder

        self.configure_model(device=device)
        logger.info(f"{self.__class__.__name__}: {self.config}")
        logger.info(f"{self}")

    def configure_model(self, device: torch.device | str | None = None) -> None:
        if self.encoder is None:
            encoder_conf = self.config.model_copy(update={"is_decoder": True})
            encoder = init_bert(encoder_conf)
            self.encoder = to_sentence_transformer(
                encoder, pooling_mode="lasttoken", device=device
            )

        if self.embedder is None:
            embedding_conf = self.config.model_copy(
                update={
                    "vocab_size": None,
                    "max_position_embeddings": None,
                    "pooling_mode": "mean",
                }
            )
            embedder = init_bert(embedding_conf)
            self.embedder = to_sentence_transformer(embedder, device=self.device)

    @property
    def device(self) -> torch.device:
        return self.encoder.device

    @property
    def max_seq_length(self) -> int:
        return self.encoder.max_seq_length

    def save(self, path: str) -> None:
        path = pathlib.Path(path)
        embedder_path = (path / EMBEDDER_PATH).as_posix()
        self.embedder.save_pretrained(embedder_path)
        logger.info(f"embedder saved: {embedder_path}")

        encoder_path = (path / ENCODER_PATH).as_posix()
        self.encoder.save_pretrained(encoder_path)
        logger.info(f"encoder saved: {encoder_path}")

    @classmethod
    def load(cls, path: str, device: torch.device | str | None = None) -> SeqRecModel:
        path = pathlib.Path(path)
        encoder_path = (path / ENCODER_PATH).as_posix()
        encoder = SentenceTransformer(
            encoder_path, device=device, local_files_only=True
        )
        logger.info(f"encoder loaded: {encoder_path}")

        embedder_path = (path / EMBEDDER_PATH).as_posix()
        embedder = SentenceTransformer(
            embedder_path, device=encoder.device, local_files_only=True
        )
        logger.info(f"embedder loaded: {embedder_path}")

        tokenizer_name = embedder[0].tokenizer.name_or_path
        pooling_mode = embedder[1].get_pooling_mode_str()
        encoder_conf = encoder[0].auto_model.config
        config = SeqRecModelConfig.model_validate(
            encoder_conf, from_attributes=True
        ).model_copy(
            update={"tokenizer_name": tokenizer_name, "pooling_mode": pooling_mode}
        )
        return cls(config, embedder=embedder, encoder=encoder)

    def embed_item_text(self, item_text: list[str]) -> torch.Tensor:
        tokenized = self.embedder.tokenize(item_text)
        tokenized = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in tokenized.items()
        }
        return self.embedder(tokenized)["sentence_embedding"]

    def embed_item_text_sequence(
        self, item_text_sequence: list[list[str]]
    ) -> torch.Tensor:
        import itertools

        from torch.nn.utils.rnn import pad_sequence

        item_text_sequence = [seq[-self.max_seq_length :] for seq in item_text_sequence]
        num_items = [len(seq) for seq in item_text_sequence]
        embeddings = self.embed_item_text(list(itertools.chain(*item_text_sequence)))
        return pad_sequence(torch.split(embeddings, num_items), batch_first=True)

    def forward(
        self, item_text_sequence: list[list[str]] | None = None
    ) -> dict[str, torch.Tensor]:
        inputs_embeds = self.embed_item_text_sequence(item_text_sequence)
        attention_mask = (inputs_embeds != 0).any(-1).long()
        features = {"attention_mask": attention_mask, "inputs_embeds": inputs_embeds}
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

        pos_embeds = self.embed_item_text_sequence(pos_item_text)[attention_mask]
        # shape: (batch_size * seq_len, hidden_size)
        neg_embeds = self.embed_item_text_sequence(neg_item_text)[attention_mask]
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
            logits, labels_bce, reduction="sum"
        )

        # positive item is always at zero index
        labels_ce = torch.zeros_like(logits[:, 0], dtype=torch.long)
        # shape: (batch_size * seq_len)
        loss_ce = torch.nn.functional.cross_entropy(logits, labels_ce, reduction="sum")

        batch_size, seq_len = attention_mask.size()
        numel = attention_mask.numel()
        non_zero = (attention_mask != 0).sum().item()
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
