from __future__ import annotations

import itertools
import pathlib
from typing import Literal

import torch
from loguru import logger
from sentence_transformers import SentenceTransformer
from torch.nn.utils.rnn import pad_sequence

from xfmr_rec.models import ModelConfig, init_sent_transformer


class SeqRecModelConfig(ModelConfig):
    vocab_size: int | None = 1
    hidden_size: int | None = 32
    num_hidden_layers: int | None = 1
    num_attention_heads: int | None = 4
    intermediate_size: int | None = 32
    max_seq_length: int | None = 32
    is_decoder: bool = True

    pooling_mode: Literal["mean", "max", "cls", "lasttoken"] = "lasttoken"


class SeqRecModel(torch.nn.Module):
    ENCODER_PATH = "encoder"
    EMBEDDER_PATH = "embedder"

    def __init__(
        self,
        config: SeqRecModelConfig,
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
        logger.info(repr(self.config))
        logger.info(self)

    @property
    def device(self) -> torch.device:
        return self.encoder.device

    @property
    def max_seq_length(self) -> int:
        return self.encoder.max_seq_length

    def configure_model(self, device: torch.device | str | None = None) -> None:
        if self.encoder is None:
            self.encoder = init_sent_transformer(self.config, device=device)

        if self.embedder is None:
            embedding_conf = self.config.model_copy(
                update={
                    "vocab_size": None,
                    "max_seq_length": None,
                    "is_decoder": False,
                    "pooling_mode": "mean",
                }
            )
            self.embedder = init_sent_transformer(embedding_conf, device=self.device)

    def save(self, path: str) -> None:
        path = pathlib.Path(path)
        encoder_path = (path / self.ENCODER_PATH).as_posix()
        self.encoder.save(encoder_path)
        logger.info(f"encoder saved: {encoder_path}")

        embedder_path = (path / self.EMBEDDER_PATH).as_posix()
        self.embedder.save(embedder_path)
        logger.info(f"embedder saved: {embedder_path}")

    @classmethod
    def load(cls, path: str, device: torch.device | str | None = None) -> SeqRecModel:
        path = pathlib.Path(path)
        encoder_path = (path / cls.ENCODER_PATH).as_posix()
        encoder = SentenceTransformer(
            encoder_path, device=device, local_files_only=True
        )
        logger.info(f"encoder loaded: {encoder_path}")

        embedder_path = (path / cls.EMBEDDER_PATH).as_posix()
        embedder = SentenceTransformer(
            embedder_path, device=encoder.device, local_files_only=True
        )
        logger.info(f"embedder loaded: {embedder_path}")

        tokenizer_name = embedder[0].tokenizer.name_or_path
        pooling_mode = encoder[1].get_pooling_mode_str()
        encoder_conf = encoder[0].auto_model.config
        config = SeqRecModelConfig.model_validate(
            encoder_conf, from_attributes=True
        ).model_copy(
            update={
                "max_seq_length": encoder_conf.max_position_embeddings,
                "pretrained_model_name": tokenizer_name,
                "pooling_mode": pooling_mode,
            }
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

    def compute_embeds(
        self,
        history_item_text: list[list[str]],
        pos_item_text: list[list[str]],
        neg_item_text: list[list[str]],
    ) -> dict[str, torch.Tensor]:
        output = self(history_item_text)
        attention_mask = output["attention_mask"].bool()
        # shape: (batch_size, seq_len)
        query_embed = output["token_embeddings"]
        # shape: (batch_size, seq_len, hidden_size)
        query_embed = query_embed[attention_mask]
        # shape: (batch_size * seq_len, hidden_size)

        pos_embed = self.embed_item_text_sequence(pos_item_text)[attention_mask]
        # shape: (batch_size * seq_len, hidden_size)
        neg_embed = self.embed_item_text_sequence(neg_item_text)[attention_mask]
        # shape: (batch_size * seq_len, hidden_size)
        candidate_embed = torch.cat([pos_embed, neg_embed])
        # shape: (2 * batch_size * seq_len, hidden_size)
        return {
            "query_embed": query_embed,
            "candidate_embed": candidate_embed,
            "attention_mask": attention_mask,
        }
