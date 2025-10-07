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
    hidden_size: int | None = 64
    num_hidden_layers: int | None = 1
    num_attention_heads: int | None = 2
    intermediate_size: int | None = 64
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
        """Device property delegated to the encoder.

        Returns:
            torch.device: Device used by the encoder SentenceTransformer.
        """
        assert self.encoder is not None
        return self.encoder.device

    @property
    def max_seq_length(self) -> int:
        """Maximum sequence length supported by the encoder.

        Returns:
            int: Maximum sequence length (tokens) used by the encoder.
        """
        assert self.encoder is not None
        return self.encoder.max_seq_length

    def configure_model(self, device: torch.device | str | None = None) -> None:
        """Ensure encoder and embedder SentenceTransformer instances exist.

        If an encoder or embedder is not provided at construction this
        method initialises them using :func:`init_sent_transformer`. The
        embedder uses a copy of the configuration adapted for item
        embedding (non-decoder and mean pooling).

        Args:
            device (torch.device | str | None): Device to place newly
                created SentenceTransformer instances on. If ``None`` the
                existing device settings are preserved.
        """
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

    def save(self, path: str | pathlib.Path) -> None:
        """Save the encoder and embedder SentenceTransformer models.

        The encoder and embedder are saved into separate subdirectories
        under the provided path. The encoder is saved to ``<path>/encoder``
        and the embedder to ``<path>/embedder``.

        Args:
            path (str): Directory where the model components will be saved.
        """
        assert self.encoder is not None
        assert self.embedder is not None
        path = pathlib.Path(path)
        encoder_path = (path / self.ENCODER_PATH).as_posix()
        self.encoder.save(encoder_path)
        logger.info(f"encoder saved: {encoder_path}")

        embedder_path = (path / self.EMBEDDER_PATH).as_posix()
        self.embedder.save(embedder_path)
        logger.info(f"embedder saved: {embedder_path}")

    @classmethod
    def load(
        cls, path: str | pathlib.Path, device: torch.device | str | None = None
    ) -> SeqRecModel:
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
        """Compute embeddings for a list of item text strings.

        This method tokenizes the provided item texts using the embedder's
        tokenizer, moves any tensors to the model device, and returns the
        computed sentence embeddings.

        Args:
            item_text (list[str]): List of item textual descriptions.

        Returns:
            torch.Tensor: Tensor of shape (len(item_text), hidden_size)
                containing the item embeddings.
        """
        assert self.embedder is not None
        tokenized = self.embedder.tokenize(item_text)
        tokenized = {key: value.to(self.device) for key, value in tokenized.items()}
        return self.embedder(tokenized)["sentence_embedding"]

    def embed_item_text_sequence(
        self, item_text_sequence: list[list[str]]
    ) -> torch.Tensor:
        """Embed a batch of sequences of item texts.

        Each sequence is truncated to ``max_seq_length`` from the end. The
        flattened item texts are embedded and then reshaped back into a
        padded tensor of shape (batch_size, seq_len, hidden_size).

        Args:
            item_text_sequence (list[list[str]]): Batch of item text
                sequences (one sequence per example).

        Returns:
            torch.Tensor: Padded tensor containing embeddings for each
                sequence with shape (batch_size, seq_len, hidden_size).
        """
        item_text_sequence = [seq[-self.max_seq_length :] for seq in item_text_sequence]
        num_items = [len(seq) for seq in item_text_sequence]
        embeddings = self.embed_item_text(
            list(itertools.chain.from_iterable(item_text_sequence))
        )
        return pad_sequence(list(torch.split(embeddings, num_items)), batch_first=True)

    def forward(self, item_text_sequence: list[list[str]]) -> dict[str, torch.Tensor]:
        """Encode a batch of item text sequences using the encoder.

        The method embeds the provided sequences, constructs an attention
        mask (non-zero token embeddings), and forwards the features to the
        encoder SentenceTransformer. The encoder output includes token
        embeddings and pooled sentence embeddings.

        Args:
            item_text_sequence (list[list[str]]): Batch of item text
                sequences.

        Returns:
            dict[str, torch.Tensor]: Encoder output dictionary produced by
                the SentenceTransformer model. Contains keys such as
                ``token_embeddings``, ``sentence_embedding``, and
                ``attention_mask``.
        """
        assert self.encoder is not None
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
        """Compute query and candidate embeddings used for training.

        This method encodes the history sequences to produce token-level
        query embeddings (only positions indicated by the attention mask
        are kept). Positive and negative candidate embeddings are
        computed for the same mask positions and shaped so that the first
        candidate along dimension 1 is the positive example and the
        remaining are negatives. The resulting tensors are suitable for
        computing ranking or contrastive losses.

        Args:
            history_item_text (list[list[str]]): Batch of history item
                text sequences.
            pos_item_text (list[list[str]]): Batch of positive item text
                sequences aligned to the history positions.
            neg_item_text (list[list[str]]): Batch of negative item text
                sequences aligned to the history positions.

        Returns:
            dict[str, torch.Tensor]: Dictionary with keys:
                - ``query_embed``: tensor of shape (num_valid_positions, hidden_size)
                - ``candidate_embed``: tensor of shape (num_valid_positions, 1 + num_candidates, hidden_size)
                - ``attention_mask``: boolean mask of valid token positions
        """
        output = self(history_item_text)
        attention_mask = output["attention_mask"].bool()
        # shape: (batch_size, seq_len)
        query_embed = output["token_embeddings"][attention_mask]
        # shape: (batch_size * seq_len, hidden_size)

        pos_embed = self.embed_item_text_sequence(pos_item_text)[attention_mask]
        # shape: (batch_size * seq_len, hidden_size)
        pos_embed = pos_embed[:, None, :]
        # shape: (batch_size * seq_len, 1, hidden_size)
        neg_embed = self.embed_item_text_sequence(neg_item_text)[attention_mask]
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
