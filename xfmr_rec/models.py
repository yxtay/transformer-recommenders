from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING, Literal

import pandas as pd
import pydantic
import torch
import torch.nn.functional as torch_fn
from loguru import logger
from sentence_transformers import SentenceTransformer, models
from transformers import AutoModel, AutoTokenizer
from transformers.models.bert import BertConfig, BertModel

from xfmr_rec.params import PRETRAINED_MODEL_NAME

if TYPE_CHECKING:
    import datasets
    from transformers.modeling_utils import PreTrainedModel


class ModelConfig(pydantic.BaseModel):
    """Configuration for the sequential transformer model.

    Attributes:
        vocab_size (int | None): Vocabulary size for the transformer.
        hidden_size (int | None): Hidden dimension of the transformer.
        num_hidden_layers (int | None): Number of transformer layers.
        num_attention_heads (int | None): Number of attention heads.
        intermediate_size (int | None): Intermediate (FFN) size.
        max_seq_length (int | None): Maximum sequence length.
        is_decoder (bool): Whether to build the model as a decoder.
        pretrained_model_name (str): Name of the pretrained HF model.
        pooling_mode (Literal): Pooling strategy for sentence embeddings.
        is_normalized (bool): Whether to apply L2 normalization.
    """

    vocab_size: int | None = 1
    hidden_size: int | None = None
    num_hidden_layers: int | None = 1
    num_attention_heads: int | None = 12
    intermediate_size: int | None = 48
    max_seq_length: int | None = 32
    is_decoder: bool = True

    pretrained_model_name: str = PRETRAINED_MODEL_NAME
    pooling_mode: Literal["mean", "max", "cls", "lasttoken"] = "mean"
    is_normalized: bool = False


def init_bert(config: ModelConfig) -> BertModel:
    """Create a ``BertModel`` instance from a :class:`ModelConfig`.

    The function inspects the provided ``config`` and will attempt to
    fill missing fields by loading a pretrained tokenizer and/or model
    from HuggingFace when ``pretrained_model_name`` is specified. The
    assembled :class:`transformers.models.bert.BertConfig` is then used
    to instantiate a new :class:`transformers.models.bert.BertModel`.

    Args:
        config (ModelConfig): Configuration describing the desired model
            topology. Missing numeric fields (vocab_size, hidden_size,
            etc.) will be inferred from a pretrained model when
            ``pretrained_model_name`` is provided.

    Returns:
        BertModel: A freshly constructed ``BertModel`` instance.
    """
    tokenizer = None
    if None in (config.vocab_size, config.max_seq_length):
        tokenizer = AutoTokenizer.from_pretrained(  # nosec
            config.pretrained_model_name
        )
    if config.vocab_size is None:
        config.vocab_size = tokenizer.vocab_size
    if config.max_seq_length is None:
        config.max_seq_length = tokenizer.model_max_length

    params = [
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "intermediate_size",
    ]
    if any(getattr(config, p) is None for p in params):
        model = AutoModel.from_pretrained(  # nosec
            config.pretrained_model_name
        )
        for p in params:
            if getattr(config, p) is None:
                setattr(config, p, getattr(model.config, p))

    bert_config = BertConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        max_position_embeddings=config.max_seq_length,
        is_decoder=config.is_decoder,
    )
    return BertModel(bert_config)


def to_sent_transformer(
    config: ModelConfig,
    model: PreTrainedModel,
    *,
    device: torch.device | str | None = "cpu",
) -> SentenceTransformer:
    """Wrap a HuggingFace ``PreTrainedModel`` into a
    :class:`sentence_transformers.SentenceTransformer` pipeline.

    The provided HF ``model`` is saved to a temporary directory and then
    loaded into a ``sentence_transformers.models.Transformer`` module.
    A pooling layer is appended according to ``config.pooling_mode`` and
    an optional normalization layer is added if ``config.is_normalized``
    is True.

    Args:
        config (ModelConfig): Model configuration that controls pooling
            and normalization behaviour.
        model (PreTrainedModel): HuggingFace model instance to wrap.

    Keyword Args:
        device (torch.device | str | None): Device or device string to
            place the returned SentenceTransformer on. Defaults to
            ``"cpu"``.

    Returns:
        SentenceTransformer: Ready-to-use sentence transformer which
            accepts text or tensor inputs and produces fixed-size
            embeddings.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir)
        modules: list[torch.nn.Module] = [
            models.Transformer(
                tmpdir, tokenizer_name_or_path=config.pretrained_model_name
            )
        ]

    modules.append(
        models.Pooling(model.config.hidden_size, pooling_mode=config.pooling_mode)
    )
    if config.is_normalized:
        modules.append(models.Normalize())

    return SentenceTransformer(modules=modules, device=device)


def init_sent_transformer(
    config: ModelConfig, device: torch.device | str | None = "cpu"
) -> SentenceTransformer:
    """Initialise a :class:`SentenceTransformer` instance from a
    :class:`ModelConfig`.

    This is a convenience wrapper that first creates a ``BertModel`` via
    :func:`init_bert` and then wraps it using
    :func:`to_sent_transformer` to produce a SentenceTransformer with
    pooling and optional normalization.

    Args:
        config (ModelConfig): Model configuration used to build the
            underlying BERT model and pooling behaviour.
        device (torch.device | str | None): Device or device string for
            the returned SentenceTransformer. Defaults to ``"cpu"``.

    Returns:
        SentenceTransformer: The initialised sentence-transformer model.
    """
    bert_model = init_bert(config)
    return to_sent_transformer(config, bert_model, device=device)


class RecommenderModel(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        device: torch.device | str | None = None,
        model: SentenceTransformer | None = None,
    ) -> None:
        """Initialize the Recommender model.

        Args:
            config (ModelConfig): Configuration for the model.
            device (torch.device | str | None): Device where the model
                should be allocated.
            model (SentenceTransformer | None): Pre-initialized
                SentenceTransformer model.
        """
        super().__init__()
        self.config = config
        self.model = model
        self.embeddings: torch.nn.Embedding | None = None
        self.id2idx: pd.Series[int] | None = None

        self.configure_model(device=device)
        logger.info(repr(self.config))
        logger.info(self)

    @property
    def device(self) -> torch.device:
        """Device property delegated to the underlying SentenceTransformer.

        Returns:
            torch.device: Device used by the SentenceTransformer model.
        """
        assert self.model is not None
        return self.model.device

    @property
    def max_seq_length(self) -> int:
        """Maximum sequence length supported by the model.

        Returns:
            int: Maximum sequence length (tokens) for the model.
        """
        assert self.model is not None
        return self.model.max_seq_length

    def configure_model(self, device: torch.device | str | None = None) -> None:
        """Initialise the SentenceTransformer model if not provided.

        Args:
            device (torch.device | str | None): Device where the model
                should be allocated. If ``None`` the SentenceTransformer
                default will be used.
        """
        if self.model is None:
            self.model = init_sent_transformer(self.config, device=device)

    def configure_embeddings(self, items_dataset: datasets.Dataset) -> None:
        """Configure pretrained item embeddings and id-to-index mapping.

        This method extracts item embeddings from the provided dataset,
        prepends a zero vector for padding (index 0) and builds a frozen
        torch.nn.Embedding layer. It also constructs a pandas Series
        mapping item IDs to embedding indices.

        Args:
            items_dataset (datasets.Dataset): Dataset containing at least
                the columns ``"embedding"`` (tensor/array) and
                ``"item_id"``.
        """
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
        """Save the SentenceTransformer model to disk.

        Args:
            path (str): Path where the SentenceTransformer should be saved.
        """
        assert self.model is not None
        self.model.save(path)
        logger.info(f"model saved: {path}")

    @classmethod
    def load(cls, path: str, device: torch.device | str | None = None) -> RecommenderModel:
        """Load a RecommenderModel from a saved SentenceTransformer.

        The method inspects the saved model to reconstruct a
        :class:`ModelConfig` and returns an initialised
        :class:`RecommenderModel` instance.

        Args:
            path (str): Path to the saved SentenceTransformer directory.
            device (torch.device | str | None): Device to load the model on.

        Returns:
            RecommenderModel: Initialised model instance with reconstructed
                configuration.
        """
        model = SentenceTransformer(path, device=device, local_files_only=True)
        logger.info(f"model loaded: {path}")

        tokenizer_name = model[0].tokenizer.name_or_path
        pooling_mode = model[1].pooling_mode
        model_conf = model[0].auto_model.config
        config = ModelConfig.model_validate(
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
        """Forward pass that accepts either item indices or item embeddings.

        Exactly one of ``item_idx`` or ``item_embeds`` must be provided.
        The input is truncated to ``max_seq_length`` and an attention mask
        is created from non-zero embeddings before forwarding to the
        SentenceTransformer model.

        Args:
            item_idx (torch.Tensor | None): Tensor of item indices with
                shape (batch_size, seq_len).
            item_embeds (torch.Tensor | None): Precomputed item embeddings
                with shape (batch_size, seq_len, hidden_size).

        Returns:
            dict[str, torch.Tensor]: Dictionary output from the underlying
                SentenceTransformer (contains keys like
                ``sentence_embedding`` and ``token_embeddings``).
        """
        assert self.embeddings is not None
        assert self.model is not None

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
        """Encode a list of item IDs into their pooled sequence embedding.

        Item IDs not present in the configured ``id2idx`` mapping are
        silently dropped. The returned vector corresponds to the pooled
        sentence embedding for the sequence of provided item IDs.

        Args:
            item_ids (list[str]): List of item identifier strings.

        Returns:
            torch.Tensor: 1D tensor with the pooled embedding for the
                provided item IDs.
        """
        assert self.id2idx is not None
        item_ids = [item_id for item_id in item_ids if item_id in self.id2idx.index]
        item_idx = torch.as_tensor(self.id2idx[item_ids].to_numpy(), device=self.device)
        return self(item_idx[None, :])["sentence_embedding"][0]

    def compute_embeds(
        self,
        history_item_idx: torch.Tensor,
        pos_item_idx: torch.Tensor,
        neg_item_idx: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute query and candidate embeddings used for training.

        Args:
            history_item_idx (torch.Tensor): Tensor of history item
                indices with shape (batch_size, seq_len).
            pos_item_idx (torch.Tensor): Tensor of positive item indices
                aligned with the history positions.
            neg_item_idx (torch.Tensor): Tensor of negative item indices
                aligned with the history positions.

        Returns:
            dict[str, torch.Tensor]: Dictionary with keys:
                - ``query_embed``: tensor of shape (num_valid_positions, hidden_size)
                - ``candidate_embed``: tensor of shape (num_valid_positions, 1 + num_candidates, hidden_size)
                - ``attention_mask``: boolean mask of valid token positions
        """
        assert self.embeddings is not None
        output = self(history_item_idx)
        attention_mask = output["attention_mask"].bool()
        # shape: (batch_size, seq_len)
        query_embed = output["token_embeddings"][attention_mask]
        # shape: (batch_size * seq_len, hidden_size)
        if self.config.is_normalized:
            query_embed = torch_fn.normalize(query_embed, dim=-1)
        # shape: (batch_size * seq_len, hidden_size)

        pos_item_idx = pos_item_idx[attention_mask]
        # shape: (batch_size * seq_len)
        pos_embed = self.embeddings(pos_item_idx)
        # shape: (batch_size * seq_len, hidden_size)
        pos_embed = pos_embed[:, None, :]
        # shape: (batch_size * seq_len, 1, hidden_size)
        neg_item_idx = neg_item_idx[attention_mask]
        # shape: (batch_size * seq_len)
        neg_embed = self.embeddings(neg_item_idx)
        # shape: (batch_size * seq_len, hidden_size)
        neg_embed = neg_embed[None, :, :].expand(pos_embed.size(0), -1, -1)
        # shape: (batch_size * seq_len, batch_size * seq_len, hidden_size)
        candidate_embed = torch.cat([pos_embed, neg_embed], dim=1)
        # shape: (batch_size * seq_len, 1 + batch_size * seq_len, hidden_size)
        # remove samples with padding as positive
        pos_mask = pos_item_idx != 0
        return {
            "query_embed": query_embed[pos_mask],
            "candidate_embed": candidate_embed[pos_mask],
            "attention_mask": attention_mask,
            "positive_mask": pos_mask,
        }
