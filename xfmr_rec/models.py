from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING, Literal

import pydantic
from sentence_transformers import SentenceTransformer, models
from transformers import AutoModel, AutoTokenizer
from transformers.models.bert import BertConfig, BertModel

from xfmr_rec.params import PRETRAINED_MODEL_NAME

if TYPE_CHECKING:
    import torch
    from transformers.modeling_utils import PreTrainedModel


class ModelConfig(pydantic.BaseModel):
    """Simple pydantic container for model hyperparameters.

    Fields correspond to common transformer configuration options. Any
    unset fields may be inferred from a pretrained model when
    ``init_bert`` is called.

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

    vocab_size: int | None = None
    hidden_size: int | None = None
    num_hidden_layers: int | None = None
    num_attention_heads: int | None = None
    intermediate_size: int | None = None
    max_seq_length: int | None = None
    is_decoder: bool = False

    pretrained_model_name: str = PRETRAINED_MODEL_NAME
    pooling_mode: Literal["mean", "max", "cls", "lasttoken"] = "mean"
    is_normalized: bool = True


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
        modules = [
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
