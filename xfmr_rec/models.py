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
    bert_model = init_bert(config)
    return to_sent_transformer(config, bert_model, device=device)
