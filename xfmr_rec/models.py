from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING, Literal

import pydantic
from sentence_transformers import SentenceTransformer, models
from transformers import AutoTokenizer
from transformers.models.bert import BertConfig, BertModel

from xfmr_rec.params import PRETRAINED_MODEL_NAME

if TYPE_CHECKING:
    import torch
    from transformers.modeling_utils import PreTrainedModel


class ModelConfig(pydantic.BaseModel):
    vocab_size: int | None = None
    hidden_size: int = 384
    num_hidden_layers: int = 3
    num_attention_heads: int = 12
    intermediate_size: int = 1536
    hidden_act: Literal["gelu", "relu", "silu", "gelu_new"] = "gelu"
    max_seq_length: int | None = None
    is_decoder: bool = False

    tokenizer_name: str = PRETRAINED_MODEL_NAME
    pooling_mode: Literal["mean", "max", "cls", "lasttoken"] = "mean"


def init_bert(config: ModelConfig) -> BertModel:
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)  # nosec

    if config.vocab_size is None:
        config.vocab_size = tokenizer.vocab_size

    if config.max_seq_length is None:
        config.max_seq_length = tokenizer.model_max_length

    bert_config = BertConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        hidden_act=config.hidden_act,
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

        transformer = models.Transformer(
            tmpdir, tokenizer_name_or_path=config.tokenizer_name
        )
        pooling = models.Pooling(
            transformer.get_word_embedding_dimension(), pooling_mode=config.pooling_mode
        )
        normalize = models.Normalize()

        return SentenceTransformer(
            modules=[transformer, pooling, normalize], device=device
        )


def init_sent_transformer(
    config: ModelConfig, device: torch.device | str | None = "cpu"
) -> SentenceTransformer:
    bert_model = init_bert(config)
    return to_sent_transformer(config, bert_model, device=device)
