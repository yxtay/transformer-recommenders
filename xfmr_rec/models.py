from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pydantic

from xfmr_rec.params import PRETRAINED_MODEL_NAME

if TYPE_CHECKING:
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers.modeling_utils import PreTrainedModel
    from transformers.models.bert import BertModel


class ModelConfig(pydantic.BaseModel):
    vocab_size: int | None = None
    hidden_size: int = 384
    num_hidden_layers: int = 3
    num_attention_heads: int = 12
    intermediate_size: int = 1536
    hidden_act: Literal["gelu", "relu", "silu", "gelu_new"] = "gelu"
    max_position_embeddings: int | None = None
    is_decoder: bool = False

    tokenizer_name: str = PRETRAINED_MODEL_NAME
    pooling_mode: Literal["mean", "max", "cls", "lasttoken"] = "mean"


def init_bert(config: ModelConfig) -> BertModel:
    from transformers import AutoTokenizer
    from transformers.models.bert import BertConfig, BertModel

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    if config.vocab_size is None:
        config.vocab_size = tokenizer.vocab_size

    if config.max_position_embeddings is None:
        config.max_position_embeddings = tokenizer.model_max_length

    bert_config = BertConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        hidden_act=config.hidden_act,
        max_position_embeddings=config.max_position_embeddings,
        is_decoder=config.is_decoder,
    )
    return BertModel(bert_config)


def to_sentence_transformer(
    model: PreTrainedModel,
    *,
    tokenizer_name: str = "google-bert/bert-base-uncased",
    pooling_mode: str = "mean",
    device: torch.device | str | None = "cpu",
) -> SentenceTransformer:
    import tempfile

    from sentence_transformers import SentenceTransformer, models

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir)

        transformer = models.Transformer(tmpdir, tokenizer_name_or_path=tokenizer_name)
        pooling = models.Pooling(
            transformer.get_word_embedding_dimension(), pooling_mode=pooling_mode
        )
        normalize = models.Normalize()

        return SentenceTransformer(
            modules=[transformer, pooling, normalize], device=device
        )
