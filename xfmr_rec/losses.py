from __future__ import annotations

import abc
from typing import Literal

import pydantic
import torch
import torch.nn.functional as torch_fn


class LossConfig(pydantic.BaseModel):
    target_position: Literal["first", "diagonal"] | None = "diagonal"
    mask_hard_negatives: bool = True
    num_negatives: int = 0
    scale: float = 1.0
    margin: float = 0.5


def squared_distance_matrix(
    query_embed: torch.Tensor, candidate_embed: torch.Tensor
) -> torch.Tensor:
    return torch.cdist(query_embed, candidate_embed) ** 2 / 2


def dot_product_matrix(
    query_embed: torch.Tensor, candidate_embed: torch.Tensor
) -> torch.Tensor:
    return (query_embed[:, None, :] * candidate_embed[None, :, :]).sum(dim=-1)


def cosine_similarity_matrix(
    query_embed: torch.Tensor, candidate_embed: torch.Tensor
) -> torch.Tensor:
    return torch_fn.cosine_similarity(
        query_embed[:, None, :], candidate_embed[None, :, :], dim=-1
    )


def weighted_mean(
    values: torch.Tensor,
    sample_weights: torch.Tensor,
    *,
    dim: int | None = None,
    keepdim: bool = False,
) -> torch.Tensor:
    denominator = sample_weights.sum(dim=dim, keepdim=True) + 1e-9
    return (values * sample_weights / denominator).sum(dim=dim, keepdim=keepdim)


class EmbedLoss(torch.nn.Module, abc.ABC):
    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        query_embed: torch.Tensor,
        candidate_embed: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.check_embeds(query_embed, candidate_embed)
        logits = self.compute_logits(query_embed, candidate_embed)
        target = self.check_target(logits, target)
        negative_masks = self.mask_false_negatives(logits, target)
        negative_masks = self.mine_hard_negatives(logits, negative_masks)
        return self.loss(logits, target, negative_masks)

    def check_embeds(
        self, query_embed: torch.Tensor, candidate_embed: torch.Tensor
    ) -> None:
        n_dim = 3
        if query_embed.dim() != n_dim or candidate_embed.dim() != n_dim:
            msg = (
                f"inputs should have {n_dim} dimensions: "
                f"{query_embed.dim() = }, "
                f"{candidate_embed.dim() = }"
            )
            raise ValueError(msg)

        if (
            candidate_embed.size(0) != query_embed.size(0)
            and candidate_embed.size(0) != 1
        ):
            msg = (
                "query_embed and candidate_embed should have same number of rows: "
                f"{query_embed.size(0) = }, "
                f"{candidate_embed.size(0) = }"
            )
            raise ValueError(msg)

        if query_embed.size(1) != 1:
            msg = f"query_embed should have a single column: {query_embed.size(1) = }"
            raise ValueError(msg)

        embedding_dim = query_embed.size(-1)
        if candidate_embed.size(-1) != embedding_dim:
            msg = (
                "embedding_dim should match: "
                f"{ query_embed.size(-1) = }, "
                f"{ candidate_embed.size(-1) = }"
            )
            raise ValueError(msg)

    def compute_logits(
        self, query_embed: torch.Tensor, candidate_embed: torch.Tensor
    ) -> torch.Tensor:
        return (query_embed * candidate_embed).sum(dim=-1)
        # shape: (batch_size, num_candidates)

    def cosine_similarity_logits(
        self, query_embed: torch.Tensor, candidate_embed: torch.Tensor
    ) -> torch.Tensor:
        return torch_fn.cosine_similarity(query_embed, candidate_embed, dim=-1)
        # shape: (batch_size, num_candidates)

    def check_target(
        self, logits: torch.Tensor, target: torch.Tensor | None
    ) -> torch.Tensor:
        if target is None and self.config.target_position is None:
            msg = "either `targets` or `config.target_position` must be provided"
            raise ValueError(msg)

        if target is not None and self.config.target_position is not None:
            msg = "only one of `targets` or `config.target_position` should be provided"
            raise ValueError(msg)

        if target is None:
            match self.config.target_position:
                case "first":
                    target = torch.zeros(
                        logits.size(0), dtype=torch.long, device=logits.device
                    )
                case "diagonal":
                    target = torch.arange(
                        logits.size(0), dtype=torch.long, device=logits.device
                    )
                case _:
                    msg = f"invalid {self.config.target_position = }"
                    raise ValueError(msg)

        if target.dim() != 1:
            msg = f"targets should be a 1 dimensional: {target.dim() = }"
            raise ValueError(msg)

        if target.size(0) != logits.size(0):
            msg = (
                "targets and logits should have same number of rows: "
                f"{target.size(0) = }, "
                f"{logits.size(0) = }"
            )
            raise ValueError(msg)

        return target[:, None]

    def mask_false_negatives(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if not self.config.mask_hard_negatives:
            return torch.ones_like(logits, dtype=torch.bool).scatter(
                dim=1, index=target, value=False
            )
            # shape: (batch_size, num_candidates)

        target_logits = logits.gather(dim=1, index=target)
        # items with logits >= target logits are false negatives
        # this also masks the target logits
        return logits < target_logits
        # shape: (batch_size, num_candidates)

    def mine_hard_negatives(
        self, logits: torch.Tensor, negative_masks: torch.Tensor
    ) -> torch.Tensor:
        if self.config.num_negatives <= 0:
            return negative_masks

        if self.config.num_negatives >= logits.size(1):
            return negative_masks

        # take top-k logits from negatives only
        indices = (
            logits.where(negative_masks, -torch.inf)
            .topk(k=self.config.num_negatives, dim=-1, sorted=False)
            .indices
        )
        # shape: (batch_size, num_negatives)
        # use scatter to set selected indices to True
        # bool_and with negative masks to ensure true negatives only
        negative_masks &= torch.zeros_like(negative_masks).scatter(
            dim=-1, index=indices, value=True
        )
        # shape: (batch_size, num_candidates)
        return negative_masks

    @abc.abstractmethod
    def loss(
        self, logits: torch.Tensor, target: torch.Tensor, negative_masks: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def alignment_loss(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        target_logits = logits.gather(dim=1, index=target)
        return (1 - target_logits).sum()

    def contrastive_loss(
        self, logits: torch.Tensor, negative_masks: torch.Tensor
    ) -> torch.Tensor:
        losses = (logits - 1 + self.config.margin).relu()
        # shape: (batch_size, num_candidates)
        return weighted_mean(losses, negative_masks, dim=-1).sum()


class LogitsStatistics(EmbedLoss):
    def loss(
        self, logits: torch.Tensor, target: torch.Tensor, negative_masks: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        # num_negatives should exclude the diagonal positives
        num_negatives = negative_masks.size(1) - 1
        if self.config.num_negatives > 0:
            num_negatives = min(num_negatives, self.config.num_negatives)

        neg_density = (negative_masks.sum(dim=-1) / (num_negatives + 1e-9)).mean()
        stats = {"logits/neg/density": neg_density.item()}

        for key, value in {
            "pos": logits.gather(dim=1, index=target),
            "neg": logits[negative_masks],
        }.items():
            if value.numel() > 0:
                stats |= {
                    f"logits/{key}/mean": value.mean().item(),
                    f"logits/{key}/std": value.std().item(),
                    f"logits/{key}/min": value.min().item(),
                    f"logits/{key}/max": value.max().item(),
                }
        return stats


class AlignmentLoss(EmbedLoss):
    def compute_logits(
        self, query_embed: torch.Tensor, candidate_embed: torch.Tensor
    ) -> torch.Tensor:
        return self.cosine_similarity_logits(query_embed, candidate_embed)
        # shape: (batch_size, num_candidates)

    def loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        negative_masks: torch.Tensor,  # noqa: ARG002
    ) -> torch.Tensor:
        return self.alignment_loss(logits, target)


class AlignmentContrastiveLoss(EmbedLoss):
    def compute_logits(
        self, query_embed: torch.Tensor, candidate_embed: torch.Tensor
    ) -> torch.Tensor:
        return self.cosine_similarity_logits(query_embed, candidate_embed)
        # shape: (batch_size, num_candidates)

    def loss(
        self, logits: torch.Tensor, target: torch.Tensor, negative_masks: torch.Tensor
    ) -> torch.Tensor:
        return self.alignment_loss(logits, target) + self.contrastive_loss(
            logits, negative_masks
        )


class ContrastiveLoss(EmbedLoss):
    def compute_logits(
        self, query_embed: torch.Tensor, candidate_embed: torch.Tensor
    ) -> torch.Tensor:
        return self.cosine_similarity_logits(query_embed, candidate_embed)
        # shape: (batch_size, num_candidates)

    def loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,  # noqa: ARG002
        negative_masks: torch.Tensor,
    ) -> torch.Tensor:
        return self.contrastive_loss(logits, negative_masks)


class InfoNCELoss(EmbedLoss):
    def loss(
        self, logits: torch.Tensor, target: torch.Tensor, negative_masks: torch.Tensor
    ) -> torch.Tensor:
        # include target logits for cross entropy
        logit_masks = negative_masks.scatter(dim=-1, index=target, value=True)
        # shape: (batch_size, num_candidates)
        # set false negative logits to -inf
        logits = logits.where(logit_masks, -torch.inf) * self.config.scale
        # shape: (batch_size, num_candidates)
        return torch_fn.cross_entropy(logits, target[:, 0], reduction="sum")


class NCELoss(EmbedLoss):
    def loss(
        self, logits: torch.Tensor, target: torch.Tensor, negative_masks: torch.Tensor
    ) -> torch.Tensor:
        # positive logits are in the diagonal
        binary_targets = torch.zeros_like(logits).scatter(
            dim=1, index=target, value=1.0
        )
        # shape: (batch_size, num_candidates)
        nce_losses = torch_fn.binary_cross_entropy_with_logits(
            logits, binary_targets, reduction="none"
        )
        # shape: (batch_size, num_candidates)
        pos_loss = nce_losses.diagonal()
        # shape: (batch_size,)
        return (pos_loss + weighted_mean(nce_losses, negative_masks, dim=-1)).sum()


class PairwiseHingeLoss(EmbedLoss):
    def loss(
        self, logits: torch.Tensor, target: torch.Tensor, negative_masks: torch.Tensor
    ) -> torch.Tensor:
        target_logits = logits.gather(dim=1, index=target)
        # shape: (batch_size, 1)
        scores = logits - target_logits * (1 - self.config.margin)
        # shape: (batch_size, num_candidates)
        return weighted_mean(scores.relu(), negative_masks, dim=-1).sum()


class PairwiseLogisticLoss(EmbedLoss):
    def loss(
        self, logits: torch.Tensor, target: torch.Tensor, negative_masks: torch.Tensor
    ) -> torch.Tensor:
        target_logits = logits.gather(dim=1, index=target)
        # shape: (batch_size, 1)
        scores = logits - target_logits * (1 - self.config.margin)
        # shape: (batch_size, num_candidates)
        return weighted_mean(torch_fn.softplus(scores), negative_masks, dim=-1).sum()


LOSS_CLASSES: list[type[EmbedLoss]] = [
    AlignmentLoss,
    AlignmentContrastiveLoss,
    ContrastiveLoss,
    InfoNCELoss,
    NCELoss,
    PairwiseHingeLoss,
    PairwiseLogisticLoss,
]

LossType = Literal[
    AlignmentLoss.__name__,
    AlignmentContrastiveLoss.__name__,
    ContrastiveLoss.__name__,
    InfoNCELoss.__name__,
    NCELoss.__name__,
    PairwiseHingeLoss.__name__,
    PairwiseLogisticLoss.__name__,
]
