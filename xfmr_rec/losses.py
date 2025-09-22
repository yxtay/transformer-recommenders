from __future__ import annotations

import abc
from typing import Literal

import pydantic
import torch
import torch.nn.functional as torch_fn

LossType = Literal[
    "AlignmentLoss",
    "AlignmentContrastiveLoss",
    "ContrastiveLoss",
    "InfoNCELoss",
    "NCELoss",
    "PairwiseHingeLoss",
    "PairwiseLogisticLoss",
]


class LossConfig(pydantic.BaseModel):
    num_negatives: int = 0
    margin: float = 0.5


def squared_distance_matrix(
    query_embed: torch.Tensor, candidate_embed: torch.Tensor
) -> torch.Tensor:
    return torch.cdist(query_embed, candidate_embed) ** 2 / 2


def dot_product_matrix(
    anchor_embed: torch.Tensor, candidate_embed: torch.Tensor
) -> torch.Tensor:
    return (anchor_embed[:, None, :] * candidate_embed[None, :, :]).sum(dim=-1)


def cosine_similarity_matrix(
    anchor_embed: torch.Tensor, candidate_embed: torch.Tensor
) -> torch.Tensor:
    return torch_fn.cosine_similarity(
        anchor_embed[:, None, :], candidate_embed[None, :, :], dim=-1
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
    def __init__(self, *, config: LossConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        anchor_embed: torch.Tensor,
        pos_embed: torch.Tensor,
        neg_embed: torch.Tensor,
    ) -> torch.Tensor:
        self.check_inputs(anchor_embed, pos_embed, neg_embed)
        logits = self.compute_logits(anchor_embed, pos_embed, neg_embed)
        negative_masks = self.mask_false_negatives(logits)
        negative_masks = self.mine_hard_negatives(logits, negative_masks)
        return self.loss(logits, negative_masks)

    def check_inputs(
        self,
        anchor_embed: torch.Tensor,
        pos_embed: torch.Tensor,
        neg_embed: torch.Tensor,
    ) -> None:
        n_dim = 2
        if (
            anchor_embed.dim() != n_dim
            or pos_embed.dim() != n_dim
            or neg_embed.dim() != n_dim
        ):
            msg = (
                "inputs should have 2 dimensions: "
                f"{anchor_embed.dim() = }, "
                f"{pos_embed.dim() = }, "
                f"{neg_embed.dim() = }"
            )
            raise ValueError(msg)

        if anchor_embed.size(0) != pos_embed.size(0):
            msg = (
                "batch_size should match: "
                f"{anchor_embed.size(0) = }, "
                f"{pos_embed.size(0) = }"
            )
            raise ValueError(msg)

        embedding_dim = anchor_embed.size(1)
        if pos_embed.size(1) != embedding_dim or neg_embed.size(1) != embedding_dim:
            msg = (
                "embedding_dim should match: "
                f"{ anchor_embed.size(1) = }, "
                f"{ pos_embed.size(1) = }, "
                f"{ neg_embed.size(1) = }"
            )
            raise ValueError(msg)

    def compute_logits(
        self,
        anchor_embed: torch.Tensor,
        pos_embed: torch.Tensor,
        neg_embed: torch.Tensor,
    ) -> torch.Tensor:
        candidate_embed = torch.cat([pos_embed, neg_embed])
        # shape: (2 * batch_size, embedding_dim)
        return dot_product_matrix(anchor_embed, candidate_embed)
        # shape: (batch_size, 2 * batch_size)

    def cosine_similarity_logits(
        self,
        anchor_embed: torch.Tensor,
        pos_embed: torch.Tensor,
        neg_embed: torch.Tensor,
    ) -> torch.Tensor:
        candidate_embed = torch.cat([pos_embed, neg_embed])
        # shape: (2 * batch_size, embedding_dim)
        return cosine_similarity_matrix(anchor_embed, candidate_embed)
        # shape: (batch_size, 2 * batch_size)

    def mask_false_negatives(self, logits: torch.Tensor) -> torch.Tensor:
        # items with logits >= positive logits are false negatives
        # this also masks the diagonal positive logits
        return logits < logits.diagonal()[:, None]
        # shape: (batch_size, num_items)

    def mine_hard_negatives(
        self, logits: torch.Tensor, negative_masks: torch.Tensor
    ) -> torch.Tensor:
        if self.config.num_negatives <= 0:
            return negative_masks

        if self.config.num_negatives >= logits.size(1):
            return negative_masks

        # modifiy logits of false negatives to be -inf
        # take top-k from modified logits
        indices = (
            torch.where(negative_masks, logits, -torch.inf)
            .topk(k=self.config.num_negatives, dim=-1, sorted=False)
            .indices
        )
        # shape: (batch_size, num_negatives)
        # torch scatter gives boolean mask of selected negatives
        # bool and with negative masks to ensure true negatives only
        negative_masks &= torch.zeros_like(negative_masks).scatter_(
            dim=-1, index=indices, value=True
        )
        # shape: (batch_size, num_items)
        return negative_masks

    @abc.abstractmethod
    def loss(self, logits: torch.Tensor, negative_masks: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def alignment_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return (1 - logits.diagonal()).sum()

    def contrastive_loss(
        self, logits: torch.Tensor, negative_masks: torch.Tensor
    ) -> torch.Tensor:
        losses = (logits - 1 + self.config.margin).relu()
        # shape: (batch_size, num_items)
        return weighted_mean(losses, negative_masks, dim=-1).sum()


class NegativeDensity(EmbedLoss):
    def loss(self, logits: torch.Tensor, negative_masks: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        return negative_masks.float().mean(dim=-1).sum()


class AlignmentLoss(EmbedLoss):
    def compute_logits(
        self,
        anchor_embed: torch.Tensor,
        pos_embed: torch.Tensor,
        neg_embed: torch.Tensor,
    ) -> torch.Tensor:
        return self.cosine_similarity_logits(anchor_embed, pos_embed, neg_embed)
        # shape: (batch_size, 2 * batch_size)

    def loss(self, logits: torch.Tensor, negative_masks: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        return self.alignment_loss(logits)


class AlignmentContrastiveLoss(EmbedLoss):
    def compute_logits(
        self,
        anchor_embed: torch.Tensor,
        pos_embed: torch.Tensor,
        neg_embed: torch.Tensor,
    ) -> torch.Tensor:
        return self.cosine_similarity_logits(anchor_embed, pos_embed, neg_embed)
        # shape: (batch_size, 2 * batch_size)

    def loss(self, logits: torch.Tensor, negative_masks: torch.Tensor) -> torch.Tensor:
        return self.alignment_loss(logits) + self.contrastive_loss(
            logits, negative_masks
        )


class ContrastiveLoss(EmbedLoss):
    def compute_logits(
        self,
        anchor_embed: torch.Tensor,
        pos_embed: torch.Tensor,
        neg_embed: torch.Tensor,
    ) -> torch.Tensor:
        return self.cosine_similarity_logits(anchor_embed, pos_embed, neg_embed)
        # shape: (batch_size, 2 * batch_size)

    def loss(self, logits: torch.Tensor, negative_masks: torch.Tensor) -> torch.Tensor:
        return self.contrastive_loss(logits, negative_masks)


class InfoNCELoss(EmbedLoss):
    def loss(self, logits: torch.Tensor, negative_masks: torch.Tensor) -> torch.Tensor:
        # include diagonal positive logits for cross entropy
        negative_masks |= torch.eye(
            *logits.size(), dtype=torch.bool, device=negative_masks.device
        )
        # shape: (batch_size, num_items)
        # set false negative logits to -inf
        logits = torch.where(negative_masks, logits, -torch.inf)
        # shape: (batch_size, num_items)
        # targets are indices of diagonal positive logits
        targets = torch.arange(logits.size(0), dtype=torch.long, device=logits.device)
        # shape: (batch_size,)
        return torch_fn.cross_entropy(logits, targets, reduction="sum")


class NCELoss(EmbedLoss):
    def loss(self, logits: torch.Tensor, negative_masks: torch.Tensor) -> torch.Tensor:
        # positive logits are in the diagonal
        targets = torch.eye(*logits.size(), device=logits.device)
        # shape: (batch_size, num_items)
        nce_losses = torch_fn.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        # shape: (batch_size, num_items)
        pos_loss = nce_losses.diagonal()
        # shape: (batch_size,)
        return (pos_loss + weighted_mean(nce_losses, negative_masks, dim=-1)).sum()


class PairwiseHingeLoss(EmbedLoss):
    def loss(self, logits: torch.Tensor, negative_masks: torch.Tensor) -> torch.Tensor:
        scores = logits - logits.diagonal()[:, None] * (1 - self.config.margin)
        # shape: (batch_size, num_items)
        return weighted_mean(scores.relu(), negative_masks, dim=-1).sum()


class PairwiseLogisticLoss(EmbedLoss):
    def loss(self, logits: torch.Tensor, negative_masks: torch.Tensor) -> torch.Tensor:
        scores = logits - logits.diagonal()[:, None] * (1 - self.config.margin)
        # shape: (batch_size, num_items)
        return weighted_mean(torch_fn.softplus(scores), negative_masks, dim=-1).sum()
