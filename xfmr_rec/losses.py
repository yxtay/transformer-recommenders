from __future__ import annotations

import abc
from typing import Literal

import pydantic
import torch
import torch.nn.functional as torch_fn

"""Loss functions and utilities for embedding-based recommendation models.

This module implements a set of configurable loss classes and helper
functions for computing pairwise and contrastive losses over batched
query/candidate embeddings.
"""


class LossConfig(pydantic.BaseModel):
    """Configuration for embedding losses.

    Attributes:
        target_position: If "first", the positive example is at index 0. If
            "diagonal", the positive example is aligned on the diagonal.
        mask_false_negatives: Whether to mask examples that appear to be
            false negatives (have score >= positive score).
        num_hard_negatives: Number of hard negatives to mine per example.
        scale: Temperature / scaling factor applied for some losses.
        margin: Margin used for margin-based losses.
    """

    target_position: Literal["first", "diagonal"] | None = "first"
    mask_false_negatives: bool = True
    num_hard_negatives: int = 0
    scale: float = 1.0
    margin: float = 0.5


def squared_distance_matrix(
    query_embed: torch.Tensor, candidate_embed: torch.Tensor
) -> torch.Tensor:
    """Return the pairwise squared Euclidean distance matrix.

    The returned tensor has shape (batch_size, num_candidates) and contains
    0.5 * ||q - c||^2 for each pair.
    """
    return torch.cdist(query_embed, candidate_embed) ** 2 / 2


def dot_product_matrix(
    query_embed: torch.Tensor, candidate_embed: torch.Tensor
) -> torch.Tensor:
    """Compute the pairwise dot-product matrix between queries and candidates.

    Returns a (batch_size, num_candidates) tensor of q . c values.
    """
    return query_embed.mm(candidate_embed.mT)


def cosine_similarity_matrix(
    query_embed: torch.Tensor, candidate_embed: torch.Tensor
) -> torch.Tensor:
    """Compute cosine similarity for every query-candidate pair.

    Uses broadcasting to return a (batch_size, num_candidates) tensor.
    """
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
    """Compute a weighted mean along `dim` using `sample_weights`.

    A small epsilon is added to the denominator for numerical stability.
    """
    denominator = sample_weights.sum(dim=dim, keepdim=True) + 1e-9
    return (values * sample_weights / denominator).sum(dim=dim, keepdim=keepdim)


class EmbedLoss(torch.nn.Module, abc.ABC):
    """Base class for embedding-based losses.

    Subclasses should implement the `loss` method which receives precomputed
    `logits`, `target` indices (shape (batch_size, 1)), and boolean
    `negative_masks` indicating valid negatives. The `forward` method wraps
    common preprocessing: shape checks, logits computation, target handling,
    false-negative masking and optional hard-negative mining.
    """

    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        query_embed: torch.Tensor,
        candidate_embed: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, float]:
        """Compute the loss for a batch of query and candidate embeddings.

        Args:
            query_embed: Tensor of shape (batch_size, embedding_dim).
            candidate_embed: Tensor of shape (batch_size, num_candidates,
                embedding_dim).
            target: Optional 1-D tensor of positive indices per row. If not
                provided, `config.target_position` is used to infer targets.

        Returns:
            A scalar tensor containing the summed loss over the batch.
        """
        self.check_embeds(query_embed, candidate_embed)
        logits = self.compute_logits(query_embed, candidate_embed)
        target = self.check_target(logits, target)
        negative_masks = self.mask_false_negatives(logits, target)
        negative_masks = self.mine_hard_negatives(logits, negative_masks)
        return self.loss(logits, target, negative_masks)

    def check_embeds(
        self, query_embed: torch.Tensor, candidate_embed: torch.Tensor
    ) -> None:
        """Validate shapes of query and candidate embedding tensors.

        Raises a ValueError for mismatched shapes or embedding dimensions.
        """
        if query_embed.dim() != 2:
            msg = f"query_embed should have 2 dimensions: {query_embed.dim() = }"
            raise ValueError(msg)

        if candidate_embed.dim() != 3:
            msg = (
                f"candidate_embed should have 3 dimensions: {candidate_embed.dim() = }"
            )
            raise ValueError(msg)

        if query_embed.size(0) != candidate_embed.size(0):
            msg = (
                "batch_size should match: "
                f"{query_embed.size(0) = }, "
                f"{candidate_embed.size(0) = }"
            )
            raise ValueError(msg)

        if query_embed.size(-1) != candidate_embed.size(-1):
            msg = (
                "embedding_dim should match: "
                f"{query_embed.size(-1) = }, "
                f"{candidate_embed.size(-1) = }"
            )
            raise ValueError(msg)

    def compute_logits(
        self, query_embed: torch.Tensor, candidate_embed: torch.Tensor
    ) -> torch.Tensor:
        """Compute raw dot-product logits between queries and candidates.

        This method performs a batched matrix-multiply to produce a
        (batch_size, num_candidates) tensor of logits where each entry is
        the dot-product between a query embedding and a candidate embedding.

        Implementation note: we add a length-1 middle dimension to
        ``query_embed`` so that a batched ``bmm`` with a transposed
        ``candidate_embed`` yields shape (batch_size, 1, num_candidates),
        which we then squeeze to (batch_size, num_candidates).
        """
        # query_embed: (batch_size, embedding_dim) -> (batch_size, 1, embedding_dim)
        # candidate_embed: (batch_size, num_candidates, embedding_dim) -> transpose last two dims
        # batched bmm produces (batch_size, 1, num_candidates) -> squeeze to (batch_size, num_candidates)
        return query_embed[:, None, :].bmm(candidate_embed.mT)[:, 0, :]

    def cosine_similarity_logits(
        self, query_embed: torch.Tensor, candidate_embed: torch.Tensor
    ) -> torch.Tensor:
        """Compute cosine-similarity logits between queries and candidates.

        Returns a (batch_size, num_candidates) tensor.
        """
        return torch_fn.cosine_similarity(
            query_embed[:, None, :], candidate_embed, dim=-1
        )
        # shape: (batch_size, num_candidates)

    def check_target(
        self, logits: torch.Tensor, target: torch.Tensor | None
    ) -> torch.Tensor:
        """Validate or construct `target` indices for the batch.

        If `target` is None, builds targets according to
        `config.target_position` ("first" or "diagonal"). Returns a
        column vector of shape (batch_size, 1).
        """
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
            msg = f"targets should have 1 dimension: {target.dim() = }"
            raise ValueError(msg)

        if target.size(0) != logits.size(0):
            msg = f"batch_size should match: {target.size(0) = }, {logits.size(0) = }"
            raise ValueError(msg)

        return target[:, None]

    def mask_false_negatives(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Return a boolean mask of negatives considered valid.

        When `mask_false_negatives` is True, candidates with score >= the
        target score are masked out as potential false negatives.
        """
        if not self.config.mask_false_negatives:
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
        """Optional mining of a fixed number of hard negatives per example.

        This keeps only the top-k negatives (by logit) per row as valid
        negatives when `num_hard_negatives` > 0.
        """
        if self.config.num_hard_negatives <= 0:
            return negative_masks

        if self.config.num_hard_negatives >= logits.size(1):
            return negative_masks

        # take top-k logits from negatives only
        indices = (
            logits.where(negative_masks, -torch.inf)
            .topk(k=self.config.num_hard_negatives, dim=1, sorted=False)
            .indices
        )
        # shape: (batch_size, num_negatives)
        # use scatter to set selected indices to True
        # bool_and with negative masks to ensure true negatives only
        negative_masks &= torch.zeros_like(negative_masks).scatter(
            dim=1, index=indices, value=True
        )
        # shape: (batch_size, num_candidates)
        return negative_masks

    @abc.abstractmethod
    def loss(
        self, logits: torch.Tensor, target: torch.Tensor, negative_masks: torch.Tensor
    ) -> torch.Tensor | dict[str, float]:
        raise NotImplementedError

    def alignment_loss(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Simple alignment loss pushing positive logits toward 1.

        Returns sum(1 - target_logit) over the batch.
        """
        target_logits = logits.gather(dim=1, index=target)
        return (1 - target_logits).sum()

    def contrastive_loss(
        self, logits: torch.Tensor, negative_masks: torch.Tensor
    ) -> torch.Tensor:
        """Margin-based contrastive loss over negatives.

        Computes ReLU(logit - 1 + margin) for all entries, averages using
        `negative_masks`, then sums over the batch.
        """
        losses = (logits - 1 + self.config.margin).relu()
        # shape: (batch_size, num_candidates)
        return weighted_mean(losses, negative_masks, dim=1).sum()


class LogitsStatistics(EmbedLoss):
    """Collect simple statistics over logits for monitoring/logging.

    This class does not compute a training loss; instead it returns a
    dictionary of scalar statistics (means, std, min, max) for positives
    and negatives which can be logged during training.
    """

    def loss(
        self, logits: torch.Tensor, target: torch.Tensor, negative_masks: torch.Tensor
    ) -> dict[str, float]:
        # num_negatives should exclude 1 target per row
        num_negatives = negative_masks.size(1) - 1
        if self.config.num_hard_negatives > 0:
            num_negatives = min(num_negatives, self.config.num_hard_negatives)

        neg_density = (negative_masks.sum(dim=1) / (num_negatives + 1e-9)).mean()
        stats: dict[str, float] = {"logits/neg/density": neg_density.item()}

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
    """Alignment loss that encourages positive pairs to have high cosine.

    Uses cosine similarity logits and a simple L1-like alignment objective.
    """

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
    """Combine alignment and contrastive margin loss.

    Encourages positives to have high similarity while pushing negatives
    below a margin.
    """

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
    """Contrastive margin-based loss operating on cosine similarity logits.

    Uses a margin such that logits greater than (1 - margin) are penalised
    for negatives.
    """

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
    """InfoNCE-style loss using cross-entropy over positives and negatives.

    Scales logits by `config.scale` and computes cross-entropy with the
    positive index as the target.
    """

    def loss(
        self, logits: torch.Tensor, target: torch.Tensor, negative_masks: torch.Tensor
    ) -> torch.Tensor:
        # include target logits for cross entropy
        logit_masks = negative_masks.scatter(dim=1, index=target, value=True)
        # shape: (batch_size, num_candidates)
        # set false negative logits to -inf
        logits = logits.where(logit_masks, -torch.inf) * self.config.scale
        # shape: (batch_size, num_candidates)
        return torch_fn.cross_entropy(logits, target[:, 0], reduction="sum")


class NCELoss(EmbedLoss):
    """Binary NCE loss using sigmoid/binary cross-entropy on pairwise logits.

    Treats the positive as label=1 and negatives as label=0 and averages
    binary cross-entropy losses.
    """

    def loss(
        self, logits: torch.Tensor, target: torch.Tensor, negative_masks: torch.Tensor
    ) -> torch.Tensor:
        binary_targets = torch.zeros_like(logits).scatter(
            dim=1, index=target, value=1.0
        )
        # shape: (batch_size, num_candidates)
        nce_losses = torch_fn.binary_cross_entropy_with_logits(
            logits, binary_targets, reduction="none"
        )
        # shape: (batch_size, num_candidates)
        pos_loss = nce_losses.gather(dim=1, index=target)[:, 0]
        # shape: (batch_size,)
        return (pos_loss + weighted_mean(nce_losses, negative_masks, dim=1)).sum()


class PairwiseHingeLoss(EmbedLoss):
    """Pairwise hinge loss computed between positive and negative logits.

    Uses a margin-styled hinge implemented with ReLU.
    """

    def loss(
        self, logits: torch.Tensor, target: torch.Tensor, negative_masks: torch.Tensor
    ) -> torch.Tensor:
        target_logits = logits.gather(dim=1, index=target)
        # shape: (batch_size, 1)
        scores = logits - target_logits * (1 - self.config.margin)
        # shape: (batch_size, num_candidates)
        return weighted_mean(scores.relu(), negative_masks, dim=1).sum()


class PairwiseLogisticLoss(EmbedLoss):
    """Pairwise logistic loss using softplus on pairwise score differences.

    Smooth alternative to hinge, implemented with softplus.
    """

    def loss(
        self, logits: torch.Tensor, target: torch.Tensor, negative_masks: torch.Tensor
    ) -> torch.Tensor:
        target_logits = logits.gather(dim=1, index=target)
        # shape: (batch_size, 1)
        scores = logits - target_logits * (1 - self.config.margin)
        # shape: (batch_size, num_candidates)
        return weighted_mean(torch_fn.softplus(scores), negative_masks, dim=1).sum()


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
    "AlignmentLoss",
    "AlignmentContrastiveLoss",
    "ContrastiveLoss",
    "InfoNCELoss",
    "NCELoss",
    "PairwiseHingeLoss",
    "PairwiseLogisticLoss",
]
