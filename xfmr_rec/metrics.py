import torch
import torchmetrics.functional.retrieval as tm_retrieval

"""Evaluation helpers for retrieval-style recommendation metrics.

This module provides lightweight wrappers around torchmetrics' retrieval
functions to compute common metrics (AUC, precision@k, recall@k, NDCG,
MRR, etc.) for a ranked list of recommended item IDs vs. ground-truth
target IDs.
"""


def compute_retrieval_metrics(
    rec_ids: list[str], target_ids: list[str], top_k: int
) -> dict[str, torch.Tensor]:
    """Compute several retrieval metrics for a single ranked recommendation list.

    Args:
        rec_ids: Ranked list of recommended item IDs (most to least relevant).
        target_ids: Ground-truth set/list of positive item IDs for the query.
        top_k: Evaluate metrics at this cutoff (will be clipped to len(rec_ids)).

    Returns:
        A mapping from metric name to a 0-dim torch.Tensor result. If
        `rec_ids` is empty an empty dict is returned.

    Notes:
        This function builds a combined list of candidate IDs by appending any
        missing ground-truth IDs after the ranked recommendations so that
        metrics that consider the whole candidate set can be computed.
    """
    if len(rec_ids) == 0:
        return {}

    top_k = min(top_k, len(rec_ids))
    target_set = set(target_ids)
    # rec_ids first, followed by any missing target_ids appended at the end
    all_items = rec_ids + [tid for tid in target_ids if tid not in set(rec_ids)]
    preds = torch.linspace(1, 0, len(all_items))
    target = torch.as_tensor([item in target_set for item in all_items])

    return {
        metric_fn.__name__: metric_fn(preds=preds, target=target, top_k=top_k)
        for metric_fn in [
            tm_retrieval.retrieval_auroc,
            tm_retrieval.retrieval_average_precision,
            tm_retrieval.retrieval_hit_rate,
            tm_retrieval.retrieval_normalized_dcg,
            tm_retrieval.retrieval_precision,
            tm_retrieval.retrieval_recall,
            tm_retrieval.retrieval_reciprocal_rank,
        ]
    }
