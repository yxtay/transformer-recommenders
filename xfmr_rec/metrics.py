import torch
import torchmetrics.functional.retrieval as tm_retrieval


def compute_retrieval_metrics(
    rec_ids: list[str], target_ids: list[str], top_k: int
) -> dict[str, torch.Tensor]:
    """Compute retrieval metrics for a single ranked recommendation list.

    Args:
        rec_ids (list[str]): Ranked list of recommended item IDs (most to
            least relevant).
        target_ids (list[str]): Ground-truth set or list of positive item IDs
            for the query.
        top_k (int): Evaluate metrics at this cutoff. The value will be
            clipped to ``len(rec_ids)`` if necessary.

    Returns:
        dict[str, torch.Tensor]: Mapping from metric function name to a
        0-dim :class:`torch.Tensor` containing the metric value. If
        ``rec_ids`` is empty an empty dictionary is returned.

    Notes:
        The function creates a combined candidate list by taking ``rec_ids``
        first and then appending any missing ``target_ids``. This ensures
        metrics that expect a complete candidate set can be computed.
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
