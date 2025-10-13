from collections.abc import Callable

import torch
import torchmetrics.functional.retrieval as tm_retrieval

METRIC_FNS: list[Callable[..., torch.Tensor]] = [
    tm_retrieval.retrieval_normalized_dcg,
    tm_retrieval.retrieval_average_precision,
    tm_retrieval.retrieval_auroc,
    tm_retrieval.retrieval_precision,
    tm_retrieval.retrieval_recall,
    tm_retrieval.retrieval_hit_rate,
    tm_retrieval.retrieval_reciprocal_rank,
]


def compute_retrieval_metrics(
    rec_ids: list[str], target_ids: list[str], top_k: int
) -> dict[str, torch.Tensor]:
    """Compute common retrieval metrics for one ranked recommendation list.

    This helper computes several retrieval metrics (AUC, AP, Hit Rate,
    nDCG, Precision, Recall, Reciprocal Rank) for a single query by
    combining the model's ranked predictions with the ground-truth
    positives and delegating to the corresponding functions in
    ``torchmetrics.functional.retrieval``.

    Behavior notes
    - If either ``rec_ids`` or ``target_ids`` is empty the function
        returns an empty dict — there is nothing meaningful to evaluate.
    - ``top_k`` is clipped to the length of ``rec_ids`` so callers may
        pass a larger cutoff and still get safe behaviour.
    - To support metrics that expect a complete candidate set, the
        function constructs ``all_items`` by placing the ranked
        ``rec_ids`` first and then appending any ``target_ids`` that were
        not already present. The predicted scores are synthetic and only
        used to create a monotonically decreasing ranking (higher score
        = more relevant) consistent with ``rec_ids``.

    Args:
        rec_ids: Ranked list of recommended item IDs (most -> least
            relevant). Items must be comparable to the values in
            ``target_ids`` (usually strings or ints converted to str).
        target_ids: Iterable of ground-truth positive item IDs for the
            query. This can be a set or list; duplicates are ignored.
        top_k: Cutoff at which to evaluate rank-sensitive metrics. If
            ``top_k`` is greater than ``len(rec_ids)`` it will be
            reduced to ``len(rec_ids)``.

    Returns:
        A mapping from metric function name (the Python function
        ``__name__``) to a 0-dim :class:`torch.Tensor` containing the
        metric value. If the function cannot compute metrics because
        input lists are empty, an empty dict is returned.

    Raises:
        TypeError: If inputs are not indexable sequences of comparable
            IDs (this function performs membership checks and relies on
            sequence operations).

    Example:
        >>> rec_ids = ["i10", "i3", "i7"]
        >>> target_ids = ["i3"]
        >>> compute_retrieval_metrics(rec_ids, target_ids, top_k=3)
        {"retrieval_auroc": tensor(1.), ...}

    Implementation detail:
        Predictions are synthetic scores generated with ``torch.linspace``
        to reflect the input ranking (first item has the highest score).
        This keeps metric computations consistent with the provided
        ranking while avoiding a dependency on raw model scores.
    """
    if len(target_ids) == 0 or len(rec_ids) == 0:
        return {}

    top_k = min(top_k, len(rec_ids))
    target_set = set(target_ids)
    # rec_ids first, followed by any missing target_ids appended at the end
    all_items = rec_ids + [tid for tid in target_ids if tid not in set(rec_ids)]
    preds = torch.linspace(1, 0, len(all_items))
    target = torch.as_tensor([item in target_set for item in all_items])

    return {
        metric_fn.__name__: metric_fn(preds=preds, target=target, top_k=top_k)
        for metric_fn in METRIC_FNS
    }
