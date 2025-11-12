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
    rec_ids: list[str], target_ids: set[str] | list[str], top_k: int
) -> dict[str, torch.Tensor]:
    """Compute several retrieval metrics for one ranked recommendation list.

    This function evaluates a single ranked recommendation list against a
    set (or list) of ground-truth positive item IDs. It synthesizes
    predicted scores that follow the provided ranking in ``rec_ids``
    (higher scores for earlier items) and then delegates to the
    corresponding functions in ``torchmetrics.functional.retrieval``.

    Key behaviour
    - If either ``rec_ids`` or ``target_ids`` is empty, an empty dict
        is returned because there is nothing meaningful to evaluate.
    - ``top_k`` is clipped to ``len(rec_ids)`` so callers may pass a
        larger cutoff safely.
    - Some retrieval metrics expect a complete candidate set. To support
        those, the function constructs ``all_items`` by placing ``rec_ids``
        first and appending any ground-truth items not already present.

    Args:
        rec_ids (list[str]): Ranked list of recommended item IDs (most -> least
            relevant). Items must be comparable to values in ``target_ids``.
        target_ids (set[str] | list[str]): Iterable of ground-truth positive
            item IDs for the query. Can be a list or set; duplicates are
            ignored.
        top_k (int): Cutoff at which to evaluate rank-sensitive metrics. If
            ``top_k`` is greater than ``len(rec_ids)`` it will be reduced
            to ``len(rec_ids)``.

    Returns:
        dict[str, torch.Tensor]: Mapping from metric function name (the
        function ``__name__``) to a 0-dim :class:`torch.Tensor` containing
        the metric value. Returns an empty dict when inputs are empty.

    Raises:
        TypeError: If inputs are not indexable sequences of comparable IDs
            (membership checks and sequence operations are used).

    Example:
        >>> rec_ids = ["i10", "i3", "i7"]
        >>> target_ids = {"i3"}
        >>> compute_retrieval_metrics(rec_ids, target_ids, top_k=3)
        {"retrieval_auroc": tensor(1.), ...}
    """
    if len(target_ids) == 0:
        return {}

    if len(rec_ids) < top_k:
        # pad rec_ids with empty string if fewer than top_k items
        rec_ids += [""] * (top_k - len(rec_ids))

    target_ids = set(target_ids)
    # rec_ids first, followed by any missing target_ids appended at the end
    all_items = rec_ids + list(target_ids - set(rec_ids))
    preds = torch.linspace(1, 0, len(all_items))
    target = torch.as_tensor([item in target_ids for item in all_items])

    return {
        metric_fn.__name__: metric_fn(preds=preds, target=target, top_k=top_k)
        for metric_fn in METRIC_FNS
    }
