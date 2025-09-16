import torch


def compute_retrieval_metrics(
    rec_ids: list[str], target_ids: list[str], top_k: int
) -> dict[str, torch.Tensor]:
    import torchmetrics.functional.retrieval as tm_retrieval

    if len(rec_ids) == 0:
        return {}

    top_k = min(top_k, len(rec_ids))
    target_ids = set(target_ids)
    # rec_ids first, followed by target_ids at the end
    all_items = rec_ids + list(target_ids - set(rec_ids))
    preds = torch.linspace(1, 0, len(all_items))
    target = torch.as_tensor([item in target_ids for item in all_items])

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
