from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F  # noqa: N812
import torch.utils.data as torch_data
import torch.utils.data._utils.collate as torch_collate

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    import pyarrow.dataset as ds


def select_fields(example: dict[str, Any], *, fields: list[str]) -> dict[str, Any]:
    return {key: example[key] for key in fields}


def nest_example(example: dict[str, Any], key: str) -> dict[str, dict[str, Any]]:
    return {key: example}


def merge_examples(examples: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for example in examples:
        merged.update(example)
    return merged


def embed_example(example: dict[str, Any], *, model: torch.nn.Module) -> dict[str, Any]:
    return {
        **example,
        "embedding": model([example["text"]]).squeeze(0).numpy(force=True),
    }


def pad_tensors(
    batch: Iterable[torch.Tensor],
    dim: int = -1,
    *,
    pad_start: bool = False,
    pad_value: int = 0,
) -> torch.Tensor:
    elem = next(iter(batch))
    pad_size = max(example.size(dim) for example in iter(batch))
    pad = [0] * (elem.dim() * 2)
    pad_dim = 2 * dim + pad_start

    def pad_tensor(tensor: torch.Tensor) -> torch.Tensor:
        # pad tuple dimensions is reversed
        pad[-pad_dim - 1] = pad_size - tensor.size(dim)
        return F.pad(tensor, pad, value=pad_value)

    return torch.stack([pad_tensor(example) for example in batch])


def collate_tensor_fn(
    batch: Iterable[torch.Tensor], *, collate_fn_map: dict | None = None
) -> torch.Tensor:
    it = iter(batch)
    elem_size = next(it).size()
    if any(elem.size() != elem_size for elem in it):
        it = iter(batch)
        if all(elem.size()[:-1] == elem_size[:-1] for elem in it):
            # only last dimension different
            return pad_tensors(batch, dim=-1)
        if all(elem.size()[1:] == elem_size[1:] for elem in it):
            # only first dimension different
            return pad_tensors(batch, dim=0)

    return torch_collate.collate_tensor_fn(batch, collate_fn_map=collate_fn_map)


torch_collate.default_collate_fn_map[torch.Tensor] = collate_tensor_fn


@torch_data.functional_datapipe("load_parquet_as_dict")
class ParquetDictLoaderIterDataPipe(torch_data.IterDataPipe[dict[str, Any]]):
    def __init__(
        self,
        source_datapipe: torch_data.IterDataPipe[str],
        *,
        columns: Iterable[str] | None = None,
        filter_expr: ds.Expression | None = None,
        batch_size: int = 2**10,
    ) -> None:
        super().__init__()
        self.source_datapipe = source_datapipe
        self.columns = columns
        self.filter_expr = filter_expr
        self.batch_size = batch_size

    def pyarrow_dataset(self, source: str) -> ds.Dataset:
        import pyarrow.dataset as ds

        return ds.dataset(source)

    def __len__(self) -> int:
        return sum(
            self.pyarrow_dataset(source).count_rows(filter=self.filter_expr)
            for source in self.source_datapipe
        )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for source in self.source_datapipe:
            dataset = self.pyarrow_dataset(source)
            for batch in dataset.to_batches(
                columns=self.columns,
                filter=self.filter_expr,
                batch_size=self.batch_size,
            ):
                yield from batch.to_pylist()


@torch_data.functional_datapipe("cycle")
class CyclerIterDataPipe(torch_data.IterDataPipe[dict[str, Any]]):
    def __init__(
        self,
        source_datapipe: torch_data.IterDataPipe[dict[str, Any]],
        count: int = 1,
    ) -> None:
        super().__init__()
        self.source_datapipe = source_datapipe
        self.count = count
        if count < 0:
            msg = f"requires count >= 0, but {count = }"
            raise ValueError(msg)

    def __len__(self) -> int:
        if self.count <= 0:
            # use arbitrary large number so that valid length is shown for zip
            return 2**31 - 1  # max 32-bit signed integer
        return len(self.source_datapipe) * self.count

    def __iter__(self) -> Iterator[dict[str, Any]]:
        i = 0
        # if count <= 0, cycle indefinitely
        while self.count <= 0 or i < self.count:
            yield from self.source_datapipe
            i += 1
