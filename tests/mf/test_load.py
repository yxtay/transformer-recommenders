import pytest
import torch


@pytest.mark.parametrize(
    ("batch_sizes", "dim", "pad_start", "expected_size"),
    [
        ([(1,), (3,)], 0, False, (2, 3)),
        ([(1,), (3,)], -1, False, (2, 3)),
        ([(3, 2), (5, 2)], 0, False, (2, 5, 2)),
        ([(2, 3), (2, 5)], 1, False, (2, 2, 5)),
        ([(2, 3), (2, 5)], -1, False, (2, 2, 5)),
        ([(3, 2), (5, 2)], -2, False, (2, 5, 2)),
        ([(1,), (3,)], 0, True, (2, 3)),
        ([(1,), (3,)], -1, True, (2, 3)),
        ([(3, 2), (5, 2)], 0, True, (2, 5, 2)),
        ([(2, 3), (2, 5)], 1, True, (2, 2, 5)),
        ([(2, 3), (2, 5)], -1, True, (2, 2, 5)),
        ([(3, 2), (5, 2)], -2, True, (2, 5, 2)),
    ],
)
def test_pad_tensors(
    *, batch_sizes: tuple[int], dim: int, pad_start: bool, expected_size: tuple[int]
) -> None:
    from xfmr_rec.mf.load import pad_tensors

    batch = [torch.rand(size) for size in batch_sizes]  # devskim: ignore DS148264
    padded = pad_tensors(batch, dim=dim, pad_start=pad_start)
    assert padded.size() == expected_size, f"{padded.size() = } != {expected_size = }"
