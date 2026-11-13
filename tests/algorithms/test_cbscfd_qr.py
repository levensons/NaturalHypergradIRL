"""QR compression must preserve CBSCFD's represented system, including solves."""

import pytest
import torch

from src.algorithms.approximations import CBSCFD


def assert_close_norm(actual, expected, tolerance):
    """Compare Frobenius/vector error without amplifying near-zero entries."""
    error = torch.linalg.vector_norm((actual - expected).double())
    scale = torch.linalg.vector_norm(expected.double()).clamp(min=1.0)
    assert (error / scale).item() < tolerance


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("m,dim", [(1, 2), (1, 9), (3, 6), (4, 17)])
@pytest.mark.parametrize("pattern", ["random", "zero", "rank_one", "repeated"])
@pytest.mark.parametrize("batching", ["rows", "cross_boundary", "all"])
def test_qr_preserves_states_and_inverse_solve(dtype, m, dim, pattern, batching):
    generator = torch.Generator().manual_seed(761)
    n = 7 * m + max(1, m - 1)
    rows = 0.3 * torch.randn(n, dim, dtype=dtype, generator=generator)
    if pattern == "zero":
        rows.zero_()
    elif pattern == "rank_one":
        rows = rows[:, :1] @ torch.randn(1, dim, dtype=dtype, generator=generator)
    elif pattern == "repeated":
        rows = rows[:1].repeat(n, 1)
    rows.requires_grad_()

    batch_size = {"rows": 1, "cross_boundary": m + 1, "all": n}[batching]
    baseline = CBSCFD(dim, m, 0.7, dtype=dtype)
    qr = CBSCFD(dim, m, 0.7, dtype=dtype, variant="qr_only")
    tolerance = 3e-5 if dtype == torch.float32 else 3e-12

    # Check the initial sketch, then partial buffers and repeated compressions.
    for start in [None, *range(0, n, batch_size)]:
        if start is not None:
            chunk = rows[start:start + batch_size]
            if batching == "rows":
                chunk = chunk[0]  # Also cover the single-row public API.
            baseline.extend(chunk)
            qr.extend(chunk)

        assert qr.ptr == baseline.ptr
        for counter in ("svd_count", "rows_processed", "iteration_count", "extend_count"):
            assert getattr(qr, counter) == getattr(baseline, counter)
        assert_close_norm(qr.alpha, baseline.alpha, tolerance)
        assert_close_norm(qr.H, baseline.H, tolerance)
        # SVD signs/bases need not agree; the represented Gram matrix must.
        assert_close_norm(qr.Z.T @ qr.Z, baseline.Z.T @ baseline.Z, tolerance)
        assert torch.count_nonzero(qr.Z[qr.ptr:]) == 0
        for value in (qr.Z, qr.H, qr.alpha):
            assert torch.isfinite(value).all()
            assert not value.requires_grad

        g = torch.randn(dim, dtype=dtype, generator=generator, requires_grad=True)
        active = qr.get_Z()
        system = qr.alpha * torch.eye(dim, dtype=dtype) + active.T @ active
        direct = torch.linalg.solve(system, g.detach())
        for cholesky in (True, False):
            result = qr.solve(g, cholesky=cholesky)
            assert not result.requires_grad
            assert torch.isfinite(result).all()
            assert_close_norm(result, direct, tolerance)
            assert_close_norm(result, baseline.solve(g, cholesky=cholesky), tolerance)
            relative_residual = torch.linalg.vector_norm(system @ result - g.detach()) / g.detach().norm()
            assert relative_residual.item() < tolerance
            assert torch.count_nonzero(qr.solve(torch.zeros_like(g), cholesky=cholesky)) == 0

    assert qr.svd_count == n // m
    assert rows.grad is None
    profile = qr.profiling
    assert profile["other_time"] == profile["extend_time"] - profile["svd_time"]
    assert profile["total_time"] == profile["extend_time"] + profile["solve_time"]


def test_qr_compresses_using_small_svd(monkeypatch):
    """Catch accidentally bypassing QR while still matching the baseline."""
    shapes = []
    original = torch.linalg.svd

    def record_shape(matrix, **kwargs):
        shapes.append(tuple(matrix.shape))
        return original(matrix, **kwargs)

    monkeypatch.setattr(torch.linalg, "svd", record_shape)
    sketch = CBSCFD(17, 4, 0.7, variant="qr_only")
    sketch.extend(torch.eye(17)[:4])
    assert shapes == [(8, 8)]


@pytest.mark.parametrize("variant,expected_time", [("baseline", 6.0), ("qr_only", 10.0)])
def test_svd_timer_covers_the_complete_factorization(monkeypatch, variant, expected_time):
    """Fixed kernel costs catch omitting QR or counting its time twice."""
    clock = [0.0]
    original_qr, original_svd = torch.linalg.qr, torch.linalg.svd

    def qr(*args, **kwargs):
        result = original_qr(*args, **kwargs)
        clock[0] += 2.0
        return result

    def svd(*args, **kwargs):
        result = original_svd(*args, **kwargs)
        clock[0] += 3.0
        return result

    monkeypatch.setattr(torch.linalg, "qr", qr)
    monkeypatch.setattr(torch.linalg, "svd", svd)
    monkeypatch.setattr("src.algorithms.approximations.time.perf_counter", lambda: clock[0])
    sketch = CBSCFD(17, 4, .7, variant=variant)
    sketch.extend(torch.eye(17)[:8])  # Two compressions: QR=2 and SVD=3 each.
    stats = sketch.profiling
    assert stats["svd_count"] == 2
    assert stats["svd_time"] == expected_time
    assert stats["mean_svd_time"] == expected_time / 2
    assert stats["other_time"] == 0.0
    assert stats["timing_schema_version"] == 2


@pytest.mark.parametrize("variant", ["invalid", "solve_only", "qr_and_solve"])
def test_unsupported_variant_raises(variant):
    with pytest.raises(ValueError, match="Unknown CBSCFD variant"):
        CBSCFD(8, 2, 0.7, variant=variant)
