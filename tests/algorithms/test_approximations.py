import torch

from src.algorithms.approximations import CBSCFD


def test_cbscfd_solve_shape_and_finiteness():
    torch.manual_seed(0)

    n = 64
    d = 8
    m = 4

    X = torch.randn(n, d)
    g = torch.randn(d)
    reg = 1e-2

    sketch = CBSCFD(d, m, reg, dtype=torch.float32)
    sketch.extend(X)

    solution = sketch.solve(g)

    assert solution.shape == g.shape
    assert torch.isfinite(solution).all()


def test_cbscfd_larger_sketch_improves_approximation():
    torch.manual_seed(0)

    n = 64
    d = 32
    reg = 1e-2

    X = torch.randn(n, d)
    g = torch.randn(d)

    fisher = X.T @ X + reg * torch.eye(d)
    exact_solution = torch.linalg.solve(fisher, g)

    errors = []
    for m in (4, 8, 16):
        sketch = CBSCFD(d, m, reg, dtype=torch.float32)
        sketch.extend(X)
        sketch_solution = sketch.solve(g)

        relative_error = (
            torch.linalg.vector_norm(sketch_solution - exact_solution)
            / torch.linalg.vector_norm(exact_solution)
        )
        errors.append(relative_error.item())

    assert errors[1] <= errors[0]
    assert errors[2] <= errors[1]
