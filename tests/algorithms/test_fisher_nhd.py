import pytest
import torch
import torch.nn as nn

from src.algorithms.fisher_nhd import FisherNHD


class DummyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)

    def log_prob(self, states, actions):
        logits = self.linear(states)
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs.gather(1, actions.long().reshape(-1, 1)).squeeze(1)


class DummyReward(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, states, actions):
        return self.linear(states).squeeze(-1)


@pytest.fixture
def nhd():
    torch.manual_seed(0)

    return FisherNHD(
        reward=DummyReward(),
        policy=DummyPolicy(),
        gamma=0.9,
        alpha=0.5,
        lr=1e-2,
        fisher_reg=1e-2,
        fisher_batch_size=3,
    )


@pytest.fixture
def trajectories():
    return [
        {
            "states": torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
            "actions": torch.tensor([0, 1, 0]),
        },
        {
            "states": torch.tensor([[0.5, 1.0], [1.0, -0.5]]),
            "actions": torch.tensor([1, 0]),
        },
    ]


@pytest.fixture
def gradient(nhd: FisherNHD):
    torch.manual_seed(1)
    return torch.randn(nhd.policy_num_params)


@pytest.fixture
def score_matrix(nhd: FisherNHD, trajectories):
    return nhd._weighted_score_matrix(trajectories, verbose=False)


# FISHER MATRIX


def test_explicit_fisher_is_symmetric(nhd: FisherNHD, trajectories):
    fisher = nhd.explicit_fisher(trajectories, verbose=False)

    assert fisher.shape == (nhd.policy_num_params, nhd.policy_num_params)
    assert torch.allclose(fisher, fisher.T, rtol=1e-6, atol=1e-7)


def test_explicit_fisher_is_positive_semidefinite(nhd: FisherNHD, trajectories):
    fisher = nhd.explicit_fisher(trajectories, verbose=False)
    eigenvalues = torch.linalg.eigvalsh(fisher)

    assert eigenvalues.min().item() >= -1e-6


def test_weighted_score_matrix_matches_explicit_fisher(nhd: FisherNHD, trajectories, score_matrix):
    expected = nhd.explicit_fisher(trajectories, verbose=False)
    result = score_matrix.T @ score_matrix

    assert score_matrix.shape == (5, nhd.policy_num_params)
    assert torch.allclose(result, expected, rtol=1e-5, atol=1e-6)


def test_fisher_vector_product_matches_explicit_fisher(nhd: FisherNHD, trajectories, gradient):
    fisher = nhd.explicit_fisher(trajectories, verbose=False)
    expected = fisher @ gradient + nhd.fisher_reg * gradient

    result = nhd._fisher_vector_product(trajectories, gradient, verbose=False)

    assert torch.allclose(result, expected, rtol=1e-5, atol=1e-6)


# EXPLICIT FISHER


def test_explicit_fisher_streaming_has_small_residual(nhd: FisherNHD, trajectories, gradient):
    fisher = nhd.explicit_fisher(trajectories, verbose=False)
    fisher.diagonal().add_(nhd.fisher_reg)

    result = nhd.fisher_solve_explicit_fisher_streaming(trajectories, gradient, verbose=False)

    residual = fisher @ result - gradient
    relative_residual = torch.linalg.vector_norm(residual) / torch.linalg.vector_norm(gradient)

    assert relative_residual.item() < 1e-5


def test_explicit_fisher_materialized_has_small_residual(nhd: FisherNHD, gradient, score_matrix):
    fisher = score_matrix.T @ score_matrix
    fisher.diagonal().add_(nhd.fisher_reg)

    result = nhd.fisher_solve_explicit_fisher_materialized(score_matrix, gradient)

    residual = fisher @ result - gradient
    relative_residual = torch.linalg.vector_norm(residual) / torch.linalg.vector_norm(gradient)

    assert relative_residual.item() < 1e-5


def test_explicit_fisher_materialized_matches_streaming(nhd: FisherNHD, trajectories, gradient, score_matrix):
    expected = nhd.fisher_solve_explicit_fisher_streaming(trajectories, gradient, verbose=False)
    result = nhd.fisher_solve_explicit_fisher_materialized(score_matrix, gradient)

    assert torch.allclose(result, expected, rtol=1e-5, atol=1e-6)


# CONJUGATE GRADIENTS


def test_cg_streaming_matches_explicit_solve(nhd: FisherNHD, trajectories, gradient):
    fisher = nhd.explicit_fisher(trajectories, verbose=False)
    fisher.diagonal().add_(nhd.fisher_reg)

    expected = torch.linalg.solve(fisher, gradient)
    result = nhd.fisher_solve_cg_streaming(trajectories, gradient, max_iters=100, tol=1e-8, verbose=False)

    assert torch.allclose(result, expected, rtol=1e-4, atol=1e-5)


def test_cg_streaming_has_small_residual(nhd: FisherNHD, trajectories, gradient):
    result = nhd.fisher_solve_cg_streaming(trajectories, gradient, max_iters=100, tol=1e-8, verbose=False)

    residual = nhd._fisher_vector_product(trajectories, result, verbose=False) - gradient
    relative_residual = torch.linalg.vector_norm(residual) / torch.linalg.vector_norm(gradient)

    assert relative_residual.item() < 1e-5


def test_cg_materialized_matches_explicit_solve(nhd: FisherNHD, gradient, score_matrix):
    expected = nhd.fisher_solve_explicit_fisher_materialized(score_matrix, gradient)
    result = nhd.fisher_solve_cg_materialized(score_matrix, gradient, max_iters=100, tol=1e-8, verbose=False)

    assert torch.allclose(result, expected, rtol=1e-4, atol=1e-5)


def test_cg_materialized_has_small_residual(nhd: FisherNHD, gradient, score_matrix):
    result = nhd.fisher_solve_cg_materialized(score_matrix, gradient, max_iters=100, tol=1e-8, verbose=False)

    residual = score_matrix.T @ (score_matrix @ result) + nhd.fisher_reg * result - gradient
    relative_residual = torch.linalg.vector_norm(residual) / torch.linalg.vector_norm(gradient)

    assert relative_residual.item() < 1e-5


def test_cg_materialized_matches_streaming(nhd: FisherNHD, trajectories, gradient, score_matrix):
    expected = nhd.fisher_solve_cg_streaming(trajectories, gradient, max_iters=100, tol=1e-8, verbose=False)
    result = nhd.fisher_solve_cg_materialized(score_matrix, gradient, max_iters=100, tol=1e-8, verbose=False)

    assert torch.allclose(result, expected, rtol=1e-4, atol=1e-5)


# SKETCH


@pytest.mark.parametrize("sketch_size", [1, 2])
def test_sketch_materialized_matches_streaming(nhd: FisherNHD, trajectories, gradient, score_matrix, sketch_size):
    expected = nhd.fisher_solve_sketch_streaming(trajectories, gradient, sketch_size=sketch_size, verbose=False)
    result = nhd.fisher_solve_sketch_materialized(score_matrix, gradient, sketch_size=sketch_size)

    assert torch.allclose(result, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("sketch_size", [1, 2])
def test_sketch_streaming_returns_finite_solution(nhd: FisherNHD, trajectories, gradient, sketch_size):
    result = nhd.fisher_solve_sketch_streaming(trajectories, gradient, sketch_size=sketch_size, verbose=False)

    assert result.shape == gradient.shape
    assert torch.isfinite(result).all()


@pytest.mark.parametrize("sketch_size", [1, 2])
def test_sketch_materialized_returns_finite_solution(nhd: FisherNHD, gradient, score_matrix, sketch_size):
    result = nhd.fisher_solve_sketch_materialized(score_matrix, gradient, sketch_size=sketch_size)

    assert result.shape == gradient.shape
    assert torch.isfinite(result).all()


@pytest.mark.parametrize("implementation", ["streaming", "materialized"])
@pytest.mark.parametrize("sketch_size", [1, 2])
def test_sketch_profiling(nhd: FisherNHD, trajectories, gradient, score_matrix, implementation, sketch_size):
    if implementation == "streaming":
        nhd.fisher_solve_sketch_streaming(trajectories, gradient, sketch_size=sketch_size, verbose=False)
    else:
        nhd.fisher_solve_sketch_materialized(score_matrix, gradient, sketch_size=sketch_size)

    assert nhd.sketch_rows_processed == score_matrix.shape[0]
    assert nhd.sketch_svd_count >= 0
    assert nhd.sketch_extend_time >= 0
    assert nhd.sketch_svd_time >= 0
    assert nhd.sketch_solve_time >= 0
    assert nhd.sketch_total_time >= 0


# SCORE GRADIENTS


def test_grad_log_pi_shape_and_finiteness(nhd: FisherNHD, trajectories):
    states = trajectories[0]["states"]
    actions = trajectories[0]["actions"]

    grads = nhd._grad_log_pi_a_s(states, actions)

    assert grads.shape == (states.shape[0], nhd.policy_num_params)
    assert torch.isfinite(grads).all()


# VALIDATION


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="Unknown Fisher solve mode"):
        FisherNHD(
            reward=DummyReward(),
            policy=DummyPolicy(),
            gamma=0.9,
            alpha=0.5,
            lr=1e-2,
            fisher_reg=1e-2,
            mode="invalid",
        )


def test_invalid_implementation_raises():
    with pytest.raises(ValueError, match="Unknown Fisher implementation"):
        FisherNHD(
            reward=DummyReward(),
            policy=DummyPolicy(),
            gamma=0.9,
            alpha=0.5,
            lr=1e-2,
            fisher_reg=1e-2,
            implementation="invalid",
        )


@pytest.mark.parametrize("sketch_size", [None, 0, -1])
def test_sketch_mode_requires_positive_sketch_size(sketch_size):
    with pytest.raises(ValueError, match="sketch_size must be a positive integer"):
        FisherNHD(
            reward=DummyReward(),
            policy=DummyPolicy(),
            gamma=0.9,
            alpha=0.5,
            lr=1e-2,
            fisher_reg=1e-2,
            mode="sketch",
            sketch_size=sketch_size,
        )


@pytest.mark.parametrize("cg_max_iters,cg_tol", [(None, 1e-6), (0, 1e-6), (100, None), (100, 0)])
def test_cg_mode_requires_valid_parameters(cg_max_iters, cg_tol):
    with pytest.raises(ValueError):
        FisherNHD(
            reward=DummyReward(),
            policy=DummyPolicy(),
            gamma=0.9,
            alpha=0.5,
            lr=1e-2,
            fisher_reg=1e-2,
            mode="cg",
            cg_max_iters=cg_max_iters,
            cg_tol=cg_tol,
        )


@pytest.mark.parametrize("fisher_batch_size", [0, -1])
def test_invalid_fisher_batch_size_raises(fisher_batch_size):
    with pytest.raises(ValueError, match="fisher_batch_size must be positive"):
        FisherNHD(
            reward=DummyReward(),
            policy=DummyPolicy(),
            gamma=0.9,
            alpha=0.5,
            lr=1e-2,
            fisher_reg=1e-2,
            fisher_batch_size=fisher_batch_size,
        )


def test_negative_fisher_regularization_raises():
    with pytest.raises(ValueError, match="fisher_reg must be non-negative"):
        FisherNHD(
            reward=DummyReward(),
            policy=DummyPolicy(),
            gamma=0.9,
            alpha=0.5,
            lr=1e-2,
            fisher_reg=-1e-2,
        )
