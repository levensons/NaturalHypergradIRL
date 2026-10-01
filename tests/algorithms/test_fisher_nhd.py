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

    policy = DummyPolicy()
    reward = DummyReward()

    return FisherNHD(
        reward=reward,
        policy=policy,
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


def test_explicit_fisher_is_symmetric(nhd, trajectories):
    fisher = nhd.explicit_fisher(trajectories, verbose=False)

    assert fisher.shape == (nhd.policy_num_params, nhd.policy_num_params)
    assert torch.allclose(fisher, fisher.T, rtol=1e-6, atol=1e-7)


def test_explicit_fisher_is_positive_semidefinite(nhd, trajectories):
    fisher = nhd.explicit_fisher(trajectories, verbose=False)
    eigenvalues = torch.linalg.eigvalsh(fisher)

    assert eigenvalues.min().item() >= -1e-6


def test_fisher_vector_product_matches_explicit_fisher(nhd, trajectories):
    torch.manual_seed(1)

    v = torch.randn(nhd.policy_num_params)

    fisher = nhd.explicit_fisher(trajectories, verbose=False)
    expected = fisher @ v + nhd.fisher_reg * v

    result = nhd._fisher_vector_product(trajectories, v, verbose=False)

    assert torch.allclose(result, expected, rtol=1e-5, atol=1e-6)


def test_conjugate_gradients_matches_explicit_solve(nhd, trajectories):
    torch.manual_seed(1)

    g = torch.randn(nhd.policy_num_params)

    fisher = nhd.explicit_fisher(trajectories, verbose=False)
    fisher.diagonal().add_(nhd.fisher_reg)

    expected = torch.linalg.solve(fisher, g)

    result = nhd.fisher_solve_conjugate_gradients(
        trajectories,
        g,
        max_iters=100,
        tol=1e-8,
        verbose=False,
    )

    assert torch.allclose(result, expected, rtol=1e-4, atol=1e-5)


def test_conjugate_gradients_has_small_residual(nhd, trajectories):
    torch.manual_seed(1)

    g = torch.randn(nhd.policy_num_params)

    result = nhd.fisher_solve_conjugate_gradients(
        trajectories,
        g,
        max_iters=100,
        tol=1e-8,
        verbose=False,
    )

    residual = nhd._fisher_vector_product(trajectories, result, verbose=False) - g
    relative_residual = torch.linalg.vector_norm(residual) / torch.linalg.vector_norm(g)

    assert relative_residual.item() < 1e-5


def test_grad_log_pi_shape_and_finiteness(nhd, trajectories):
    states = trajectories[0]["states"]
    actions = trajectories[0]["actions"]

    grads = nhd._grad_log_pi_a_s(states, actions)

    assert grads.shape == (states.shape[0], nhd.policy_num_params)
    assert torch.isfinite(grads).all()


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
