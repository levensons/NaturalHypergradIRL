import pytest
import torch

from src.utils.trajectories import (
    discount_weights,
    mean_trajectory_length,
    mean_trajectory_return,
    trajectory_return,
    trajectory_summary,
)


def test_trajectory_return():
    traj = {"rewards": torch.tensor([1.0, 2.0, -0.5])}

    assert trajectory_return(traj) == pytest.approx(2.5)


def test_mean_trajectory_length():
    trajs = [
        {
            "states": torch.zeros(2, 3),
            "rewards": torch.zeros(2),
        },
        {
            "states": torch.zeros(4, 3),
            "rewards": torch.zeros(4),
        },
    ]

    assert mean_trajectory_length(trajs) == pytest.approx(3.0)


def test_mean_trajectory_return():
    trajs = [
        {
            "states": torch.zeros(2, 3),
            "rewards": torch.tensor([1.0, 2.0]),
        },
        {
            "states": torch.zeros(2, 3),
            "rewards": torch.tensor([4.0, 5.0]),
        },
    ]

    assert mean_trajectory_return(trajs) == pytest.approx(6.0)


def test_trajectory_summary():
    trajs = [
        {
            "states": torch.zeros(2, 3),
            "rewards": torch.tensor([1.0, 2.0]),
        },
        {
            "states": torch.zeros(4, 3),
            "rewards": torch.tensor([4.0, 5.0, 6.0, 7.0]),
        },
    ]

    summary = trajectory_summary(trajs)

    assert summary["len"] == pytest.approx(3.0)
    assert summary["return"] == pytest.approx(12.5)


def test_discount_weights_single_trajectory():
    weights = discount_weights(trajectory_lengths=4, gamma=0.5)
    expected = torch.tensor([1.0, 0.5, 0.25, 0.125])

    assert torch.allclose(weights, expected)


def test_discount_weights_multiple_trajectories():
    weights = discount_weights(trajectory_lengths=[2, 4], gamma=0.5)
    expected = [torch.tensor([1.0, 0.5]), torch.tensor([1.0, 0.5, 0.25, 0.125])]

    assert len(weights) == 2
    assert torch.allclose(weights[0], expected[0])
    assert torch.allclose(weights[1], expected[1])
