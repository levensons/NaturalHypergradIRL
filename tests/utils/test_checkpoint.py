import torch
import torch.nn as nn

from src.utils.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)

    policy = nn.Linear(4, 2)
    reward = nn.Linear(4, 1)

    arch = {
        "state_dim": 4,
        "action_dim": 2,
        "policy_hidden": 64,
        "reward_hidden": 64,
    }

    path = tmp_path / "checkpoints" / "test.pt"

    save_checkpoint(
        path=str(path),
        policy=policy,
        reward=reward,
        arch=arch,
        outer_step=42,
        outer_loss=1.23,
    )

    checkpoint = load_checkpoint(str(path))

    assert path.exists()

    assert checkpoint["format_version"] == 1
    assert checkpoint["arch"] == arch
    assert checkpoint["outer_step"] == 42
    assert checkpoint["outer_loss"] == 1.23

    for name, parameter in policy.state_dict().items():
        assert torch.equal(checkpoint["policy_state_dict"][name], parameter)

    for name, parameter in reward.state_dict().items():
        assert torch.equal(checkpoint["reward_state_dict"][name], parameter)


def test_load_legacy_checkpoint_adds_default_arch(tmp_path):
    path = tmp_path / "legacy.pt"

    torch.save(
        {
            "state_dim": 4,
            "action_dim": 2,
        },
        path,
    )

    checkpoint = load_checkpoint(str(path))

    assert checkpoint["arch"] == {
        "state_dim": 4,
        "action_dim": 2,
        "policy_hidden": 64,
        "policy_n_hidden_layers": 2,
        "reward_hidden": 64,
        "reward_gamma": 0.99,
    }
