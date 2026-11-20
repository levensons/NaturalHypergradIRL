import pytest
import torch
import torch.nn as nn

from src.utils.torch import assign_flat_gradients, flat_grad, num_params, safe_clip_grad, set_optimizer_lr, to_device


def test_flat_grad():
    grads = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([5.0, 6.0]),
    ]

    result = flat_grad(grads)
    expected = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    assert torch.equal(result, expected)


def test_flat_grad_ignores_none():
    grads = [
        torch.tensor([1.0, 2.0]),
        None,
        torch.tensor([3.0]),
    ]

    result = flat_grad(grads)
    expected = torch.tensor([1.0, 2.0, 3.0])

    assert torch.equal(result, expected)


def test_flat_grad_batched():
    grads = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([[5.0], [6.0]]),
    ]

    result = flat_grad(grads, flat_dim=1)
    expected = torch.tensor([[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]])

    assert torch.equal(result, expected)


def test_flat_grad_rejects_invalid_dimension():
    grads = [torch.tensor([1.0, 2.0])]

    with pytest.raises(ValueError):
        flat_grad(grads, flat_dim=2)


def test_num_params():
    module = nn.Sequential(
        nn.Linear(3, 4),
        nn.Linear(4, 2),
    )

    # First layer: 3 * 4 weights + 4 biases = 16
    # Second layer: 4 * 2 weights + 2 biases = 10
    assert num_params(module) == 26


def test_assign_flat_gradients():
    module = nn.Sequential(
        nn.Linear(2, 2),
        nn.Linear(2, 1),
    )

    flat = torch.arange(num_params(module), dtype=torch.float32)

    assign_flat_gradients(module, flat)

    reconstructed = torch.cat([parameter.grad.reshape(-1) for parameter in module.parameters()])

    assert torch.equal(reconstructed, flat)


def test_assign_flat_gradients_rejects_too_few_elements():
    module = nn.Linear(2, 2)
    flat = torch.zeros(num_params(module) - 1)

    with pytest.raises(ValueError, match="not enough elements"):
        assign_flat_gradients(module, flat)


def test_assign_flat_gradients_rejects_too_many_elements():
    module = nn.Linear(2, 2)
    flat = torch.zeros(num_params(module) + 1)

    with pytest.raises(ValueError, match="too many elements"):
        assign_flat_gradients(module, flat)


def test_safe_clip_grad_without_clipping():
    grad = torch.tensor([3.0, 4.0])

    result, valid = safe_clip_grad(grad, max_norm=None)

    assert valid
    assert torch.equal(result, grad)


def test_safe_clip_grad():
    grad = torch.tensor([3.0, 4.0])

    result, valid = safe_clip_grad(grad, max_norm=1.0)

    assert valid
    assert result.norm().item() == pytest.approx(1.0)


def test_safe_clip_grad_nonfinite():
    grad = torch.tensor([1.0, float("nan"), 2.0])

    result, valid = safe_clip_grad(grad, max_norm=1.0)

    assert not valid
    assert torch.equal(result, torch.zeros_like(grad))


def test_to_device_nested_structure():
    obj = {
        "tensor": torch.tensor([1.0]),
        "list": [torch.tensor([2.0])],
        "tuple": (torch.tensor([3.0]),),
        "value": 42,
    }

    result = to_device(obj, "cpu")

    assert result["tensor"].device.type == "cpu"
    assert result["list"][0].device.type == "cpu"
    assert result["tuple"][0].device.type == "cpu"
    assert result["value"] == 42


def test_set_optimizer_lr():
    model = nn.Linear(2, 1)

    optimizer = torch.optim.SGD(
        [
            {"params": [model.weight], "lr": 0.1},
            {"params": [model.bias], "lr": 0.01},
        ]
    )

    set_optimizer_lr(optimizer, lr=0.001)

    assert all(group["lr"] == pytest.approx(0.001) for group in optimizer.param_groups)
