from typing import Any, Tuple
import torch
from torch import nn


def flat_grad(params):
    return torch.cat([p.flatten() for p in params])


def num_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def assign_flat_gradients(module, flat_grad: torch.Tensor):
    i = 0

    for p in module.parameters():
        n = p.numel()
        grad_chunk = flat_grad[i : i + n]
        if grad_chunk.numel() != n:
            raise ValueError("Flat gradient has incorrect size: not enough elements.")
        p.grad = grad_chunk.reshape(p.shape).clone()
        i += n

    if i != flat_grad.numel():
        raise ValueError("Flat gradient has incorrect size: too many elements.")


def safe_clip_grad(grad: torch.Tensor, max_norm: float | None) -> Tuple[torch.Tensor, bool]:
    if not torch.isfinite(grad).all():
        return torch.zeros_like(grad), False

    if max_norm is None:
        return grad, True

    norm = grad.norm()

    if norm.item() == 0.0:
        return grad, True

    if norm.item() > max_norm:
        grad = grad * (max_norm / norm)

    return grad, True


def to_device(obj: Any, device: torch.device | str) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)

    if isinstance(obj, nn.Module):
        return obj.to(device)

    if isinstance(obj, dict):
        return {key: to_device(value, device) for key, value in obj.items()}

    if isinstance(obj, list):
        return [to_device(value, device) for value in obj]

    if isinstance(obj, tuple):
        return tuple(to_device(value, device) for value in obj)

    return obj
