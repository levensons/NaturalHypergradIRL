from tqdm import tqdm

import torch
from torch import nn

from src.utils.torch import flat_grad, assign_flat_gradients, to_device
from src.utils.trajectories import discount_weights


class MLIRL:
    def __init__(
        self,
        reward: nn.Module,
        alpha: float,
        gamma: float,
        lr: float,
        max_grad_norm: float | None = None,
    ):
        self.reward = reward
        self.alpha = alpha
        self.gamma = gamma

        self.max_grad_norm = max_grad_norm
        self.optimizer = torch.optim.SGD(self.reward.parameters(), lr=lr)

        self.raw_grad_norm = 0.0
        self.clipped_grad_norm = 0.0
        self.loss_value = 0.0
        self.expert_learned_return = 0.0
        self.agent_learned_return = 0.0

    @property
    def device(self) -> torch.device:
        return next(self.reward.parameters()).device

    @property
    def lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def mean_discounted_reward(self, trajs) -> torch.Tensor:
        out = []

        for traj in trajs:
            states = to_device(traj["states"], self.device)
            actions = to_device(traj["actions"], self.device) 
            T = states.size(0)

            weights = discount_weights(T, self.gamma, self.device, states.dtype)
            rewards = self.reward(states, actions)
            out.append(torch.sum(weights * rewards))

        out = torch.stack(out).mean()
        return out

    def gradient(self, expert_trajs, agent_trajs) -> torch.Tensor:
        agent_value = self.mean_discounted_reward(agent_trajs)
        expert_value = self.mean_discounted_reward(expert_trajs)
        loss = (agent_value - expert_value) / self.alpha

        gradient = torch.autograd.grad(
            loss,
            self.reward.parameters(),
            retain_graph=False,
            create_graph=False,
        )
        gradient = flat_grad(gradient).detach()

        self.loss_value = float(loss.detach().item())
        self.expert_learned_return = float(expert_value.detach().item())
        self.agent_learned_return = float(agent_value.detach().item())

        return gradient

    def step(self, expert_trajs, agent_trajs) -> torch.Tensor:
        gradient = self.gradient(expert_trajs, agent_trajs)

        if not torch.isfinite(gradient).all():
            raise FloatingPointError("ML-IRL gradient contains NaN or Inf.")

        self.raw_grad_norm = float(gradient.norm().item())

        if self.max_grad_norm is not None and self.max_grad_norm > 0.0 and self.raw_grad_norm > self.max_grad_norm:
            scale = self.max_grad_norm / self.raw_grad_norm
            gradient = gradient * scale

        self.clipped_grad_norm = float(gradient.norm().item())

        self.optimizer.zero_grad(set_to_none=True)
        assign_flat_gradients(self.reward, gradient)
        self.optimizer.step()

        return gradient

    def stats(self) -> dict[str, float]:
        return {
            "reward_loss": self.loss_value,
            "expert_learned_return": (self.expert_learned_return),
            "agent_learned_return": (self.agent_learned_return),
            "reward_grad_raw": (self.raw_grad_norm),
            "reward_grad_clipped": (self.clipped_grad_norm),
            "lr_reward": self.lr,
        }
