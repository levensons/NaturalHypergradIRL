import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Reward(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_hidden_layers: int = 2,
        hidden_dim: int = 64,
        clamp_magnitude: float = 10.0,
    ):
        super().__init__()

        self.action_dim = int(action_dim)
        self.clamp_magnitude = float(clamp_magnitude)

        layers = []
        in_dim = state_dim + action_dim

        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def _prepare_actions(self, actions: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        actions = torch.as_tensor(actions, device=actions.device)

        if actions.ndim > 0 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)

        actions = actions.long()
        return F.one_hot(actions, num_classes=self.action_dim).to(dtype=dtype)

    def forward(self,states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        actions_one_hot = self._prepare_actions(actions, states.dtype)
        out = torch.cat([states, actions_one_hot], dim=-1)
        out = self.net(out)
        out = torch.clamp(out, -self.clamp_magnitude, self.clamp_magnitude)
        out = out.squeeze(-1)
        return out  # (B,)

    def rewards(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.forward(states, actions)

    def trajectory_return(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.forward(states, actions).sum()

    def as_fn(self):
        def reward_fn(states: torch.Tensor, actions: torch.Tensor):
            device = next(self.parameters()).device
            was_training = self.training

            if was_training:
                self.eval()

            with torch.no_grad():
                states_device = torch.as_tensor(
                    states,
                    dtype=torch.float32,
                    device=device,
                )
                actions_device = torch.as_tensor(actions, device=device)
                out = self.rewards(states_device, actions_device)
                return out

            if was_training:
                self.train()

        return reward_fn


class Policy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        n_hidden_layers: int = 1,
    ):
        super().__init__()

        layers = []
        in_dim = state_dim

        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def reset_parameters(self):
        for layer in self.modules():
            if layer is not self and hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)

    def action_distribution(self, states: torch.Tensor) -> Categorical:
        return Categorical(logits=self.forward(states))

    def log_prob(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim > 0 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)

        dist = self.action_distribution(states)
        log_probs = dist.log_prob(actions.long())
        return log_probs

    def sample(
        self,
        states: torch.Tensor,
        deterministic: bool = False,
        return_log_probs: bool = False,
    ):
        dist = self.action_distribution(states)
        actions = torch.argmax(dist.logits, dim=-1) if deterministic else dist.sample()

        if not return_log_probs:
            return actions

        return actions, dist.log_prob(actions)
