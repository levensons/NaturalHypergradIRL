import torch
from torch import nn


class Reward(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        n_hidden_layers: int = 2,
        hidden_dim: int = 64,
        clamp_magnitude: float = 10.0,
    ):
        super().__init__()

        self.clamp_magnitude = clamp_magnitude

        layers = []
        in_dim = state_dim + action_dim
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        out = self.net(torch.cat([states, actions], dim=-1))
        out = torch.clamp(out, -self.clamp_magnitude, self.clamp_magnitude)
        out = out.squeeze(-1)
        return out  # (B,)

    def rewards(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.forward(states, actions)  # (B,)

    def trajectory_return(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.forward(states, actions).sum()

    def as_fn(self):
        def reward_fn(states: torch.Tensor, actions: torch.Tensor):
            was_training = self.training
            if was_training:
                self.eval()

            with torch.no_grad():
                out = self.rewards(states, actions)

            if was_training:
                self.train()

            return out  # (B,)

        return reward_fn


class Policy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_low: float,
        action_high: float,
        hidden_dim: int = 64,
        n_hidden_layers: int = 2,
        log_std_min: float = -10,
        log_std_max: float = 1,
    ):
        super().__init__()

        in_dim = state_dim
        layers = []
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        self.mean_head = nn.Linear(in_dim, action_dim)
        self.log_std_head = nn.Linear(in_dim, action_dim)

        self.register_buffer("action_low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(action_high, dtype=torch.float32))
        self.register_buffer("action_scale", (self.action_high - self.action_low) / 2)
        self.register_buffer("action_bias", (self.action_high + self.action_low) / 2)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def reset_parameters(self):
        for layer in self.modules():
            if layer is not self and hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def forward(self, states: torch.Tensor):
        x = self.backbone(states)
        mean = self.mean_head(x)  # (B, action_dim)
        log_std = self.log_std_head(x)  # (B, action_dim)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        return mean, log_std

    def action_distribution(self, states: torch.Tensor):
        # states: (B, state_dim)
        # actions: (B, action_dim)
        mean, log_std = self.forward(states)  # (B, action_dim) x 2
        dist = torch.distributions.Normal(mean, torch.exp(log_std))
        return dist

    def log_prob(self, states: torch.Tensor, actions: torch.Tensor):
        # states: (B, state_dim)
        # actions: (B, action_dim)
        dist = self.action_distribution(states)

        squashed_actions = (actions - self.action_bias) / self.action_scale  # (-1, 1)
        squashed_actions = torch.clamp(squashed_actions, min=-1.0 + 1e-6, max=1.0 - 1e-6)

        raw_actions = 0.5 * (torch.log1p(squashed_actions) - torch.log1p(-squashed_actions))

        log_probs = dist.log_prob(raw_actions)
        correction = torch.log(self.action_scale * (1.0 - torch.pow(squashed_actions, 2)))
        log_probs = log_probs - correction
        return log_probs.sum(dim=-1)  # (B,)

    def sample(
        self,
        states: torch.Tensor,
        deterministic: bool = False,
        return_log_probs: bool = False,
    ):
        # states: (B, state_dim)
        dist = self.action_distribution(states)

        raw_actions = dist.mean if deterministic else dist.rsample()  # (-inf, +inf)
        squashed_actions = torch.tanh(raw_actions)  # (-1; 1)
        actions = self.action_bias + self.action_scale * squashed_actions  # (low, high)
        actions = torch.clamp(actions, min=self.action_low, max=self.action_high)

        if not return_log_probs:
            return actions

        log_probs = dist.log_prob(raw_actions)  # (B, action_dim)
        correction = torch.log(self.action_scale * (1.0 - squashed_actions.pow(2)) + 1e-6)  # (B, action_dim)
        log_probs = log_probs - correction
        log_probs = log_probs.sum(dim=-1)  # (B,)

        return actions, log_probs
