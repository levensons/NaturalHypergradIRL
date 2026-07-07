from typing import Any, Protocol
import numpy as np
import torch


class Policy(Protocol):
    def sample(
        self,
        states: Any,
        deterministic: bool = False,
        return_log_probs: bool = False,
    ) -> torch.Tensor:
        ...

    def eval(self) -> None:
        ...

    def train(self) -> None:
        ...


class SB3PolicyWrapper:
    def __init__(self, model, deterministic: bool = True):
        self.model = model
        self.deterministic = deterministic

    def sample(
        self,
        states: Any,
        deterministic: bool | None = None,
        return_log_probs: bool = False,
    ) -> torch.Tensor:
        if return_log_probs:
            raise NotImplementedError("SB3PolicyWrapper does not support return_log_probs=True.")

        if deterministic is None:
            deterministic = self.deterministic

        if isinstance(states, torch.Tensor):
            states_np = states.detach().cpu().numpy()
        else:
            states_np = np.asarray(states)

        action, _ = self.model.predict(states_np, deterministic=deterministic)

        return torch.as_tensor(action, dtype=torch.float32).detach()

    def eval(self) -> None:
        pass

    def train(self) -> None:
        pass


class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def sample(
        self,
        states: Any,
        deterministic: bool = False,
        return_log_probs: bool = False,
    ) -> torch.Tensor:
        if return_log_probs:
            raise NotImplementedError("RandomPolicy does not support return_log_probs=True.")

        action = self.action_space.sample()

        if np.issubdtype(np.asarray(action).dtype, np.integer):
            return torch.as_tensor(action, dtype=torch.long).detach()

        return torch.as_tensor(action, dtype=torch.float32).detach()

    def eval(self) -> None:
        pass

    def train(self) -> None:
        pass