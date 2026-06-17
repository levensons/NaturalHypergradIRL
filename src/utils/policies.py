from typing import Any, Protocol


class Policy(Protocol):
    def sample_action(self, state: Any) -> Any: ...


class SB3PolicyWrapper:
    def __init__(self, model, deterministic: bool = True):
        self.model = model
        self.deterministic = deterministic

    def sample_action(self, state):
        action, _ = self.model.predict(state, deterministic=self.deterministic)
        return action

    def eval(self):
        pass

    def train(self):
        pass


class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def sample_action(self, state):
        return self.action_space.sample()

    def eval(self):
        pass

    def train(self):
        pass
