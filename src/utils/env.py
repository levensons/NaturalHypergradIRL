import gymnasium as gym
from gymnasium import spaces
import torch
import numpy as np


class Environment:
    def __init__(self, id: str, seed: int, custom_reward_fn = None):
        self.env = gym.make(id)
        self.env.reset(seed=seed)
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)

        if not isinstance(self.env.observation_space, spaces.Box):
            raise TypeError(
                f"Unsupported observation space: {type(self.env.observation_space)}. "
                "Only Box observation spaces are supported."
            )

        self.state_dim = int(np.prod(self.env.observation_space.shape))

        self.is_discrete = isinstance(self.env.action_space, spaces.Discrete)
        self.is_continuous = isinstance(self.env.action_space, spaces.Box)

        if self.is_discrete:
            self.action_dim = self.env.action_space.n
            self.action_low = None
            self.action_high = None

        elif self.is_continuous:
            self.action_dim = int(np.prod(self.env.action_space.shape))
            self.action_low = torch.as_tensor(self.env.action_space.low, dtype=torch.float32)
            self.action_high = torch.as_tensor(self.env.action_space.high, dtype=torch.float32)

        else:
            raise TypeError(
                f"Unsupported action space: {type(self.env.action_space)}. "
                "Only Discrete and Box action spaces are supported."
            )
        
        self.custom_reward_fn = custom_reward_fn

        self.reset()

    @property
    def action_space(self):
        return self.env.action_space
    
    @property
    def observation_space(self):
        return self.env.observation_space

    def reset(self):
        state, _ = self.env.reset()
        state = torch.as_tensor(state, dtype=torch.float32).flatten()
        self.state = state
        return state
    
    def step(self, action: torch.Tensor):
        if self.is_discrete:
            action_for_env = int(action.item())
        else:
            action_for_env = action.detach().cpu().numpy()
            action_for_env = np.asarray(action_for_env, dtype=np.float32).reshape(self.env.action_space.shape)

        next_state, reward, terminated, truncated, _ = self.env.step(action_for_env)
        done = terminated or truncated

        if self.custom_reward_fn is not None:
            reward = self.custom_reward_fn(self.state, action)
        
        next_state = torch.as_tensor(next_state, dtype=torch.float32).flatten()
        reward = torch.as_tensor(reward, dtype=torch.float32).reshape(1)
        done = torch.as_tensor(done, dtype=torch.float32).reshape(1)

        self.state = next_state

        return next_state, reward, done
    
    def get_random_action(self):
        action = self.env.action_space.sample()
        action = torch.as_tensor(action, dtype=torch.float32)
        return action
    
    def close(self):
        self.env.close()
