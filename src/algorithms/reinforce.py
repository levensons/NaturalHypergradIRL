import torch
from tqdm import tqdm

from src.utils.policies import Policy
from src.utils.env import Environment
from src.utils.torch import flat_grad, assign_flat_gradients, to_device, num_params
from src.utils.trajectories import collect_trajectories, discount_weights


class REINFORCE:
    def __init__(
        self,
        policy: Policy,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        alpha: float = 1.0,
    ):
        self.policy = policy
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.alpha = alpha

    def gradient(self, trajs) -> torch.Tensor:
        device = next(self.policy.parameters()).device
        policy_params = list(self.policy.parameters())

        gradient = torch.zeros(num_params(self.policy), device=device, dtype=torch.float32)
        for traj in trajs:
            states = to_device(traj["states"], device)
            actions = to_device(traj["actions"], device)
            rewards = to_device(traj["rewards"], device)
            T = states.size(0)
            weights = discount_weights(T, self.gamma, device, torch.float32)

            log_probs = self.policy.log_prob(states, actions)  # (T,)
            log_probs_sum = log_probs.sum()  # (1,)
            weighted_log_prob_sum = (weights * log_probs).sum()  # (1,)
            coef = (weights * (self.alpha * log_probs.detach() - rewards)).sum().detach()  # (1,)

            grad_sum_log_probs = torch.autograd.grad(log_probs_sum, policy_params, retain_graph=True, create_graph=False)
            grad_sum_log_probs = flat_grad(grad_sum_log_probs).detach()

            grad_sum_weighted_log_probs = torch.autograd.grad(
                weighted_log_prob_sum,
                policy_params,
                retain_graph=False,
                create_graph=False,
            )
            grad_sum_weighted_log_probs = flat_grad(grad_sum_weighted_log_probs).detach()

            grad_traj = coef * grad_sum_log_probs + self.alpha * grad_sum_weighted_log_probs
            gradient += grad_traj

        gradient /= len(trajs)
        return gradient

    def optimize(
        self,
        train_env: Environment,
        total_steps: int = 100,
        n_traj_per_update: int = 10,
        max_grad_norm: float = None,
        actor_lr: float = 1e-3,
        scheduler_gamma: float = 1.0,
        validate_fn=None,
        validate_every: int = 10,
    ):
        policy_params = list(self.policy.parameters())

        policy_optimizer = torch.optim.Adam(policy_params, lr=actor_lr)
        policy_scheduler = torch.optim.lr_scheduler.ExponentialLR(policy_optimizer, gamma=scheduler_gamma)

        for ts in tqdm(range(total_steps), desc="REINFORCE optimization", leave=False):
            trajs = collect_trajectories(
                env=train_env,
                policy=self.policy,
                n=n_traj_per_update,
                deterministic=False,
                verbose=False,
            )

            gradient = self.gradient(trajs)

            self.raw_grad_norm = float(gradient.norm().item())
            if max_grad_norm is not None and self.raw_grad_norm > max_grad_norm:
                gradient = gradient * (max_grad_norm / (self.raw_grad_norm + 1e-12))
            self.clipped_grad_norm = float(gradient.norm().item())

            policy_optimizer.zero_grad()
            assign_flat_gradients(self.policy, gradient)
            policy_optimizer.step()
            policy_scheduler.step()

            if validate_fn is not None and ts % validate_every == 0:
                validate_fn(ts=ts)
