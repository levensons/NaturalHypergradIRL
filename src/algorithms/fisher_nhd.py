from typing import Literal
from tqdm import tqdm
import time
from memory_profiler import profile

import torch
from torch import nn

from src.algorithms.approximations import CBSCFD
from src.utils.policies import Policy
from src.utils.torch import flat_grad, num_params, assign_flat_gradients, to_device
from src.utils.trajectories import discount_weights


class FisherNHD:
    def __init__(
        self,
        reward: nn.Module,
        policy: Policy,
        gamma: float,
        alpha: float,
        lr: float,
        fisher_reg: float,
        max_grad_norm: float | None = None,
        scheduler_gamma: float = 1.0,
        mode: Literal["explicit", "cg", "sketch"] = "explicit",
        sketch_size: int | None = None,
        fisher_batch_size: int = 256,
    ):
        if mode not in {"explicit", "cg", "sketch"}:
            raise ValueError(f"Unknown Fisher solve mode: {mode}. Expected one of {'explicit', 'cg', 'sketch'}.")

        if mode == "sketch" and (sketch_size is None or sketch_size <= 0):
            raise ValueError("sketch_size must be a positive integer when mode='sketch'.")
        
        self.reward = reward
        self.policy = policy
        self.fisher_reg = fisher_reg
        self.gamma = gamma
        self.alpha = alpha
        self.max_grad_norm = max_grad_norm
        self.scheduler_gamma = scheduler_gamma
        self.mode = mode
        self.sketch_size = sketch_size
        self.fisher_batch_size = fisher_batch_size

        self.raw_grad_norm = 0.0
        self.clipped_grad_norm = 0.0

        self.optimizer = torch.optim.SGD(self.reward.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, scheduler_gamma)

    @property
    def device(self) -> torch.device:
        return next(self.reward.parameters()).device

    @property
    def lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    @property
    def reward_num_params(self) -> int:
        return num_params(self.reward)

    @property
    def policy_num_params(self) -> int:
        return num_params(self.policy)

    def d_outer_d_policy(self, expert_trajs, verbose: bool = True) -> torch.Tensor:
        g = torch.zeros(self.policy_num_params, dtype=torch.float32, device=self.device)

        for traj in tqdm(expert_trajs, desc="Outer grad", leave=False, disable=not verbose):
            states = to_device(traj["states"], self.device)
            actions = to_device(traj["actions"], self.device)
            T = states.size(0)

            log_pi_a_s = self.policy.log_prob(states, actions)  # (T,)
            weights = discount_weights(T, self.gamma, self.device, log_pi_a_s.dtype) # (T,)
            sum_weighted_log_pi_a_s = torch.sum(weights * log_pi_a_s) # (1,)

            grad = torch.autograd.grad(
                sum_weighted_log_pi_a_s,
                self.policy.parameters(),
                retain_graph=False,
                create_graph=False,
            )
            grad = flat_grad(grad).detach() # (policy_dim,)

            g.add_(grad)

        g.div_(len(expert_trajs))
        g.neg_()
        return g

    def _grad_log_pi_a_s(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        T = states.size(0)

        log_pi_a_s = self.policy.log_prob(states, actions)  # (T,)
        grad_outputs = torch.eye(T, dtype=log_pi_a_s.dtype, device=self.device)  # (T, T)

        grads = torch.autograd.grad(
            log_pi_a_s,
            self.policy.parameters(),
            grad_outputs=grad_outputs,
            is_grads_batched=True,
            retain_graph=False,
            create_graph=False,
        )
        grads = flat_grad(grads, flat_dim=1).detach()  # (T, policy_dim)

        return grads
    
    def _jvp_grad_log_pi_a_s(self, states: torch.Tensor, actions: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute J @ v, where J = d(log_pi_a_s)/d(theta), (T, policy_dim)
        """
        log_pi_a_s = self.policy.log_prob(states, actions)  # (T,)
        u = torch.zeros_like(log_pi_a_s, requires_grad=True)
        g = torch.autograd.grad(log_pi_a_s, self.policy.parameters(), grad_outputs=u, create_graph=True)
        g_flat = torch.cat([gi.reshape(-1) for gi in g])  # (policy_dim,)
        dot = g_flat @ v
        jv = torch.autograd.grad(dot, u)[0]
        return jv.detach()

    @profile
    def fisher_solve_sketch(self, trajs, g: torch.Tensor, sketch_size: int, verbose: bool = True) -> torch.Tensor:
        sketch = CBSCFD(dim=self.policy_num_params, m=sketch_size, reg=self.fisher_reg, dtype=torch.float32)

        states_buffer = []
        actions_buffer = []
        weights_buffer = []
        buffer_size = 0

        def flush():
            nonlocal buffer_size

            batch_states = torch.cat(states_buffer, dim=0)
            batch_actions = torch.cat(actions_buffer, dim=0)
            batch_weights = torch.cat(weights_buffer, dim=0)

            grad_log_pi_a_s = self._grad_log_pi_a_s(batch_states, batch_actions)  # (B, policy_dim)

            with torch.no_grad():
                row_scale = torch.sqrt((self.alpha / len(trajs)) * batch_weights)  # (B,)
                X_rows = row_scale.reshape(-1, 1) * grad_log_pi_a_s  # (B, policy_dim)

            sketch.extend(X_rows)

            states_buffer.clear()
            actions_buffer.clear()
            weights_buffer.clear()
            buffer_size = 0

        for traj in tqdm(trajs, desc="Fisher sketch", leave=False, disable=not verbose):
            states = to_device(traj["states"], self.device)
            actions = to_device(traj["actions"], self.device)
            T = states.size(0)
            weights = discount_weights(T, self.gamma, self.device, torch.float32)

            start = 0
            while start < T:
                take = min(self.fisher_batch_size - buffer_size, T - start)

                states_buffer.append(states[start : start + take])
                actions_buffer.append(actions[start : start + take])
                weights_buffer.append(weights[start : start + take])

                buffer_size += take
                start += take

                if buffer_size == self.fisher_batch_size:
                    flush()

        if buffer_size > 0:
            flush()

        return sketch.solve(g)

    @profile
    def _fisher_vector_product(self, trajs, v: torch.Tensor, verbose: bool = True) -> torch.Tensor:
        v = v.detach()

        with torch.no_grad():
            out = self.fisher_reg * v.clone()

        states_buffer = []
        actions_buffer = []
        weights_buffer = []
        buffer_size = 0

        policy_params = list(self.policy.parameters())

        def flush():
            nonlocal buffer_size

            batch_states = torch.cat(states_buffer, dim=0)
            batch_actions = torch.cat(actions_buffer, dim=0)
            batch_weights = torch.cat(weights_buffer, dim=0)

            log_pi_a_s = self.policy.log_prob(batch_states, batch_actions)  # (B,)

            u = torch.zeros_like(log_pi_a_s, requires_grad=True)
            jt_u = torch.autograd.grad(
                outputs=log_pi_a_s,
                inputs=policy_params,
                grad_outputs=u,
                create_graph=True,
                retain_graph=True,
            )
            jt_u_flat = flat_grad(jt_u)  # (policy_dim,)

            scalar = torch.dot(jt_u_flat, v)
            jv = torch.autograd.grad(
                outputs=scalar,
                inputs=u,
                retain_graph=True,
                create_graph=False,
            )[0].detach()  # (B,)

            scaled_weights = (self.alpha * batch_weights / len(trajs))  # (B,)

            x = scaled_weights * jv  # (B,)
            jt_x = torch.autograd.grad(
                outputs=log_pi_a_s,
                inputs=policy_params,
                grad_outputs=x,
                retain_graph=False,
                create_graph=False,
            )
            jt_x_flat = flat_grad(jt_x).detach()  # (policy_dim,)

            with torch.no_grad():
                out.add_(jt_x_flat)

            states_buffer.clear()
            actions_buffer.clear()
            weights_buffer.clear()
            buffer_size = 0

        for traj in tqdm(trajs, desc="Fisher vector product", leave=False, disable=not verbose):
            states = to_device(traj["states"], self.device)
            actions = to_device(traj["actions"], self.device)
            T = states.size(0)
            weights = discount_weights(T, self.gamma, self.device, torch.float32)

            start = 0
            while start < T:
                take = min(self.fisher_batch_size - buffer_size, T - start)

                states_buffer.append(states[start : start + take])
                actions_buffer.append(actions[start : start + take])
                weights_buffer.append(weights[start : start + take])

                buffer_size += take
                start += take

                if buffer_size == self.fisher_batch_size:
                    flush()

        if buffer_size > 0:
            flush()

        return out

    @profile
    def fisher_solve_conjugate_gradients(self, trajs, g: torch.Tensor, max_iters: int = 1000, tol: float = 1e-6, verbose: bool = True) -> torch.Tensor:
        x = torch.zeros(self.policy_num_params, dtype=torch.float32, device=self.device)
        r = g.detach().clone()
        p = r.clone()

        with torch.no_grad():
            r_dot_r = torch.dot(r, r)
            g_norm = torch.linalg.vector_norm(g)

        for _ in tqdm(range(max_iters), desc="Fisher solve CG", leave=False, disable=not verbose):
            Fp = self._fisher_vector_product(trajs, p, verbose=False)

            with torch.no_grad():
                p_dot_Fp = torch.dot(p, Fp)
                if p_dot_Fp <= 0:
                    raise FloatingPointError(f"CG expected positive p^T (F + lambda * I) p, got {p_dot_Fp.item():.3e}.")

                alpha = r_dot_r / torch.dot(p, Fp)

                x.add_(p, alpha=alpha)
                r.add_(Fp, alpha=-alpha)

                r_new_dot_r_new = torch.dot(r, r)
                if torch.sqrt(r_new_dot_r_new) <= tol * g_norm:
                    break

                beta = r_new_dot_r_new / r_dot_r
                p.mul_(beta).add_(r)
                r_dot_r = r_new_dot_r_new

        return x

    @profile
    def explicit_fisher(self, trajs, verbose: bool = True) -> torch.Tensor:
        F = torch.zeros(self.policy_num_params, self.policy_num_params, dtype=torch.float32, device=self.device)

        states_buffer = []
        actions_buffer = []
        weights_buffer = []
        buffer_size = 0

        def flush():
            nonlocal buffer_size

            batch_states = torch.cat(states_buffer, dim=0)
            batch_actions = torch.cat(actions_buffer, dim=0)
            batch_weights = torch.cat(weights_buffer, dim=0)

            grad_log_pi_a_s = self._grad_log_pi_a_s(batch_states, batch_actions) # (B, policy_dim)
            scaled_weights = (self.alpha * batch_weights / len(trajs)).reshape(-1, 1) # (B, 1)
            F.addmm_(grad_log_pi_a_s.T, grad_log_pi_a_s * scaled_weights) # (policy_dim, policy_dim)

            states_buffer.clear()
            actions_buffer.clear()
            weights_buffer.clear()
            buffer_size = 0

        for traj in tqdm(trajs, desc="Fisher", leave=False, disable=not verbose):
            states = to_device(traj["states"], self.device)
            actions = to_device(traj["actions"], self.device)
            T = states.size(0)
            weights = discount_weights(T, self.gamma, self.device, torch.float32)

            start = 0
            while start < T:
                take = min(self.fisher_batch_size - buffer_size, T - start)

                states_buffer.append(states[start : start + take])
                actions_buffer.append(actions[start : start + take])
                weights_buffer.append(weights[start : start + take])

                buffer_size += take
                start += take

                if buffer_size == self.fisher_batch_size:
                    flush()

        if buffer_size > 0:
            flush()

        F = 0.5 * (F + F.T)
        return F

    def d_inner_d_cross_vec_product(self, trajs, v: torch.Tensor, verbose: bool = True) -> torch.Tensor:
        """
        Compute (∂²L_inner / ∂φ∂θ) @ v,
        equivalently (∇²_{θφ} L_inner)^T @ v,
        without forming the mixed-derivative matrix.
        """

        if v.numel() != self.policy_num_params:
            raise ValueError(f"`v` must have shape ({self.policy_num_params},), got {tuple(v.shape)}.")

        out = torch.zeros(self.reward_num_params, dtype=torch.float32, device=self.device)

        for traj in tqdm(trajs, desc="Cross vec product", leave=False, disable=not verbose):
            states = to_device(traj["states"], self.device)
            actions = to_device(traj["actions"], self.device)
            T = states.size(0)
            weights = discount_weights(T, self.gamma, self.device, torch.float32)

            jv = self._jvp_grad_log_pi_a_s(states, actions, v).detach()
            prefix_sum_jv = torch.cumsum(jv, dim=0)
            rewards = self.reward(states, actions) # (T,)

            scalar = torch.sum(rewards * weights * prefix_sum_jv)

            grads = torch.autograd.grad(
                scalar,
                self.reward.parameters(),
                retain_graph=False,
                create_graph=False,
            )
            grads = flat_grad(grads).detach()

            out.add_(grads)

        out.div_(len(trajs))
        out.neg_()
        return out

    @profile
    def hypergradient_with_explicit_fisher(self, expert_trajs, agent_trajs) -> torch.Tensor:
        d_outer_d_policy = self.d_outer_d_policy(expert_trajs)  # g

        fisher = self.explicit_fisher(agent_trajs)
        fisher.diagonal().add_(self.fisher_reg) # F + lambda * I

        fisher_inv_d_outer_d_policy = torch.linalg.solve(fisher, d_outer_d_policy)  # (F + lambda * I).inv @ g
        hypergrad = -self.d_inner_d_cross_vec_product(agent_trajs, fisher_inv_d_outer_d_policy)  # -C @ (F + lambda * I).inv @ g

        with torch.no_grad():
            tqdm.write(
                f"Fisher stats | "
                f"outer_grad_norm={d_outer_d_policy.norm().item():.3e} | "
                f"hypergrad_norm={hypergrad.norm().item():.3e} | "
                f"solve_norm={fisher_inv_d_outer_d_policy.norm().item():.3e} | "
                f"solve_abs_max={fisher_inv_d_outer_d_policy.abs().max().item():.3e} | "
            )

        return hypergrad

    @profile
    def hypergradient_with_conjugate_gradients(self, expert_trajs, agent_trajs) -> torch.Tensor:
        d_outer_d_policy = self.d_outer_d_policy(expert_trajs)  # g
        fisher_inv_d_outer_d_policy = self.fisher_solve_conjugate_gradients(agent_trajs, d_outer_d_policy) # (F + lambda * I).inv @ g
        hypergrad = -self.d_inner_d_cross_vec_product(agent_trajs, fisher_inv_d_outer_d_policy)  # -C @ (F + lambda * I).inv @ g

        with torch.no_grad():
            tqdm.write(
                f"Fisher stats | "
                f"outer_grad_norm={d_outer_d_policy.norm().item():.3e} | "
                f"hypergrad_norm={hypergrad.norm().item():.3e} | "
                f"solve_norm={fisher_inv_d_outer_d_policy.norm().item():.3e} | "
                f"solve_abs_max={fisher_inv_d_outer_d_policy.abs().max().item():.3e} | "
            )

        return hypergrad

    @profile
    def hypergradient_with_sketching(self, expert_trajs, agent_trajs) -> torch.Tensor:
        d_outer_d_policy = self.d_outer_d_policy(expert_trajs)  # g
        fisher_inv_d_outer_d_policy = self.fisher_solve_sketch(agent_trajs, d_outer_d_policy, self.sketch_size) # (F + lambda * I)^(-1) @ g
        hypergrad = -self.d_inner_d_cross_vec_product(agent_trajs, fisher_inv_d_outer_d_policy)  # -C @ (F + lambda * I)^(-1) @ g

        with torch.no_grad():
            tqdm.write(
                f"Fisher stats | "
                f"outer_grad_norm={d_outer_d_policy.norm().item():.3e} | "
                f"hypergrad_norm={hypergrad.norm().item():.3e} | "
                f"solve_norm={fisher_inv_d_outer_d_policy.norm().item():.3e} | "
                f"solve_abs_max={fisher_inv_d_outer_d_policy.abs().max().item():.3e} | "
            )

        return hypergrad

    def step(self, expert_trajs, agent_trajs) -> torch.Tensor:
        if self.mode == "explicit":
            hypergradient = self.hypergradient_with_explicit_fisher(expert_trajs, agent_trajs)
        elif self.mode == "cg":
            hypergradient = self.hypergradient_with_conjugate_gradients(expert_trajs, agent_trajs)
        elif self.mode == "sketch":
            hypergradient = self.hypergradient_with_sketching(expert_trajs, agent_trajs)
        else:
            raise ValueError(f"Unknown Fisher solve mode: {self.mode}. Expected one of {'explicit', 'cg', 'sketch'}.")

        if not torch.isfinite(hypergradient).all():
            raise FloatingPointError("FisherNHD gradient contains NaN or Inf.")

        self.raw_grad_norm = hypergradient.norm().item()
        if self.max_grad_norm is not None and self.raw_grad_norm > self.max_grad_norm:
            hypergradient = hypergradient * (self.max_grad_norm / self.raw_grad_norm)
        self.clipped_grad_norm = hypergradient.norm().item()

        self.optimizer.zero_grad()
        assign_flat_gradients(self.reward, hypergradient)
        self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()

        return hypergradient
    
    def d_inner_d_policy(self, trajs, verbose: bool = True) -> torch.Tensor:
        """
        DIAGNOSTICS ONLY.
        """
        out = torch.zeros(self.policy_num_params, dtype=torch.float32, device=self.device)

        for traj in tqdm(trajs, desc="Inner gradient", leave=False, disable=not verbose):
            states = to_device(traj["states"], self.device)
            actions = to_device(traj["actions"], self.device)

            log_probs = self.policy.log_prob(states, actions)  # (T,)
            rewards = self.reward(states, actions).detach()    # (T,)

            T = log_probs.numel()
            weights = discount_weights(T, self.gamma, self.device, log_probs.dtype)

            per_step_loss = weights * (self.alpha * log_probs.detach() - rewards)

            tail_losses = torch.flip(torch.cumsum(torch.flip(per_step_loss, dims=[0]), dim=0), dims=[0]).detach()

            # Score-function term:
            # sum_t tail_loss_t * grad log pi_t
            score_term = torch.autograd.grad(
                outputs=(tail_losses * log_probs).sum(),
                inputs=list(self.policy.parameters()),
                retain_graph=True,
                create_graph=False,
            )
            score_term = flat_grad(score_term).detach()

            # Explicit derivative of alpha * log pi
            entropy_direct_term = torch.autograd.grad(
                outputs=self.alpha * (weights * log_probs).sum(),
                inputs=list(self.policy.parameters()),
                retain_graph=False,
                create_graph=False,
            )
            entropy_direct_term = flat_grad(entropy_direct_term).detach()

            out.add_(score_term)
            out.add_(entropy_direct_term)

        out.div_(len(trajs))
        return out
