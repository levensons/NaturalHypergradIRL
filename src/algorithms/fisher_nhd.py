from tqdm import tqdm
import time

import torch
from torch import nn
import torch.nn.functional as F

from src.utils.policies import Policy
from src.algorithms.approximations import SCFD, CBSCFD
from src.utils.torch import flat_grad, num_params, assign_flat_gradients, to_device
from src.utils.trajectories import discount_weights
from src.evaluation.metrics import relative_error


class FisherNHD:
    def __init__(
        self,
        reward: nn.Module,
        policy: Policy,
        lr: float,
        fisher_reg: float,
        discount: float,
        alpha: float,
        max_grad_norm=None,
        scheduler_gamma: float = 1.0,
        sketch_size: int = 128,
    ):
        self.reward = reward
        self.policy = policy
        self.fisher_reg = fisher_reg
        self.discount = discount
        self.alpha = alpha
        self.max_grad_norm = max_grad_norm
        self.scheduler_gamma = scheduler_gamma
        self.sketch_size = sketch_size

        self.raw_grad_norm = 0.0
        self.clipped_grad_norm = 0.0

        self.optimizer = torch.optim.Adam(self.reward.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, scheduler_gamma)

        self.outer_step = 0

    def d_outer_d_policy(self, expert_trajs) -> torch.Tensor:
        policy_dim = num_params(self.policy)
        device = next(self.policy.parameters()).device

        E = torch.zeros(policy_dim, dtype=torch.float32, device=device)

        for traj in tqdm(expert_trajs, desc="Outer grad", leave=False):
            states = to_device(traj["states"], device)
            actions = to_device(traj["actions"], device)

            log_pi_a_s = self.policy.log_prob(states, actions)  # (T,)
            sum_log_pi_a_s = log_pi_a_s.sum() # (1,)

            grad = torch.autograd.grad(
                sum_log_pi_a_s,
                self.policy.parameters(),
                retain_graph=False,
                create_graph=False,
            )
            grad = flat_grad(grad).detach() # (policy_dim,)

            E.add_(grad)

        E.div_(len(expert_trajs))
        E.neg_()
        return E
    
    def explicit_hessian(self, trajs) -> torch.Tensor:
        "DEPRECATED"
        policy_dim = num_params(self.policy)
        policy_params = list(self.policy.parameters())
        device = next(self.policy.parameters()).device

        H = torch.zeros(policy_dim, policy_dim, dtype=torch.float32, device=device)

        for traj in tqdm(trajs, desc="Explicit Hessian", leave=False):
            states = to_device(traj["states"], device)
            actions = to_device(traj["actions"], device)
            T = states.size(0)

            weights = discount_weights(T, self.discount, device, dtype=torch.float32)

            log_probs = self.policy.log_prob(states, actions)  # (T,)
            rewards = self.reward(states, actions).detach()  # (T,)

            L = torch.sum(weights * (self.alpha * log_probs - rewards)).detach()  # (1,)

            log_probs_sum = log_probs.sum()
            weighted_log_probs_sum = (weights * log_probs).sum()

            S = torch.autograd.grad(log_probs_sum, policy_params, retain_graph=True, create_graph=False)
            S = flat_grad(S).detach()  # (policy_dim,)

            S_gamma = torch.autograd.grad(weighted_log_probs_sum, policy_params, retain_graph=True, create_graph=False)
            S_gamma = flat_grad(S_gamma).detach()  # (policy_dim,)

            hessian_scalar = L * log_probs_sum + self.alpha * weighted_log_probs_sum

            grad_hessian_scalar = torch.autograd.grad(
                hessian_scalar, policy_params, retain_graph=True, create_graph=True
            )
            grad_hessian_scalar = flat_grad(grad_hessian_scalar)  # (policy_dim,)

            grad_outputs = torch.eye(policy_dim, dtype=grad_hessian_scalar.dtype, device=device)

            second_grads = torch.autograd.grad(
                grad_hessian_scalar,
                policy_params,
                grad_outputs=grad_outputs,
                is_grads_batched=True,
                retain_graph=False,
                create_graph=False,
            )

            L_H_plus_alpha_H_gamma = flat_grad(second_grads, flat_dim=1).detach()

            H.add_(
                L * torch.outer(S, S)
                + self.alpha * (torch.outer(S_gamma, S) + torch.outer(S, S_gamma))
                + L_H_plus_alpha_H_gamma
            )

        H.div_(len(trajs))
        H = 0.5 * (H + H.T)
        return H

    def _grad_R_tail_with_discount(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        device = next(self.reward.parameters()).device
        T = states.size(0)

        r_a_s_t = self.reward(states, actions)  # (T,)
        weights = discount_weights(T, self.discount, device)  # (T,)
        r_a_s_t = weights * r_a_s_t

        grad_outputs = torch.eye(T, dtype=r_a_s_t.dtype, device=device)  # (T, T)

        grads = torch.autograd.grad(
            r_a_s_t,
            self.reward.parameters(),
            grad_outputs=grad_outputs,
            is_grads_batched=True,
            retain_graph=False,
            create_graph=False,
        )
        grads = flat_grad(grads, flat_dim=1).detach()  # (T, reward_dim)

        suffix_sums = torch.flip(torch.cumsum(torch.flip(grads, dims=[0]), dim=0), dims=[0])  # (T, reward_dim)
        return suffix_sums

    def _grad_log_pi_a_s(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        device = next(self.policy.parameters()).device
        T = states.size(0)

        log_pi_a_s = self.policy.log_prob(states, actions)  # (T,)
        grad_outputs = torch.eye(T, dtype=log_pi_a_s.dtype, device=device)  # (T, T)

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

    def fisher_solve_sketch_profile(self, trajs, g: torch.Tensor, sketch_size: int) -> torch.Tensor:
        policy_dim = num_params(self.policy)
        device = next(self.policy.parameters()).device

        g = g.to(device=device, dtype=torch.float32)
        sketch = CBSCFD(dim=policy_dim, m=sketch_size, alpha_0=self.fisher_reg)

        score_time = 0.0
        extend_time = 0.0

        n_trajs = len(trajs)
        for traj in tqdm(trajs, desc="Fisher sketch", leave=False):
            states = to_device(traj["states"], device)
            actions = to_device(traj["actions"], device)
            T = states.size(0)

            start = time.perf_counter()
            grad_log_pi_a_s = self._grad_log_pi_a_s(states, actions).to(torch.float32)  # (T, policy_dim)
            score_time += time.perf_counter() - start

            weights = discount_weights(T, self.discount, device, dtype=torch.float32)  # (T,)

            row_scale = torch.sqrt((self.alpha / n_trajs) * weights)  # (T,)
            X_rows = row_scale.reshape(-1, 1) * grad_log_pi_a_s  # (T, policy_dim)

            start = time.perf_counter()
            sketch.extend(X_rows)
            extend_time += time.perf_counter() - start

        solution = sketch.solve(g)

        stats = sketch.profiling_stats()

        stats["score_time"] = score_time
        stats["extend_time"] = extend_time
        stats["total_time"] = score_time + extend_time + stats["solve_time"]

        return solution, stats

    def fisher_solve_sketch(self, trajs, g: torch.Tensor, sketch_size: int) -> torch.Tensor:
        policy_dim = num_params(self.policy)
        device = next(self.policy.parameters()).device

        g = g.to(device=device, dtype=torch.float32)
        sketch = CBSCFD(dim=policy_dim, m=sketch_size, alpha_0=self.fisher_reg)

        n_trajs = len(trajs)
        for traj in tqdm(trajs, desc="Fisher sketch", leave=False):
            states = to_device(traj["states"], device)
            actions = to_device(traj["actions"], device)
            T = states.size(0)

            grad_log_pi_a_s = self._grad_log_pi_a_s(states, actions).to(torch.float32)  # (T, policy_dim)
            weights = discount_weights(T, self.discount, device, dtype=torch.float32)  # (T,)

            row_scale = torch.sqrt((self.alpha / n_trajs) * weights)  # (T,)
            X_rows = row_scale.reshape(-1, 1) * grad_log_pi_a_s  # (T, policy_dim)

            sketch.extend(X_rows)

        return sketch.solve(g)

    def fisher(self, trajs) -> torch.Tensor:
        "DEPRECATED"
        policy_dim = num_params(self.policy)
        device = next(self.policy.parameters()).device

        F = torch.zeros(policy_dim, policy_dim, dtype=torch.float64, device=device)

        for traj in tqdm(trajs, desc="Fisher", leave=False):
            states = to_device(traj["states"], device)
            actions = to_device(traj["actions"], device)
            T = states.size(0)

            grad_log_pi_a_s = self._grad_log_pi_a_s(states, actions).to(torch.float64)  # (T, policy_dim)
            weights = discount_weights(T, self.discount, device, dtype=torch.float64)  # (T,)
            F.add_(torch.einsum("t,ti,tj->ij", weights, grad_log_pi_a_s, grad_log_pi_a_s))  # (policy_dim, policy_dim)

        F.div_(len(trajs))
        F = 0.5 * (F + F.T)
        F.mul_(self.alpha)
        return F

    def d_inner_d_cross_vec_product(self, trajs, v: torch.Tensor) -> torch.Tensor:
        "d_inner_d_cross @ v = u"

        reward_dim = num_params(self.reward)
        policy_dim = num_params(self.policy)
        device = next(self.policy.parameters()).device

        v = v.to(dtype=torch.float32, device=device)

        if v.numel() != policy_dim:
            raise ValueError(f"`v` must have shape ({policy_dim},), got {tuple(v.shape)}.")

        U = torch.zeros(reward_dim, dtype=torch.float32, device=device)

        for traj in tqdm(trajs, desc="Cross vec product", leave=False):
            states = to_device(traj["states"], device)
            actions = to_device(traj["actions"], device)
            T = states.size(0)

            grad_log_pi_a_s = self._grad_log_pi_a_s(states, actions)  # (T, policy_dim)
            score_v = (grad_log_pi_a_s * v.reshape(1, -1)).sum(dim=1)  # (T,)
            prefix = torch.cumsum(score_v, dim=0).detach()  # (T,)

            rewards = self.reward(states, actions)  # (T,)
            weights = discount_weights(T, self.discount, device)  # (T,)
            scalar = (prefix * weights * rewards).sum()
            grad = torch.autograd.grad(scalar, self.reward.parameters(), retain_graph=False, create_graph=False)
            grad = flat_grad(grad).detach()

            U.add_(grad)  # (reward_dim,)

        U.div_(len(trajs))
        U.neg_()
        return U

    def hypergradient_with_explicit_hessian(self, expert_trajs, agent_trajs) -> torch.Tensor:
        "DEPRECATED"
        d_outer_d_policy = self.d_outer_d_policy(expert_trajs)  # (policy_dim,)
        hessian = self.explicit_hessian(agent_trajs)
        fisher_inv_d_outer_d_policy = torch.linalg.solve(hessian, d_outer_d_policy)
        hypergrad = -self.d_inner_d_cross_vec_product(agent_trajs, fisher_inv_d_outer_d_policy)  # (reward_dim,)

        with torch.no_grad():
            tqdm.write(
                f"Fisher stats | "
                f"outer_grad_norm={d_outer_d_policy.norm().item():.3e} | "
                f"hypergrad_norm={hypergrad.norm().item():.3e} | "
                f"solve_norm={fisher_inv_d_outer_d_policy.norm().item():.3e} | "
                f"solve_abs_max={fisher_inv_d_outer_d_policy.abs().max().item():.3e} | "
            )

        return hypergrad
    
    def hypergradient_with_exact_fisher(self, expert_trajs, agent_trajs) -> torch.Tensor:
        d_outer_d_policy = self.d_outer_d_policy(expert_trajs).to(dtype=torch.float64)  # (policy_dim,)

        fisher = self.fisher(agent_trajs)  # (policy_dim, policy_dim), float64
        fisher.diagonal().add_(self.fisher_reg)

        fisher_inv_d_outer_d_policy = torch.linalg.solve(fisher, d_outer_d_policy)  # (policy_dim,)
        fisher_inv_d_outer_d_policy = fisher_inv_d_outer_d_policy.to(dtype=torch.float32)

        hypergrad = -self.d_inner_d_cross_vec_product(agent_trajs, fisher_inv_d_outer_d_policy)  # (reward_dim,)

        with torch.no_grad():
            tqdm.write(
                f"Fisher stats | "
                f"outer_grad_norm={d_outer_d_policy.norm().item():.3e} | "
                f"hypergrad_norm={hypergrad.norm().item():.3e} | "
                f"solve_norm={fisher_inv_d_outer_d_policy.norm().item():.3e} | "
                f"solve_abs_max={fisher_inv_d_outer_d_policy.abs().max().item():.3e} | "
            )

        return hypergrad

    def hypergradient_with_sketching(self, expert_trajs, agent_trajs) -> torch.Tensor:
        d_outer_d_policy = self.d_outer_d_policy(expert_trajs)  # (policy_dim,)
        fisher_inv_d_outer_d_policy = self.fisher_solve_sketch(agent_trajs, d_outer_d_policy, self.sketch_size)
        hypergrad = -self.d_inner_d_cross_vec_product(agent_trajs, fisher_inv_d_outer_d_policy)  # (reward_dim,)

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
        hypergradient = self.hypergradient_with_exact_fisher(expert_trajs, agent_trajs)

        self.raw_grad_norm = hypergradient.norm().item()
        if self.max_grad_norm is not None and self.raw_grad_norm > self.max_grad_norm:
            hypergradient = hypergradient * (self.max_grad_norm / self.raw_grad_norm)
        self.clipped_grad_norm = hypergradient.norm().item()

        self.optimizer.zero_grad()
        assign_flat_gradients(self.reward, hypergradient)
        self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()

        self.outer_step += 1

        return hypergradient

    def sweep_sketch_sizes(
        self,
        expert_trajs,
        agent_trajs,
        sketch_sizes,
        compare_hypergradients: bool = False,
    ):
        d_outer_d_policy = self.d_outer_d_policy(expert_trajs)  # (policy_dim,)

        exact_start = time.perf_counter()

        fisher_exact = self.fisher(agent_trajs)
        fisher_end = time.perf_counter()

        v_exact = torch.linalg.solve(
            fisher_exact,
            d_outer_d_policy.to(dtype=fisher_exact.dtype),
        ).to(dtype=torch.float32)

        solve_end = time.perf_counter()

        print(f"\nExact Fisher construction time: " f"{fisher_end - exact_start:>8.2f}s")
        print(f"Exact Fisher solve time:        " f"{solve_end - fisher_end:>8.2f}s")
        print(f"Exact Fisher total time:        " f"{solve_end - exact_start:>8.2f}s")

        hypergrad_exact = None

        if compare_hypergradients:
            hypergrad_exact = -self.d_inner_d_cross_vec_product(
                agent_trajs,
                v_exact,
            )

        results = []

        header = (
            f"{'m':>5} | "
            f"{'solve rel':>9} | "
            f"{'solve cos':>9} | "
            f"{'norm ratio':>10} | "
            f"{'total':>9} | "
            f"{'score':>9} | "
            f"{'extend':>9} | "
            f"{'append':>9} | "
            f"{'update':>9} | "
            f"{'solve':>9} | "
            f"{'app n':>7} | "
            f"{'upd n':>7} | "
            f"{'input blk':>9} | "
            f"{'append blk':>10} | "
            f"{'max app':>7} | "
            f"{'rank':>6} | "
            f"{'alpha':>10}"
        )

        if compare_hypergradients:
            header += f" | {'hyper rel':>9}" f" | {'unit h rel':>10}" f" | {'hyper cos':>9}"

        print("\n" + header)
        print("-" * len(header))

        for sketch_size in sketch_sizes:
            v_sketch, profiling = self.fisher_solve_sketch_profile(
                agent_trajs,
                d_outer_d_policy,
                sketch_size,
            )

            solve_rel_error = relative_error(
                v_sketch,
                v_exact,
            )

            solve_cosine = F.cosine_similarity(
                v_sketch.unsqueeze(0),
                v_exact.unsqueeze(0),
                dim=1,
            ).item()

            norm_ratio = (v_sketch.norm() / (v_exact.norm() + 1e-12)).item()

            result = {
                "sketch_size": sketch_size,
                "solve_relative_error": solve_rel_error,
                "solve_cosine": solve_cosine,
                "solve_norm_ratio": norm_ratio,
                "total_time": profiling["total_time"],
                "score_time": profiling["score_time"],
                "extend_time": profiling["extend_time"],
                "append_time": profiling["append_time"],
                "update_time": profiling["update_time"],
                "solve_time": profiling["solve_time"],
                "append_calls": profiling["append_calls"],
                "update_calls": profiling["update_calls"],
                "solve_calls": profiling["solve_calls"],
                "blocks_seen": profiling["blocks_seen"],
                "rows_seen": profiling["rows_seen"],
                "mean_input_block_size": profiling["mean_input_block_size"],
                "mean_append_block_size": profiling["mean_append_block_size"],
                "max_append_block_size": profiling["max_append_block_size"],
                "final_rank": profiling["final_rank"],
                "final_alpha": profiling["final_alpha"],
            }

            row = (
                f"{sketch_size:>5} | "
                f"{solve_rel_error:>9.4f} | "
                f"{solve_cosine:>9.4f} | "
                f"{norm_ratio:>10.4f} | "
                f"{profiling['total_time']:>8.2f}s | "
                f"{profiling['score_time']:>8.2f}s | "
                f"{profiling['extend_time']:>8.2f}s | "
                f"{profiling['append_time']:>8.2f}s | "
                f"{profiling['update_time']:>8.2f}s | "
                f"{profiling['solve_time']:>8.4f}s | "
                f"{profiling['append_calls']:>7} | "
                f"{profiling['update_calls']:>7} | "
                f"{profiling['mean_input_block_size']:>9.1f} | "
                f"{profiling['mean_append_block_size']:>10.1f} | "
                f"{profiling['max_append_block_size']:>7} | "
                f"{profiling['final_rank']:>6} | "
                f"{profiling['final_alpha']:>10.3e}"
            )

            if compare_hypergradients:
                hypergrad_sketch = -self.d_inner_d_cross_vec_product(
                    agent_trajs,
                    v_sketch,
                )

                hypergrad_rel_error = relative_error(
                    hypergrad_sketch,
                    hypergrad_exact,
                )

                hypergrad_cosine = F.cosine_similarity(
                    hypergrad_sketch.unsqueeze(0),
                    hypergrad_exact.unsqueeze(0),
                    dim=1,
                ).item()

                hypergrad_sketch_unit = hypergrad_sketch / (hypergrad_sketch.norm() + 1e-12)

                hypergrad_exact_unit = hypergrad_exact / (hypergrad_exact.norm() + 1e-12)

                hypergrad_unit_rel_error = relative_error(
                    hypergrad_sketch_unit,
                    hypergrad_exact_unit,
                )

                result.update(
                    {
                        "hypergrad_relative_error": hypergrad_rel_error,
                        "hypergrad_unit_relative_error": hypergrad_unit_rel_error,
                        "hypergrad_cosine": hypergrad_cosine,
                    }
                )

                row += (
                    f" | {hypergrad_rel_error:>9.4f}"
                    f" | {hypergrad_unit_rel_error:>10.4f}"
                    f" | {hypergrad_cosine:>9.4f}"
                )

            print(row)
            results.append(result)

        return results
