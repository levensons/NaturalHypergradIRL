from tqdm import tqdm
import time

import torch
from torch import nn
import torch.nn.functional as F

from src.utils.policies import Policy
from src.algorithms.approximations import SCFD, CBSCFD
from src.utils.torch import flat_grad, num_params, assign_flat_gradients, to_device
from src.utils.trajectories import discount_weights
from src.evaluation.metrics import relative_error, inner_loss, outer_loss


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

    def d_outer_d_policy(self, expert_trajs, verbose: bool = True) -> torch.Tensor:
        policy_dim = num_params(self.policy)
        device = next(self.policy.parameters()).device

        E = torch.zeros(policy_dim, dtype=torch.float32, device=device)

        for traj in tqdm(expert_trajs, desc="Outer grad", leave=False, disable=not verbose):
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
    
    def d_inner_d_policy(self, trajs, verbose: bool = True) -> torch.Tensor:
        policy_params = list(self.policy.parameters())
        policy_dim = num_params(self.policy)
        device = next(self.policy.parameters()).device

        grad_inner = torch.zeros(policy_dim, dtype=torch.float64, device=device)

        for traj in tqdm(trajs, desc="Inner gradient", leave=False, disable=not verbose):
            states = to_device(traj["states"], device)
            actions = to_device(traj["actions"], device)
            T = states.size(0)

            weights = discount_weights(T, self.discount, device, dtype=states.dtype)
            log_probs = self.policy.log_prob(states, actions)
            rewards = self.reward(states, actions).detach()

            trajectory_loss = torch.sum(weights * (self.alpha * log_probs - rewards)).detach()

            score = torch.autograd.grad(
                log_probs.sum(),
                policy_params,
                retain_graph=True,
                create_graph=False,
            )
            score = flat_grad(score).detach().to(torch.float64)

            discounted_score = torch.autograd.grad(
                (weights * log_probs).sum(),
                policy_params,
                retain_graph=False,
                create_graph=False,
            )
            discounted_score = (
                flat_grad(discounted_score)
                .detach()
                .to(torch.float64)
            )

            grad_inner.add_(
                trajectory_loss.to(torch.float64) * score
                + self.alpha * discounted_score
            )

        grad_inner.div_(len(trajs))
        return grad_inner

    def explicit_hessian(self, trajs, verbose: bool = True) -> torch.Tensor:
        "DEPRECATED"
        policy_dim = num_params(self.policy)
        policy_params = list(self.policy.parameters())
        device = next(self.policy.parameters()).device

        H = torch.zeros(policy_dim, policy_dim, dtype=torch.float32, device=device)

        for traj in tqdm(trajs, desc="Explicit Hessian", leave=False, disable=not verbose):
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

    def fisher_solve_sketch_profile(self, trajs, g: torch.Tensor, sketch_size: int, verbose: bool = True) -> torch.Tensor:
        policy_dim = num_params(self.policy)
        device = next(self.policy.parameters()).device

        g = g.to(device=device, dtype=torch.float32)
        sketch = CBSCFD(dim=policy_dim, m=sketch_size, alpha_0=self.fisher_reg)

        score_time = 0.0
        extend_time = 0.0

        n_trajs = len(trajs)
        for traj in tqdm(trajs, desc="Fisher sketch", leave=False, disable=not verbose):
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

    def fisher_solve_sketch(self, trajs, g: torch.Tensor, sketch_size: int, verbose: bool = True) -> torch.Tensor:
        policy_dim = num_params(self.policy)
        device = next(self.policy.parameters()).device

        g = g.to(device=device, dtype=torch.float32)
        sketch = CBSCFD(dim=policy_dim, m=sketch_size, alpha_0=self.fisher_reg)

        n_trajs = len(trajs)
        for traj in tqdm(trajs, desc="Fisher sketch", leave=False, disable=not verbose):
            states = to_device(traj["states"], device)
            actions = to_device(traj["actions"], device)
            T = states.size(0)

            grad_log_pi_a_s = self._grad_log_pi_a_s(states, actions).to(torch.float32)  # (T, policy_dim)
            weights = discount_weights(T, self.discount, device, dtype=torch.float32)  # (T,)

            row_scale = torch.sqrt((self.alpha / n_trajs) * weights)  # (T,)
            X_rows = row_scale.reshape(-1, 1) * grad_log_pi_a_s  # (T, policy_dim)

            sketch.extend(X_rows)

        return sketch.solve(g)

    def fisher(self, trajs, verbose: bool = True) -> torch.Tensor:
        "DEPRECATED"
        policy_dim = num_params(self.policy)
        device = next(self.policy.parameters()).device

        F = torch.zeros(policy_dim, policy_dim, dtype=torch.float64, device=device)

        for traj in tqdm(trajs, desc="Fisher", leave=False, disable=not verbose):
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

    def d_inner_d_cross_vec_product(self, trajs, v: torch.Tensor, verbose: bool = True) -> torch.Tensor:
        "d_inner_d_cross @ v = u"

        reward_dim = num_params(self.reward)
        policy_dim = num_params(self.policy)
        device = next(self.policy.parameters()).device

        v = v.to(dtype=torch.float32, device=device)

        if v.numel() != policy_dim:
            raise ValueError(f"`v` must have shape ({policy_dim},), got {tuple(v.shape)}.")

        U = torch.zeros(reward_dim, dtype=torch.float32, device=device)

        for traj in tqdm(trajs, desc="Cross vec product", leave=False, disable=not verbose):
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
        fisher_exact.diagonal().add_(self.fisher_reg)
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
    
    def compare_hessian_and_fisher(
        self,
        trajs,
        probe_vectors: int = 20,
    ):
        device = next(self.policy.parameters()).device

        # Проверяем предпосылку оптимальности
        inner_grad = self.d_inner_d_policy(trajs)

        # Оцениваем обе матрицы на одних и тех же траекториях
        H = self.explicit_hessian(trajs).to(torch.float64)
        F_mat = self.fisher(trajs).to(torch.float64)

        # На всякий случай ещё раз симметризуем
        H = 0.5 * (H + H.T)
        F_mat = 0.5 * (F_mat + F_mat.T)

        diff = H - F_mat

        h_norm = torch.linalg.matrix_norm(H, ord="fro")
        f_norm = torch.linalg.matrix_norm(F_mat, ord="fro")
        diff_norm = torch.linalg.matrix_norm(diff, ord="fro")

        relative_frobenius = (
            diff_norm / h_norm.clamp_min(1e-12)
        ).item()

        symmetric_relative_frobenius = (
            2.0 * diff_norm
            / (h_norm + f_norm).clamp_min(1e-12)
        ).item()

        # Косинус между матрицами как между векторами
        matrix_cosine = F.cosine_similarity(
            H.reshape(1, -1),
            F_mat.reshape(1, -1),
            dim=1,
        ).item()

        trace_relative_error = (
            torch.abs(torch.trace(H) - torch.trace(F_mat))
            / torch.abs(torch.trace(H)).clamp_min(1e-12)
        ).item()

        # Спектральная ошибка: самое плохо приближённое направление
        relative_spectral = (
            torch.linalg.matrix_norm(diff, ord=2)
            / torch.linalg.matrix_norm(H, ord=2).clamp_min(1e-12)
        ).item()

        # Сравнение квадратичных форм на случайных направлениях
        quadratic_form_errors = []

        for _ in range(probe_vectors):
            z = torch.randn(
                H.shape[0],
                dtype=H.dtype,
                device=device,
            )
            z /= z.norm().clamp_min(1e-12)

            q_h = z @ H @ z
            q_f = z @ F_mat @ z

            error = torch.abs(q_h - q_f) / torch.maximum(
                torch.maximum(q_h.abs(), q_f.abs()),
                torch.tensor(1e-12, dtype=H.dtype, device=device),
            )

            quadratic_form_errors.append(error)

        quadratic_form_errors = torch.stack(quadratic_form_errors)

        results = {
            "inner_grad_norm": inner_grad.norm().item(),
            "inner_grad_rms": (
                inner_grad.norm() / inner_grad.numel() ** 0.5
            ).item(),
            "inner_grad_abs_max": inner_grad.abs().max().item(),

            "hessian_frobenius_norm": h_norm.item(),
            "fisher_frobenius_norm": f_norm.item(),
            "difference_frobenius_norm": diff_norm.item(),

            "relative_frobenius_error": relative_frobenius,
            "symmetric_relative_frobenius_error":
                symmetric_relative_frobenius,
            "relative_spectral_error": relative_spectral,
            "matrix_cosine": matrix_cosine,
            "trace_relative_error": trace_relative_error,

            "quadratic_error_mean":
                quadratic_form_errors.mean().item(),
            "quadratic_error_max":
                quadratic_form_errors.max().item(),
        }

        print("\nHessian–Fisher comparison")
        print("-" * 54)
        print(
            f"Inner grad norm:          "
            f"{results['inner_grad_norm']:.4e}"
        )
        print(
            f"Inner grad RMS:           "
            f"{results['inner_grad_rms']:.4e}"
        )
        print(
            f"||H||_F:                  "
            f"{results['hessian_frobenius_norm']:.4e}"
        )
        print(
            f"||F||_F:                  "
            f"{results['fisher_frobenius_norm']:.4e}"
        )
        print(
            f"||H-F||_F / ||H||_F:      "
            f"{results['relative_frobenius_error']:.4e}"
        )
        print(
            f"Symmetric relative error: "
            f"{results['symmetric_relative_frobenius_error']:.4e}"
        )
        print(
            f"Relative spectral error:  "
            f"{results['relative_spectral_error']:.4e}"
        )
        print(
            f"Matrix cosine:            "
            f"{results['matrix_cosine']:.6f}"
        )
        print(
            f"Trace relative error:     "
            f"{results['trace_relative_error']:.4e}"
        )
        print(
            f"Quadratic error mean:     "
            f"{results['quadratic_error_mean']:.4e}"
        )
        print(
            f"Quadratic error max:      "
            f"{results['quadratic_error_max']:.4e}"
        )

        return results

    def sweep_n_agent_trajs(
        self,
        expert_trajs,
        agent_trajs,
        n_prefixes: int = 25,
    ):
        if len(agent_trajs) == 0:
            raise ValueError("agent_trajs must contain at least one trajectory.")

        if n_prefixes <= 0:
            raise ValueError("n_prefixes must be positive.")

        n_agent_trajs = len(agent_trajs)
        n_prefixes = min(n_prefixes, n_agent_trajs)

        # prefixes = (
        #     torch.linspace(
        #         1,
        #         n_agent_trajs,
        #         steps=n_prefixes,
        #     )
        #     .round()
        #     .to(torch.int64)
        #     .unique()
        #     .tolist()
        # )

        prefixes = [1, 2, 5, 10, 25, 50, 100, 250, 300, 400, 500, 750, 800, 900, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000]

        # Общие величины, не зависящие от размера префикса
        l_outer = outer_loss(
            policy=self.policy,
            expert_trajs=expert_trajs,
        )

        policy_vector = torch.nn.utils.parameters_to_vector(
            self.policy.parameters()
        ).detach()

        policy_norm = torch.linalg.vector_norm(
            policy_vector
        ).clamp_min(1e-12)

        # Reference строим по всем доступным агентским траекториям
        reference_trajs = agent_trajs

        fisher_ref = self.fisher(
            reference_trajs,
        )

        fisher_ref_norm = torch.linalg.matrix_norm(
            fisher_ref,
            ord="fro",
        ).clamp_min(1e-12)

        d_outer = self.d_outer_d_policy(
            expert_trajs
        ).to(
            device=fisher_ref.device,
            dtype=fisher_ref.dtype,
        )

        fisher_ref_reg = fisher_ref.clone()
        fisher_ref_reg.diagonal().add_(self.fisher_reg)

        v_ref = torch.linalg.solve(
            fisher_ref_reg,
            d_outer,
        )

        v_ref_norm = torch.linalg.vector_norm(
            v_ref
        ).clamp_min(1e-12)

        header = (
            f"{'N':>6} | "
            f"{'L_inner':>10} | "
            f"{'L_outer':>10} | "
            f"{'||g||':>10} | "
            f"{'g RMS':>9} | "
            f"{'g rel':>9} | "
            f"{'|g|max':>10} | "
            f"{'F rel':>8} | "
            f"{'F cos':>8} | "
            f"{'solve rel':>10} | "
            f"{'solve cos':>9}"
        )

        print("\nFisher sample-size sweep")
        print("=" * len(header))
        print(header)
        print("-" * len(header))

        results = []

        for prefix in prefixes:
            cur_trajs = agent_trajs[:prefix]

            l_inner = inner_loss(
                policy=self.policy,
                reward=self.reward,
                trajs=cur_trajs,
                discount=self.discount,
                alpha=self.alpha,
            )

            inner_grad = self.d_inner_d_policy(
                cur_trajs,
            )

            grad_norm = torch.linalg.vector_norm(inner_grad)
            grad_rms = grad_norm / inner_grad.numel() ** 0.5
            grad_abs_max = inner_grad.abs().max()
            grad_relative = grad_norm / policy_norm

            # Fisher без регуляризации:
            # сравниваем именно оценки матрицы.
            fisher_cur = self.fisher(
                cur_trajs,
            )

            fisher_rel_error = (
                torch.linalg.matrix_norm(
                    fisher_cur - fisher_ref,
                    ord="fro",
                )
                / fisher_ref_norm
            )

            fisher_cosine = torch.nn.functional.cosine_similarity(
                fisher_cur.flatten(),
                fisher_ref.flatten(),
                dim=0,
            )

            # Fisher с регуляризацией:
            # сравниваем реально используемое решение системы.
            fisher_cur_reg = fisher_cur.clone()
            fisher_cur_reg.diagonal().add_(self.fisher_reg)

            v_cur = torch.linalg.solve(
                fisher_cur_reg,
                d_outer,
            )

            solve_rel_error = (
                torch.linalg.vector_norm(v_cur - v_ref)
                / v_ref_norm
            )

            solve_cosine = torch.nn.functional.cosine_similarity(
                v_cur,
                v_ref,
                dim=0,
            )

            row = {
                "n_trajs": prefix,
                "l_inner": float(l_inner),
                "l_outer": float(l_outer),
                "grad_norm": grad_norm.item(),
                "grad_rms": grad_rms.item(),
                "grad_relative": grad_relative.item(),
                "grad_abs_max": grad_abs_max.item(),
                "fisher_relative_error": fisher_rel_error.item(),
                "fisher_cosine": fisher_cosine.item(),
                "solve_relative_error": solve_rel_error.item(),
                "solve_cosine": solve_cosine.item(),
            }
            results.append(row)

            print(
                f"{prefix:6d} | "
                f"{row['l_inner']:10.3f} | "
                f"{row['l_outer']:10.3f} | "
                f"{row['grad_norm']:10.3e} | "
                f"{row['grad_rms']:9.3e} | "
                f"{row['grad_relative']:9.3e} | "
                f"{row['grad_abs_max']:10.3e} | "
                f"{row['fisher_relative_error']:8.4f} | "
                f"{row['fisher_cosine']:8.4f} | "
                f"{row['solve_relative_error']:10.4f} | "
                f"{row['solve_cosine']:9.4f}"
            )

        print("=" * len(header))

        return results