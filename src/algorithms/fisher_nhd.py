from tqdm import tqdm
import time

import torch
from torch import nn
import torch.nn.functional as F

from src.utils.policies import Policy
from src.algorithms.approximations import CBSCFD
from src.utils.torch import flat_grad, num_params, assign_flat_gradients, to_device
from src.utils.trajectories import discount_weights
from src.evaluation.metrics import relative_error, inner_loss, outer_loss

import math
from collections.abc import Sequence

def _safe_cosine(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    x = x.reshape(-1)
    y = y.reshape(-1)

    x_norm = torch.linalg.vector_norm(x)
    y_norm = torch.linalg.vector_norm(y)

    denominator = x_norm * y_norm

    if denominator.item() <= eps:
        return float("nan")

    return torch.dot(x, y).div(denominator).item()


def _symmetric_relative_difference(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    """
    Симметричная относительная разница:

        2 ||x - y|| / (||x|| + ||y||).

    В отличие от ||x-y||/||y||, ни один набор не считается reference.
    """
    x = x.reshape(-1)
    y = y.reshape(-1)

    numerator = 2.0 * torch.linalg.vector_norm(x - y)
    denominator = (
        torch.linalg.vector_norm(x)
        + torch.linalg.vector_norm(y)
    ).clamp_min(eps)

    return numerator.div(denominator).item()


def _norm_ratio(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    """
    Возвращает ||x|| / ||y||.
    """
    x_norm = torch.linalg.vector_norm(x)
    y_norm = torch.linalg.vector_norm(y).clamp_min(eps)

    return x_norm.div(y_norm).item()


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
        sketch_size: int = 32,
    ):
        self.reward = reward
        self.policy = policy
        self.fisher_reg = fisher_reg
        self.gamma = gamma
        self.alpha = alpha
        self.max_grad_norm = max_grad_norm
        self.scheduler_gamma = scheduler_gamma
        self.sketch_size = sketch_size

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
        policy_dim = num_params(self.policy)

        out = torch.zeros(policy_dim, dtype=torch.float32, device=self.device)

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

            out.add_(grad)

        out.div_(len(expert_trajs))
        out.neg_()
        return out

    def _grad_R_tail_with_discount(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        T = states.size(0)

        r_a_s_t = self.reward(states, actions)  # (T,)
        weights = discount_weights(T, self.gamma, self.device, r_a_s_t.dtype)  # (T,)
        r_a_s_t = weights * r_a_s_t

        grad_outputs = torch.eye(T, dtype=r_a_s_t.dtype, device=self.device)  # (T, T)

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

    def fisher_solve_sketch(self, trajs, g: torch.Tensor, sketch_size: int, batch_size: int = 128, verbose: bool = True) -> torch.Tensor:
        sketch = CBSCFD(dim=self.policy_num_params, m=sketch_size, reg=self.fisher_reg)
        n_trajs = len(trajs)

        for traj in tqdm(trajs, desc="Fisher sketch", leave=False, disable=not verbose):
            states = to_device(traj["states"], self.device)
            actions = to_device(traj["actions"], self.device)
            T = states.size(0)
            weights = discount_weights(T, self.gamma, self.device, torch.float32)  # (T,)

            for start in range(0, T, batch_size):
                batch_states = states[start:start + batch_size]
                batch_actions = actions[start:start + batch_size]
                batch_weights = weights[start:start + batch_size]

                grad_log_pi_a_s = self._grad_log_pi_a_s(batch_states, batch_actions)  # (B, policy_dim)

                row_scale = torch.sqrt((self.alpha / n_trajs) * batch_weights)  # (B,)
                X_rows = row_scale.reshape(-1, 1) * grad_log_pi_a_s  # (B, policy_dim)

                sketch.extend(X_rows)

        return sketch.solve(g)

    def exact_fisher(self, trajs, batch_size: int = 128, verbose: bool = True) -> torch.Tensor:
        F = torch.zeros(self.policy_num_params, self.policy_num_params, dtype=torch.float32, device=self.device)

        for traj in tqdm(trajs, desc="Fisher", leave=False, disable=not verbose):
            states = to_device(traj["states"], self.device)
            actions = to_device(traj["actions"], self.device)
            T = states.size(0)
            weights = discount_weights(T, self.gamma, self.device, torch.float32)  # (T,)

            for start in range(0, T, batch_size):
                batch_states = states[start:start + batch_size] # (B, state_dim)
                batch_actions = actions[start:start + batch_size] # (B, action_dim)
                batch_weights = weights[start:start + batch_size] # (B,)

                grad_log_pi_a_s = self._grad_log_pi_a_s(batch_states, batch_actions)  # (B, policy_dim)
                F.addmm_(grad_log_pi_a_s.T, grad_log_pi_a_s * batch_weights.reshape(-1, 1)) # (policy_dim, policy_dim)

        F.div_(len(trajs))
        F = 0.5 * (F + F.T)
        F.mul_(self.alpha)
        return F

    # def d_inner_d_cross_vec_product(self, trajs, v: torch.Tensor, verbose: bool = True) -> torch.Tensor:
    #     """
    #     Compute (∇²_{θφ} L_inner)^T v.
    #     """

    #     if v.numel() != self.policy_num_params:
    #         raise ValueError(f"`v` must have shape ({self.policy_num_params},), got {tuple(v.shape)}.")

    #     out = torch.zeros(self.reward_num_params, dtype=torch.float32, device=self.device)

    #     for traj in tqdm(trajs, desc="Cross vec product", leave=False, disable=not verbose):
    #         states = to_device(traj["states"], self.device)
    #         actions = to_device(traj["actions"], self.device)

    #         grad_R_tail_with_discount = self._grad_R_tail_with_discount(states, actions)  # (T, reward_dim)

    #         # using JVP
    #         jv = self._jvp_grad_log_pi_a_s(states, actions, v)  # (T,)
    #         out.add_(torch.einsum("tr,t->r", grad_R_tail_with_discount, jv))

    #     out.div_(len(trajs))
    #     out.neg_()
    #     return out

    def d_inner_d_cross_vec_product(self, trajs, v: torch.Tensor, verbose: bool = True) -> torch.Tensor:
        """
            Compute (∇²_{θφ} L_inner)^T v,
            using reverse derivation.
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
    
    def hypergradient_with_exact_fisher(self, expert_trajs, agent_trajs) -> torch.Tensor:
        d_outer_d_policy = self.d_outer_d_policy(expert_trajs)  # g

        fisher = self.exact_fisher(agent_trajs)
        fisher.diagonal().add_(self.fisher_reg) # F + lambda * I

        fisher_inv_d_outer_d_policy = torch.linalg.solve(fisher, d_outer_d_policy)  # (F + lambda * I)^(-1) @ g

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
        hypergradient = self.hypergradient_with_sketching(expert_trajs, agent_trajs)

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

    def sweep_sketch_sizes(
        self,
        expert_trajs,
        agent_trajs,
        sketch_sizes,
        compare_hypergradients: bool = False,
    ):
        d_outer_d_policy = self.d_outer_d_policy(expert_trajs)

        # Exact Fisher
        exact_start = time.perf_counter()

        fisher_exact = self.exact_fisher(agent_trajs)
        fisher_exact.diagonal().add_(self.fisher_reg)

        fisher_end = time.perf_counter()

        exact_g = d_outer_d_policy.to(
            device=fisher_exact.device,
            dtype=fisher_exact.dtype,
        )
        v_exact = torch.linalg.solve(fisher_exact, exact_g)

        solve_end = time.perf_counter()

        print(f"\nExact Fisher construction time: {fisher_end - exact_start:>8.2f}s")
        print(f"Exact Fisher solve time:        {solve_end - fisher_end:>8.2f}s")
        print(f"Exact Fisher total time:        {solve_end - exact_start:>8.2f}s")

        hypergrad_exact = None

        if compare_hypergradients:
            hypergrad_exact = -self.d_inner_d_cross_vec_product(
                agent_trajs,
                v_exact,
            )

        header = (
            f"{'m':>6} | "
            f"{'solve rel':>10} | "
            f"{'solve cos':>10} | "
            f"{'norm ratio':>10} | "
            f"{'time':>9}"
        )

        if compare_hypergradients:
            header += (
                f" | {'hyper rel':>10}"
                f" | {'unit h rel':>10}"
                f" | {'hyper cos':>10}"
            )

        print("\n" + header)
        print("-" * len(header))

        results = []

        for sketch_size in sketch_sizes:
            sketch_start = time.perf_counter()

            v_sketch = self.fisher_solve_sketch(
                agent_trajs,
                d_outer_d_policy,
                sketch_size,
                verbose=True,
            )

            sketch_time = time.perf_counter() - sketch_start

            v_exact_local = v_exact.to(
                device=v_sketch.device,
                dtype=v_sketch.dtype,
            )

            solve_rel_error = relative_error(
                v_sketch,
                v_exact_local,
            )

            solve_cosine = F.cosine_similarity(
                v_sketch,
                v_exact_local,
                dim=0,
            ).item()

            norm_ratio = (
                v_sketch.norm()
                / v_exact_local.norm().clamp_min(1e-12)
            ).item()

            result = {
                "sketch_size": sketch_size,
                "solve_relative_error": solve_rel_error,
                "solve_cosine": solve_cosine,
                "solve_norm_ratio": norm_ratio,
                "time": sketch_time,
            }

            row = (
                f"{sketch_size:>6} | "
                f"{solve_rel_error:>10.4f} | "
                f"{solve_cosine:>10.4f} | "
                f"{norm_ratio:>10.4f} | "
                f"{sketch_time:>8.2f}s"
            )

            if compare_hypergradients:
                hypergrad_sketch = -self.d_inner_d_cross_vec_product(
                    agent_trajs,
                    v_sketch,
                )

                hypergrad_exact_local = hypergrad_exact.to(
                    device=hypergrad_sketch.device,
                    dtype=hypergrad_sketch.dtype,
                )

                hypergrad_rel_error = relative_error(
                    hypergrad_sketch,
                    hypergrad_exact_local,
                )

                hypergrad_cosine = F.cosine_similarity(
                    hypergrad_sketch,
                    hypergrad_exact_local,
                    dim=0,
                ).item()

                hypergrad_sketch_unit = (
                    hypergrad_sketch
                    / hypergrad_sketch.norm().clamp_min(1e-12)
                )
                hypergrad_exact_unit = (
                    hypergrad_exact_local
                    / hypergrad_exact_local.norm().clamp_min(1e-12)
                )

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
                    f" | {hypergrad_rel_error:>10.4f}"
                    f" | {hypergrad_unit_rel_error:>10.4f}"
                    f" | {hypergrad_cosine:>10.4f}"
                )

            print(row)
            results.append(result)

        return results

    def sweep_n_agent_trajs(
        self,
        expert_trajs,
        agent_trajs,
        prefixes: Sequence[int] | None = None,
        *,
        shuffle_seed: int = 42,
        verbose: bool = False,
    ):
        """
        Проверяет, как оценки inner gradient, Fisher solve и hypergradient
        меняются с ростом числа agent trajectories.

        Все параметры policy и reward должны быть заморожены на время sweep.
        Последний доступный размер выборки используется как reference.

        Важно:
            reference и меньшие выборки являются вложенными, поэтому это
            convergence sweep, а не независимая оценка дисперсии.
        """
        if len(agent_trajs) == 0:
            raise ValueError("agent_trajs must contain at least one trajectory.")

        if len(expert_trajs) == 0:
            raise ValueError("expert_trajs must contain at least one trajectory.")

        n_total = len(agent_trajs)

        if prefixes is None:
            default_prefixes = [
                1,
                2,
                5,
                10,
                25,
                50,
                100,
                250,
                500,
                750,
                1000,
                1500,
                2000,
                3000,
                5000,
                7500,
                10000,
            ]

            prefixes = [n for n in default_prefixes if n <= n_total]

            if n_total not in prefixes:
                prefixes.append(n_total)
        else:
            prefixes = sorted({
                int(n)
                for n in prefixes
                if 1 <= int(n) <= n_total
            })

            if not prefixes:
                raise ValueError(
                    "prefixes must contain at least one value in "
                    f"[1, {n_total}]."
                )

            if prefixes[-1] != n_total:
                prefixes.append(n_total)

        # Один раз перемешиваем, после чего берём вложенные префиксы.
        # Это устраняет зависимость результата от исходного порядка trajectories.
        generator = torch.Generator()
        generator.manual_seed(shuffle_seed)

        permutation = torch.randperm(
            n_total,
            generator=generator,
        ).tolist()

        shuffled_agent_trajs = [
            agent_trajs[i]
            for i in permutation
        ]

        policy_params = tuple(self.policy.parameters())
        reward_params = tuple(self.reward.parameters())

        if not policy_params:
            raise ValueError("Policy has no trainable parameters.")

        if not reward_params:
            raise ValueError("Reward model has no trainable parameters.")

        policy_device = policy_params[0].device
        policy_dtype = policy_params[0].dtype
        policy_dim = sum(p.numel() for p in policy_params)
        reward_dim = sum(p.numel() for p in reward_params)

        # Outer objective и его gradient не зависят от agent sample size.
        l_outer = outer_loss(
            self.policy,
            expert_trajs,
            self.gamma,
        )

        d_outer = self.d_outer_d_policy(
            expert_trajs,
            verbose=verbose,
        ).to(
            device=policy_device,
            dtype=policy_dtype,
        )

        d_outer_norm = torch.linalg.vector_norm(
            d_outer,
        ).clamp_min(1e-12)

        # Reference — максимально доступная выборка.
        reference_trajs = shuffled_agent_trajs[:prefixes[-1]]

        fisher_ref = self.exact_fisher(
            reference_trajs,
            verbose=verbose
        )

        d_outer_fisher = d_outer.to(
            device=fisher_ref.device,
            dtype=fisher_ref.dtype,
        )

        fisher_ref_norm = torch.linalg.matrix_norm(
            fisher_ref,
            ord="fro",
        ).clamp_min(1e-12)

        fisher_ref_trace = torch.trace(
            fisher_ref,
        )

        fisher_ref_diag_mean = (
            fisher_ref.diagonal().mean()
        )

        fisher_ref_reg = fisher_ref.clone()
        fisher_ref_reg.diagonal().add_(self.fisher_reg)

        v_ref = torch.linalg.solve(
            fisher_ref_reg,
            d_outer_fisher,
        )

        v_ref_norm = torch.linalg.vector_norm(
            v_ref,
        ).clamp_min(1e-12)

        # Полезно сравнивать и итоговый hypergradient.
        #
        # Здесь предполагается, что метод возвращает
        #   (∇²_{policy,reward} L_inner)^T v
        # уже с тем знаком, который принят в твоей реализации.
        hypergrad_ref = self.d_inner_d_cross_vec_product(
            reference_trajs,
            v_ref,
            verbose=verbose
        )

        # Если d_inner_d_cross_vec_product возвращает mixed product без
        # внешнего минуса, поставь здесь тот же neg(), что используется
        # в основном compute_hypergradient().
        hypergrad_ref_norm = torch.linalg.vector_norm(
            hypergrad_ref,
        ).clamp_min(1e-12)

        header = (
            f"{'N':>6} | "
            f"{'L_inner':>10} | "
            f"{'||g_in||':>10} | "
            f"{'g RMS':>9} | "
            f"{'|g|max':>9} | "
            f"{'F rel':>8} | "
            f"{'F cos':>8} | "
            f"{'||v||/||go||':>12} | "
            f"{'v rel':>8} | "
            f"{'v cos':>8} | "
            f"{'resid':>8} | "
            f"{'h rel':>8} | "
            f"{'h cos':>8}"
        )

        print("\nFisher sample-size sweep")
        print(f"Reference sample size: {len(reference_trajs)}")
        print(
            "Reference Fisher: "
            f"trace={fisher_ref_trace.item():.3e}, "
            f"diag_mean={fisher_ref_diag_mean.item():.3e}, "
            f"damping={self.fisher_reg:.3e}"
        )
        print(
            "Outer gradient: "
            f"norm={d_outer_norm.item():.3e}, "
            f"L_outer={float(l_outer):.6f}"
        )
        print("=" * len(header))
        print(header)
        print("-" * len(header))

        results = []

        for prefix in prefixes:
            cur_trajs = shuffled_agent_trajs[:prefix]

            l_inner = inner_loss(
                self.policy,
                self.reward,
                cur_trajs,
                self.gamma,
                self.alpha,
            )

            inner_grad = self.d_inner_d_policy(
                cur_trajs,
                verbose=verbose,
            )

            grad_norm = torch.linalg.vector_norm(inner_grad)
            grad_rms = grad_norm / math.sqrt(policy_dim)
            grad_abs_max = inner_grad.abs().max()

            fisher_cur = self.exact_fisher(
                cur_trajs,
                verbose=verbose
            )

            if fisher_cur.shape != fisher_ref.shape:
                raise RuntimeError(
                    "Current and reference Fisher matrices have "
                    f"different shapes: {fisher_cur.shape} and "
                    f"{fisher_ref.shape}."
                )

            fisher_rel_error = (
                torch.linalg.matrix_norm(
                    fisher_cur - fisher_ref,
                    ord="fro",
                )
                / fisher_ref_norm
            )

            fisher_cosine = _safe_cosine(
                fisher_cur.reshape(-1),
                fisher_ref.reshape(-1),
            )

            fisher_cur_reg = fisher_cur.clone()
            fisher_cur_reg.diagonal().add_(self.fisher_reg)

            v_cur = torch.linalg.solve(
                fisher_cur_reg,
                d_outer_fisher,
            )

            v_cur_norm = torch.linalg.vector_norm(v_cur)

            solve_amplification = (
                v_cur_norm / d_outer_norm
            )

            solve_rel_error = (
                torch.linalg.vector_norm(v_cur - v_ref)
                / v_ref_norm
            )

            solve_cosine = _safe_cosine(
                v_cur,
                v_ref,
            )

            # Проверяем численную точность solve.
            residual = (
                torch.linalg.vector_norm(
                    fisher_cur_reg @ v_cur - d_outer_fisher
                )
                / d_outer_norm
            )

            hypergrad_cur = self.d_inner_d_cross_vec_product(
                cur_trajs,
                v_cur,
                verbose=verbose
            )

            hypergrad_rel_error = (
                torch.linalg.vector_norm(
                    hypergrad_cur - hypergrad_ref
                )
                / hypergrad_ref_norm
            )

            hypergrad_cosine = _safe_cosine(
                hypergrad_cur,
                hypergrad_ref,
            )

            row = {
                "n_trajs": prefix,
                "l_inner": float(l_inner),
                "l_outer": float(l_outer),

                "inner_grad_norm": grad_norm.item(),
                "inner_grad_rms": grad_rms.item(),
                "inner_grad_abs_max": grad_abs_max.item(),

                "fisher_relative_error": fisher_rel_error.item(),
                "fisher_cosine": fisher_cosine,
                "fisher_trace": torch.trace(fisher_cur).item(),
                "fisher_diag_mean": (
                    fisher_cur.diagonal().mean().item()
                ),

                "solve_norm": v_cur_norm.item(),
                "solve_amplification": solve_amplification.item(),
                "solve_relative_error": solve_rel_error.item(),
                "solve_cosine": solve_cosine,
                "solve_residual": residual.item(),

                "hypergrad_norm": (
                    torch.linalg.vector_norm(
                        hypergrad_cur
                    ).item()
                ),
                "hypergrad_rms": (
                    torch.linalg.vector_norm(hypergrad_cur).item()
                    / math.sqrt(reward_dim)
                ),
                "hypergrad_relative_error": (
                    hypergrad_rel_error.item()
                ),
                "hypergrad_cosine": hypergrad_cosine,
            }

            results.append(row)

            print(
                f"{prefix:6d} | "
                f"{row['l_inner']:10.3f} | "
                f"{row['inner_grad_norm']:10.3e} | "
                f"{row['inner_grad_rms']:9.3e} | "
                f"{row['inner_grad_abs_max']:9.3e} | "
                f"{row['fisher_relative_error']:8.4f} | "
                f"{row['fisher_cosine']:8.4f} | "
                f"{row['solve_amplification']:12.3e} | "
                f"{row['solve_relative_error']:8.4f} | "
                f"{row['solve_cosine']:8.4f} | "
                f"{row['solve_residual']:8.2e} | "
                f"{row['hypergrad_relative_error']:8.4f} | "
                f"{row['hypergrad_cosine']:8.4f}"
            )

        print("=" * len(header))

        return results
    
    def compare_agent_trajectory_sets(
        self,
        expert_trajs,
        agent_trajs_1,
        agent_trajs_2,
        *,
        verbose: bool = False,
    ):
        """
        Сравнивает две независимые Monte Carlo-оценки, полученные
        на двух наборах agent trajectories.

        Policy и reward должны быть полностью фиксированы.

        Оба набора используют:
            - один и тот же outer gradient по expert trajectories;
            - одинаковый damping;
            - одинаковые параметры policy и reward.

        Сравниваются:
            1. inner objective;
            2. inner gradient;
            3. empirical Fisher;
            4. v = (F + lambda I)^(-1) g_outer;
            5. hypergradient = C^T v
            с тем знаком, который возвращает
            d_inner_d_cross_vec_product().
        """
        if len(expert_trajs) == 0:
            raise ValueError(
                "expert_trajs must contain at least one trajectory."
            )

        if len(agent_trajs_1) == 0:
            raise ValueError(
                "agent_trajs_1 must contain at least one trajectory."
            )

        if len(agent_trajs_2) == 0:
            raise ValueError(
                "agent_trajs_2 must contain at least one trajectory."
            )

        policy_params = tuple(self.policy.parameters())
        reward_params = tuple(self.reward.parameters())

        if not policy_params:
            raise ValueError("Policy has no trainable parameters.")

        if not reward_params:
            raise ValueError("Reward model has no trainable parameters.")

        policy_device = policy_params[0].device
        policy_dtype = policy_params[0].dtype

        policy_dim = sum(
            parameter.numel()
            for parameter in policy_params
        )

        reward_dim = sum(
            parameter.numel()
            for parameter in reward_params
        )

        # ---------------------------------------------------------
        # Outer part is shared by both trajectory sets.
        # ---------------------------------------------------------

        l_outer = outer_loss(
            self.policy,
            expert_trajs,
            self.gamma,
        )

        d_outer = self.d_outer_d_policy(
            expert_trajs,
            verbose=verbose,
        ).to(
            device=policy_device,
            dtype=policy_dtype,
        )

        d_outer_norm = torch.linalg.vector_norm(
            d_outer
        ).clamp_min(1e-12)

        # ---------------------------------------------------------
        # Helper that computes all quantities for one trajectory set.
        # ---------------------------------------------------------

        def compute_statistics(agent_trajs):
            l_inner = inner_loss(
                self.policy,
                self.reward,
                agent_trajs,
                self.gamma,
                self.alpha,
            )

            inner_grad = self.d_inner_d_policy(
                agent_trajs,
                verbose=verbose,
            ).to(
                device=policy_device,
                dtype=policy_dtype,
            )

            fisher = self.exact_fisher(
                agent_trajs,
                verbose=verbose,
            )

            d_outer_fisher = d_outer.to(
                device=fisher.device,
                dtype=fisher.dtype,
            )

            fisher_reg = fisher.clone()
            fisher_reg.diagonal().add_(self.fisher_reg)

            solve = torch.linalg.solve(
                fisher_reg,
                d_outer_fisher,
            )

            solve_residual = (
                torch.linalg.vector_norm(
                    fisher_reg @ solve - d_outer_fisher
                )
                / torch.linalg.vector_norm(
                    d_outer_fisher
                ).clamp_min(1e-12)
            )

            hypergrad = self.d_inner_d_cross_vec_product(
                agent_trajs,
                solve,
                verbose=verbose,
            )

            # ВАЖНО:
            # Если основной compute_hypergradient() делает:
            #
            #     hypergrad.neg_()
            #
            # то для сравнения cosine между двумя выборками это ничего
            # не меняет, потому что минус применяется к обоим векторам.
            #
            # Но для полного соответствия основному алгоритму можешь
            # применить тот же знак и здесь:
            #
            # hypergrad = -hypergrad

            return {
                "n_trajs": len(agent_trajs),
                "l_inner": float(l_inner),

                "inner_grad": inner_grad,
                "inner_grad_norm": torch.linalg.vector_norm(
                    inner_grad
                ).item(),
                "inner_grad_rms": (
                    torch.linalg.vector_norm(inner_grad).item()
                    / math.sqrt(policy_dim)
                ),
                "inner_grad_abs_max": (
                    inner_grad.abs().max().item()
                ),

                "fisher": fisher,
                "fisher_norm": torch.linalg.matrix_norm(
                    fisher,
                    ord="fro",
                ).item(),
                "fisher_trace": torch.trace(fisher).item(),
                "fisher_diag_mean": (
                    fisher.diagonal().mean().item()
                ),
                "fisher_diag_min": (
                    fisher.diagonal().min().item()
                ),
                "fisher_diag_max": (
                    fisher.diagonal().max().item()
                ),

                "solve": solve,
                "solve_norm": torch.linalg.vector_norm(
                    solve
                ).item(),
                "solve_rms": (
                    torch.linalg.vector_norm(solve).item()
                    / math.sqrt(policy_dim)
                ),
                "solve_abs_max": (
                    solve.abs().max().item()
                ),
                "solve_amplification": (
                    torch.linalg.vector_norm(solve)
                    / d_outer_norm
                ).item(),
                "solve_residual": solve_residual.item(),

                "hypergrad": hypergrad,
                "hypergrad_norm": torch.linalg.vector_norm(
                    hypergrad
                ).item(),
                "hypergrad_rms": (
                    torch.linalg.vector_norm(hypergrad).item()
                    / math.sqrt(reward_dim)
                ),
                "hypergrad_abs_max": (
                    hypergrad.abs().max().item()
                ),
            }

        stats_1 = compute_statistics(agent_trajs_1)
        stats_2 = compute_statistics(agent_trajs_2)

        solve_1 = stats_1["solve"]
        solve_2 = stats_2["solve"]

        hypergrad_11 = self.d_inner_d_cross_vec_product(
            agent_trajs_1,
            solve_1,
            verbose=verbose,
        )

        hypergrad_12 = self.d_inner_d_cross_vec_product(
            agent_trajs_1,
            solve_2,
            verbose=verbose,
        )

        hypergrad_21 = self.d_inner_d_cross_vec_product(
            agent_trajs_2,
            solve_1,
            verbose=verbose,
        )

        hypergrad_22 = self.d_inner_d_cross_vec_product(
            agent_trajs_2,
            solve_2,
            verbose=verbose,
        )

        decomposition = {
            # Меняем solve, оставляя trajectory set фиксированным.
            "solve_effect_set_1_cosine": _safe_cosine(
                hypergrad_11,
                hypergrad_12,
            ),
            "solve_effect_set_2_cosine": _safe_cosine(
                hypergrad_21,
                hypergrad_22,
            ),

            # Меняем trajectory set, оставляя solve фиксированным.
            "cross_effect_solve_1_cosine": _safe_cosine(
                hypergrad_11,
                hypergrad_21,
            ),
            "cross_effect_solve_2_cosine": _safe_cosine(
                hypergrad_12,
                hypergrad_22,
            ),

            "solve_effect_set_1_sym_rel":
                _symmetric_relative_difference(
                    hypergrad_11,
                    hypergrad_12,
                ),
            "solve_effect_set_2_sym_rel":
                _symmetric_relative_difference(
                    hypergrad_21,
                    hypergrad_22,
                ),

            "cross_effect_solve_1_sym_rel":
                _symmetric_relative_difference(
                    hypergrad_11,
                    hypergrad_21,
                ),
            "cross_effect_solve_2_sym_rel":
                _symmetric_relative_difference(
                    hypergrad_12,
                    hypergrad_22,
                ),
        }

        print("\nHypergradient variability decomposition")

        print(
            "Change solve, fixed set | "
            f"set 1 cosine={decomposition['solve_effect_set_1_cosine']:.4f} | "
            f"set 2 cosine={decomposition['solve_effect_set_2_cosine']:.4f}"
        )

        print(
            "Change set, fixed solve | "
            f"solve 1 cosine={decomposition['cross_effect_solve_1_cosine']:.4f} | "
            f"solve 2 cosine={decomposition['cross_effect_solve_2_cosine']:.4f}"
        )

        print(
            "Change solve, fixed set | "
            f"set 1 sym rel={decomposition['solve_effect_set_1_sym_rel']:.4f} | "
            f"set 2 sym rel={decomposition['solve_effect_set_2_sym_rel']:.4f}"
        )

        print(
            "Change set, fixed solve | "
            f"solve 1 sym rel={decomposition['cross_effect_solve_1_sym_rel']:.4f} | "
            f"solve 2 sym rel={decomposition['cross_effect_solve_2_sym_rel']:.4f}"
        )

        if stats_1["fisher"].shape != stats_2["fisher"].shape:
            raise RuntimeError(
                "Fisher matrices have different shapes: "
                f"{stats_1['fisher'].shape} and "
                f"{stats_2['fisher'].shape}."
            )

        if stats_1["inner_grad"].shape != stats_2["inner_grad"].shape:
            raise RuntimeError(
                "Inner gradients have different shapes: "
                f"{stats_1['inner_grad'].shape} and "
                f"{stats_2['inner_grad'].shape}."
            )

        if stats_1["hypergrad"].shape != stats_2["hypergrad"].shape:
            raise RuntimeError(
                "Hypergradients have different shapes: "
                f"{stats_1['hypergrad'].shape} and "
                f"{stats_2['hypergrad'].shape}."
            )

        # ---------------------------------------------------------
        # Pairwise comparison.
        # ---------------------------------------------------------

        comparison = {
            "n_trajs_1": stats_1["n_trajs"],
            "n_trajs_2": stats_2["n_trajs"],
            "l_outer": float(l_outer),
            "outer_grad_norm": d_outer_norm.item(),

            "l_inner_1": stats_1["l_inner"],
            "l_inner_2": stats_2["l_inner"],
            "l_inner_abs_difference": abs(
                stats_1["l_inner"] - stats_2["l_inner"]
            ),

            "inner_grad_cosine": _safe_cosine(
                stats_1["inner_grad"],
                stats_2["inner_grad"],
            ),
            "inner_grad_symmetric_relative_difference":
                _symmetric_relative_difference(
                    stats_1["inner_grad"],
                    stats_2["inner_grad"],
                ),
            "inner_grad_norm_ratio": _norm_ratio(
                stats_1["inner_grad"],
                stats_2["inner_grad"],
            ),

            "fisher_cosine": _safe_cosine(
                stats_1["fisher"],
                stats_2["fisher"],
            ),
            "fisher_symmetric_relative_difference":
                _symmetric_relative_difference(
                    stats_1["fisher"],
                    stats_2["fisher"],
                ),
            "fisher_norm_ratio": _norm_ratio(
                stats_1["fisher"],
                stats_2["fisher"],
            ),

            "solve_cosine": _safe_cosine(
                stats_1["solve"],
                stats_2["solve"],
            ),
            "solve_symmetric_relative_difference":
                _symmetric_relative_difference(
                    stats_1["solve"],
                    stats_2["solve"],
                ),
            "solve_norm_ratio": _norm_ratio(
                stats_1["solve"],
                stats_2["solve"],
            ),

            "hypergrad_cosine": _safe_cosine(
                stats_1["hypergrad"],
                stats_2["hypergrad"],
            ),
            "hypergrad_symmetric_relative_difference":
                _symmetric_relative_difference(
                    stats_1["hypergrad"],
                    stats_2["hypergrad"],
                ),
            "hypergrad_norm_ratio": _norm_ratio(
                stats_1["hypergrad"],
                stats_2["hypergrad"],
            ),
        }

        # ---------------------------------------------------------
        # Printing.
        # ---------------------------------------------------------

        print("\nIndependent trajectory-set comparison")
        print(
            f"Set 1: {stats_1['n_trajs']} trajectories | "
            f"Set 2: {stats_2['n_trajs']} trajectories"
        )
        print(
            f"L_outer={float(l_outer):.6f} | "
            f"||g_outer||={d_outer_norm.item():.3e} | "
            f"damping={self.fisher_reg:.3e}"
        )

        header = (
            f"{'quantity':>14} | "
            f"{'set 1 norm':>12} | "
            f"{'set 2 norm':>12} | "
            f"{'ratio 1/2':>10} | "
            f"{'sym rel':>10} | "
            f"{'cosine':>10}"
        )

        print("=" * len(header))
        print(header)
        print("-" * len(header))

        rows = [
            (
                "inner grad",
                stats_1["inner_grad_norm"],
                stats_2["inner_grad_norm"],
                comparison["inner_grad_norm_ratio"],
                comparison[
                    "inner_grad_symmetric_relative_difference"
                ],
                comparison["inner_grad_cosine"],
            ),
            (
                "Fisher",
                stats_1["fisher_norm"],
                stats_2["fisher_norm"],
                comparison["fisher_norm_ratio"],
                comparison[
                    "fisher_symmetric_relative_difference"
                ],
                comparison["fisher_cosine"],
            ),
            (
                "solve",
                stats_1["solve_norm"],
                stats_2["solve_norm"],
                comparison["solve_norm_ratio"],
                comparison[
                    "solve_symmetric_relative_difference"
                ],
                comparison["solve_cosine"],
            ),
            (
                "hypergrad",
                stats_1["hypergrad_norm"],
                stats_2["hypergrad_norm"],
                comparison["hypergrad_norm_ratio"],
                comparison[
                    "hypergrad_symmetric_relative_difference"
                ],
                comparison["hypergrad_cosine"],
            ),
        ]

        for (
            name,
            norm_1,
            norm_2,
            ratio,
            relative_difference,
            cosine,
        ) in rows:
            print(
                f"{name:>14} | "
                f"{norm_1:12.3e} | "
                f"{norm_2:12.3e} | "
                f"{ratio:10.4f} | "
                f"{relative_difference:10.4f} | "
                f"{cosine:10.4f}"
            )

        print("-" * len(header))

        print(
            "Inner objective | "
            f"set 1={stats_1['l_inner']:.6f} | "
            f"set 2={stats_2['l_inner']:.6f} | "
            f"abs diff={comparison['l_inner_abs_difference']:.3e}"
        )

        print(
            "Fisher trace | "
            f"set 1={stats_1['fisher_trace']:.3e} | "
            f"set 2={stats_2['fisher_trace']:.3e}"
        )

        print(
            "Fisher diag mean | "
            f"set 1={stats_1['fisher_diag_mean']:.3e} | "
            f"set 2={stats_2['fisher_diag_mean']:.3e}"
        )

        print(
            "Solve amplification | "
            f"set 1={stats_1['solve_amplification']:.3e} | "
            f"set 2={stats_2['solve_amplification']:.3e}"
        )

        print(
            "Solve residual | "
            f"set 1={stats_1['solve_residual']:.3e} | "
            f"set 2={stats_2['solve_residual']:.3e}"
        )

        print("=" * len(header))

        return {
            "set_1": {
                key: value
                for key, value in stats_1.items()
                if not isinstance(value, torch.Tensor)
            },
            "set_2": {
                key: value
                for key, value in stats_2.items()
                if not isinstance(value, torch.Tensor)
            },
            "comparison": comparison,

            # Можно сохранить сами векторы отдельно для последующего анализа.
            "vectors": {
                "inner_grad_1": stats_1["inner_grad"].detach(),
                "inner_grad_2": stats_2["inner_grad"].detach(),
                "solve_1": stats_1["solve"].detach(),
                "solve_2": stats_2["solve"].detach(),
                "hypergrad_1": stats_1["hypergrad"].detach(),
                "hypergrad_2": stats_2["hypergrad"].detach(),
            },
        }

    def d_inner_d_policy(self, trajs, verbose: bool = True) -> torch.Tensor:
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