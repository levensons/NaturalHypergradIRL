from tqdm import tqdm
import time

import torch
from torch import nn
import torch.nn.functional as F

from src.utils.policies import Policy
from src.algorithms.approximations import CBSCFD
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
        use_sketch: bool = True,
        sketch_size: int = 32,
        fisher_batch_size: int = 256,
    ):
        self.reward = reward
        self.policy = policy
        self.fisher_reg = fisher_reg
        self.gamma = gamma
        self.alpha = alpha
        self.max_grad_norm = max_grad_norm
        self.scheduler_gamma = scheduler_gamma
        self.use_sketch = use_sketch
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
        policy_dim = num_params(self.policy)

        g = torch.zeros(policy_dim, dtype=torch.float32, device=self.device)

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

    def exact_fisher(self, trajs, verbose: bool = True) -> torch.Tensor:
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
        if self.use_sketch:
            hypergradient = self.hypergradient_with_sketching(expert_trajs, agent_trajs)
        else:
            hypergradient = self.hypergradient_with_exact_fisher(expert_trajs, agent_trajs)

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
        eps = 1e-12

        d_outer_d_policy = self.d_outer_d_policy(expert_trajs)

        # -------------------------------------------------------------------------
        # Exact Fisher construction
        # -------------------------------------------------------------------------

        fisher_start = time.perf_counter()
        fisher_exact = self.exact_fisher(agent_trajs, torch.float32)
        fisher_end = time.perf_counter()

        fisher_construction_time = fisher_end - fisher_start

        # -------------------------------------------------------------------------
        # Spectrum of the unregularized Fisher matrix
        # -------------------------------------------------------------------------

        # fisher_for_spectrum = fisher_exact.to(dtype=torch.float32)
        # fisher_for_spectrum = 0.5 * (fisher_for_spectrum + fisher_for_spectrum.T)

        # spectrum_start = time.perf_counter()
        # eigenvalues = torch.linalg.eigvalsh(fisher_for_spectrum)
        # spectrum_end = time.perf_counter()

        # spectrum_time = spectrum_end - spectrum_start

        # min_eigenvalue = eigenvalues[0].item()
        # max_eigenvalue = eigenvalues[-1].item()
        # mean_eigenvalue = eigenvalues.mean().item()
        # median_eigenvalue = eigenvalues.median().item()

        # rank_tolerance = torch.finfo(fisher_for_spectrum.dtype).eps * fisher_for_spectrum.shape[0] * max(abs(min_eigenvalue), abs(max_eigenvalue), 1.0)

        # positive_eigenvalues = eigenvalues[eigenvalues > rank_tolerance]
        # significant_negative_eigenvalues = eigenvalues[eigenvalues < -rank_tolerance]
        # tiny_negative_eigenvalues = eigenvalues[(eigenvalues < 0.0) & (eigenvalues >= -rank_tolerance)]

        # numerical_rank = positive_eigenvalues.numel()
        # numerical_nullity = fisher_for_spectrum.shape[0] - numerical_rank

        # if positive_eigenvalues.numel() > 0:
        #     min_positive_eigenvalue = positive_eigenvalues[0].item()
        #     positive_condition_number = max_eigenvalue / min_positive_eigenvalue
        # else:
        #     min_positive_eigenvalue = float("nan")
        #     positive_condition_number = float("inf")

        # regularized_eigenvalues = eigenvalues + self.fisher_reg
        # regularized_min_eigenvalue = regularized_eigenvalues[0].item()
        # regularized_max_eigenvalue = regularized_eigenvalues[-1].item()

        # if regularized_min_eigenvalue > 0.0:
        #     regularized_condition_number = regularized_max_eigenvalue / regularized_min_eigenvalue
        # else:
        #     regularized_condition_number = float("inf")

        # mean_eigenvalue_scale = max(abs(mean_eigenvalue), eps)
        # max_eigenvalue_scale = max(abs(max_eigenvalue), eps)

        # print("\nUnregularized Fisher spectrum")
        # print(f"Spectrum computation time:          {spectrum_time:.4f}s")
        # print(f"Minimum eigenvalue:                 {min_eigenvalue:.8e}")
        # print(f"Minimum positive eigenvalue:        {min_positive_eigenvalue:.8e}")
        # print(f"Median eigenvalue:                  {median_eigenvalue:.8e}")
        # print(f"Mean eigenvalue:                    {mean_eigenvalue:.8e}")
        # print(f"Maximum eigenvalue:                 {max_eigenvalue:.8e}")
        # print(f"Numerical rank:                     {numerical_rank}/{fisher_for_spectrum.shape[0]}")
        # print(f"Numerical nullity:                  {numerical_nullity}")
        # print(f"Significant negative eigenvalues:   {significant_negative_eigenvalues.numel()}")
        # print(f"Tiny negative eigenvalues:          {tiny_negative_eigenvalues.numel()}")
        # print(f"Positive-spectrum condition number: {positive_condition_number:.8e}")
        # print()
        # print(f"Regularization:                     {self.fisher_reg:.8e}")
        # print(f"Regularization / mean eigenvalue:   {self.fisher_reg / mean_eigenvalue_scale:.8e}")
        # print(f"Regularization / max eigenvalue:    {self.fisher_reg / max_eigenvalue_scale:.8e}")
        # print(f"Regularized minimum eigenvalue:     {regularized_min_eigenvalue:.8e}")
        # print(f"Regularized maximum eigenvalue:     {regularized_max_eigenvalue:.8e}")
        # print(f"Regularized condition estimate:     {regularized_condition_number:.8e}")

        # spectrum_stats = {
        #     "spectrum_time": spectrum_time,
        #     "min_eigenvalue": min_eigenvalue,
        #     "min_positive_eigenvalue": min_positive_eigenvalue,
        #     "median_eigenvalue": median_eigenvalue,
        #     "mean_eigenvalue": mean_eigenvalue,
        #     "max_eigenvalue": max_eigenvalue,
        #     "numerical_rank": numerical_rank,
        #     "numerical_nullity": numerical_nullity,
        #     "significant_negative_eigenvalues": significant_negative_eigenvalues.numel(),
        #     "tiny_negative_eigenvalues": tiny_negative_eigenvalues.numel(),
        #     "positive_condition_number": positive_condition_number,
        #     "regularization": float(self.fisher_reg),
        #     "regularization_to_mean_eigenvalue": self.fisher_reg / mean_eigenvalue_scale,
        #     "regularization_to_max_eigenvalue": self.fisher_reg / max_eigenvalue_scale,
        #     "regularized_min_eigenvalue": regularized_min_eigenvalue,
        #     "regularized_max_eigenvalue": regularized_max_eigenvalue,
        #     "regularized_condition_number": regularized_condition_number,
        # }

        # del fisher_for_spectrum
        # del eigenvalues
        # del positive_eigenvalues
        # del significant_negative_eigenvalues
        # del tiny_negative_eigenvalues

        # -------------------------------------------------------------------------
        # Exact regularized solve
        # -------------------------------------------------------------------------

        fisher_exact.diagonal().add_(self.fisher_reg)

        exact_g = d_outer_d_policy.to(
            device=fisher_exact.device,
            dtype=fisher_exact.dtype,
        )

        solve_start = time.perf_counter()
        v_exact = torch.linalg.solve(fisher_exact, exact_g)
        solve_end = time.perf_counter()

        exact_solve_time = solve_end - solve_start
        exact_pipeline_time = fisher_construction_time + exact_solve_time

        exact_g_norm = exact_g.norm().item()
        exact_v_norm = v_exact.norm().item()
        exact_v_abs_min = v_exact.abs().min().item()
        exact_v_abs_max = v_exact.abs().max().item()

        exact_residual = fisher_exact @ v_exact - exact_g
        exact_residual_norm = exact_residual.norm().item()
        exact_relative_residual = exact_residual_norm / max(exact_g_norm, eps)

        exact_is_finite = bool(torch.isfinite(v_exact).all().item())

        print("\nExact regularized Fisher solve")
        print(f"Fisher construction time:       {fisher_construction_time:>10.4f}s")
        # print(f"Fisher spectrum analysis time:  {spectrum_time:>10.4f}s")
        print(f"Exact Fisher solve time:        {exact_solve_time:>10.4f}s")
        print(f"Exact pipeline time:            {exact_pipeline_time:>10.4f}s")
        # print(f"Total including spectrum:       {exact_pipeline_time + spectrum_time:>10.4f}s")
        print()
        print(f"Fisher shape:                   {tuple(fisher_exact.shape)}")
        print(f"Fisher dtype:                   {fisher_exact.dtype}")
        print(f"Fisher device:                  {fisher_exact.device}")
        print(f"Fisher regularization:          {self.fisher_reg:.8e}")
        print(f"||g||:                          {exact_g_norm:.8e}")
        print(f"||v_exact||:                    {exact_v_norm:.8e}")
        print(f"min |v_exact_i|:                {exact_v_abs_min:.8e}")
        print(f"max |v_exact_i|:                {exact_v_abs_max:.8e}")
        print(f"exact residual norm:            {exact_residual_norm:.8e}")
        print(f"exact relative residual:        {exact_relative_residual:.8e}")
        print(f"exact is finite:                {exact_is_finite}")

        hypergrad_exact = None
        exact_hypergrad_norm = None

        if compare_hypergradients:
            policy_parameter = next(self.policy.parameters())

            v_exact_for_cross = v_exact.to(
                device=policy_parameter.device,
                dtype=policy_parameter.dtype,
            )

            hypergrad_exact = -self.d_inner_d_cross_vec_product(
                agent_trajs,
                v_exact_for_cross,
            )

            exact_hypergrad_norm = hypergrad_exact.norm().item()
            exact_hypergrad_is_finite = bool(torch.isfinite(hypergrad_exact).all().item())

            print(f"||hypergrad_exact||:            {exact_hypergrad_norm:.8e}")
            print(f"hypergrad exact is finite:      {exact_hypergrad_is_finite}")

        # -------------------------------------------------------------------------
        # Sketch sweep
        # -------------------------------------------------------------------------

        header = (
            f"{'m':>7} | "
            f"{'||v_s||':>13} | "
            f"{'||v_e||':>13} | "
            f"{'norm ratio':>12} | "
            f"{'abs error':>13} | "
            f"{'rel error':>11} | "
            f"{'cosine':>10} | "
            f"{'rel residual':>13} | "
            f"{'finite':>7} | "
            f"{'time':>10}"
        )

        if compare_hypergradients:
            header += (
                f" | "
                f"{'||h_s||':>13} | "
                f"{'||h_e||':>13} | "
                f"{'h norm ratio':>12} | "
                f"{'h rel error':>11} | "
                f"{'unit h rel':>11} | "
                f"{'h cosine':>10}"
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
            fisher_exact_local = fisher_exact.to(
                device=v_sketch.device,
                dtype=v_sketch.dtype,
            )
            exact_g_local = exact_g.to(
                device=v_sketch.device,
                dtype=v_sketch.dtype,
            )

            sketch_v_norm = v_sketch.norm().item()
            local_exact_v_norm = v_exact_local.norm().item()

            solve_absolute_error = (v_sketch - v_exact_local).norm().item()
            solve_relative_error = solve_absolute_error / max(local_exact_v_norm, eps)
            norm_ratio = sketch_v_norm / max(local_exact_v_norm, eps)

            sketch_is_finite = bool(torch.isfinite(v_sketch).all().item())

            if sketch_v_norm > eps and local_exact_v_norm > eps and sketch_is_finite:
                solve_cosine = F.cosine_similarity(
                    v_sketch.reshape(-1),
                    v_exact_local.reshape(-1),
                    dim=0,
                    eps=eps,
                ).item()
            else:
                solve_cosine = float("nan")

            sketch_residual = fisher_exact_local @ v_sketch - exact_g_local
            sketch_residual_norm = sketch_residual.norm().item()
            sketch_relative_residual = sketch_residual_norm / max(exact_g_local.norm().item(), eps)

            result = {
                "sketch_size": int(sketch_size),
                "exact_gradient_norm": exact_g_norm,
                "exact_solution_norm": local_exact_v_norm,
                "sketch_solution_norm": sketch_v_norm,
                "solve_norm_ratio": norm_ratio,
                "solve_absolute_error": solve_absolute_error,
                "solve_relative_error": solve_relative_error,
                "solve_cosine": solve_cosine,
                "solve_residual_norm": sketch_residual_norm,
                "solve_relative_residual": sketch_relative_residual,
                "solution_is_finite": sketch_is_finite,
                "time": sketch_time,
                # "spectrum": spectrum_stats,
            }

            row = (
                f"{sketch_size:>7} | "
                f"{sketch_v_norm:>13.5e} | "
                f"{local_exact_v_norm:>13.5e} | "
                f"{norm_ratio:>12.5e} | "
                f"{solve_absolute_error:>13.5e} | "
                f"{solve_relative_error:>11.5e} | "
                f"{solve_cosine:>10.5f} | "
                f"{sketch_relative_residual:>13.5e} | "
                f"{str(sketch_is_finite):>7} | "
                f"{sketch_time:>9.3f}s"
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

                sketch_hypergrad_norm = hypergrad_sketch.norm().item()
                local_exact_hypergrad_norm = hypergrad_exact_local.norm().item()

                hypergrad_norm_ratio = sketch_hypergrad_norm / max(local_exact_hypergrad_norm, eps)
                hypergrad_absolute_error = (hypergrad_sketch - hypergrad_exact_local).norm().item()
                hypergrad_relative_error = hypergrad_absolute_error / max(local_exact_hypergrad_norm, eps)
                hypergrad_is_finite = bool(torch.isfinite(hypergrad_sketch).all().item())

                if sketch_hypergrad_norm > eps and local_exact_hypergrad_norm > eps and hypergrad_is_finite:
                    hypergrad_cosine = F.cosine_similarity(
                        hypergrad_sketch.reshape(-1),
                        hypergrad_exact_local.reshape(-1),
                        dim=0,
                        eps=eps,
                    ).item()

                    hypergrad_sketch_unit = hypergrad_sketch / sketch_hypergrad_norm
                    hypergrad_exact_unit = hypergrad_exact_local / local_exact_hypergrad_norm
                    hypergrad_unit_relative_error = (hypergrad_sketch_unit - hypergrad_exact_unit).norm().item()
                else:
                    hypergrad_cosine = float("nan")
                    hypergrad_unit_relative_error = float("nan")

                result.update(
                    {
                        "exact_hypergrad_norm": local_exact_hypergrad_norm,
                        "sketch_hypergrad_norm": sketch_hypergrad_norm,
                        "hypergrad_norm_ratio": hypergrad_norm_ratio,
                        "hypergrad_absolute_error": hypergrad_absolute_error,
                        "hypergrad_relative_error": hypergrad_relative_error,
                        "hypergrad_unit_relative_error": hypergrad_unit_relative_error,
                        "hypergrad_cosine": hypergrad_cosine,
                        "hypergrad_is_finite": hypergrad_is_finite,
                    }
                )

                row += (
                    f" | "
                    f"{sketch_hypergrad_norm:>13.5e} | "
                    f"{local_exact_hypergrad_norm:>13.5e} | "
                    f"{hypergrad_norm_ratio:>12.5e} | "
                    f"{hypergrad_relative_error:>11.5e} | "
                    f"{hypergrad_unit_relative_error:>11.5e} | "
                    f"{hypergrad_cosine:>10.5f}"
                )

            print(row)
            results.append(result)

        return results
    
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
