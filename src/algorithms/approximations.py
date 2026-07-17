import torch
import time


class SCFD:
    def __init__(self, dim: int, m: int, alpha_0: float, eps: float = 1e-6):
        if dim <= 0:
            raise ValueError(f"`dim` must be positive, got dim={dim}.")
        if m <= 0:
            raise ValueError(f"`m` must be positive, got m={m}.")
        if m > dim:
            raise ValueError(f"`m` must be <= dim, got m={m}, dim={dim}.")
        if alpha_0 <= 0:
            raise ValueError(f"`alpha_0` must be positive, got alpha_0={alpha_0}.")

        self.dim = dim
        self.m = m
        self.eps = eps

        self.alpha = torch.tensor(alpha_0, dtype=torch.float32)
        self.Z = torch.zeros(2 * m, dim, dtype=torch.float32)
        self.rank = 0

    @property
    def active_Z(self) -> torch.Tensor:
        return self.Z[: self.rank]

    @torch.no_grad()
    def extend(self, rows: torch.Tensor) -> None:
        rows = rows.detach().to(device=self.Z.device, dtype=self.Z.dtype)

        if rows.ndim == 1:
            rows = rows.unsqueeze(0)
        if rows.ndim != 2:
            raise ValueError(f"`rows` must have shape ({self.dim},) " f"or (n, {self.dim}), got {tuple(rows.shape)}.")
        if rows.shape[1] != self.dim:
            raise ValueError(f"`rows` must have last dimension {self.dim}, " f"got {tuple(rows.shape)}.")

        start = 0

        while start < rows.shape[0]:
            free = 2 * self.m - self.rank
            take = min(free, rows.shape[0] - start)

            self.Z[self.rank : self.rank + take].copy_(rows[start : start + take])

            self.rank += take
            start += take

            if self.rank == 2 * self.m:
                self.update()

    @torch.no_grad()
    def update(self) -> None:
        if self.rank < 2 * self.m:
            return

        Z = self.active_Z

        _, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        delta = S[self.m - 1].square()
        self.alpha.add_(delta)

        S_new = torch.sqrt(torch.clamp(S.square() - delta, min=0.0))

        mask = S_new > self.eps
        S_new = S_new[mask]
        Vh = Vh[mask]

        new_rank = S_new.numel()

        self.Z.zero_()

        if new_rank > 0:
            self.Z[:new_rank].copy_(S_new.unsqueeze(1) * Vh)

        self.rank = new_rank

    @torch.no_grad()
    def solve(self, g: torch.Tensor) -> torch.Tensor:
        g = g.detach().to(device=self.Z.device, dtype=self.Z.dtype)

        if g.ndim != 1:
            raise ValueError(f"`g` must be a vector, got shape {tuple(g.shape)}.")
        if g.shape[0] != self.dim:
            raise ValueError(f"`g` must have shape ({self.dim},), " f"got {tuple(g.shape)}.")
        if self.rank == 0:
            return g / self.alpha

        Z = self.active_Z
        small = torch.matmul(Z, Z.T)
        small.diagonal().add_(self.alpha)
        q = torch.matmul(Z, g)
        u = torch.linalg.solve(small, q)
        return (g - torch.matmul(Z.T, u)) / self.alpha

    @torch.no_grad()
    def inv(self) -> torch.Tensor:
        "DEBUG ONLY"
        I = torch.eye(self.dim, device=self.Z.device, dtype=self.Z.dtype)

        if self.rank == 0:
            return I / self.alpha

        Z = self.active_Z
        small = torch.matmul(Z, Z.T)
        small.diagonal().add_(self.alpha)
        solved = torch.linalg.solve(small, Z)
        return (I - torch.matmul(Z.T, solved)) / self.alpha


class CBSCFD:
    def __init__(self, dim: int, m: int, alpha_0: float, eps: float = 1e-6):
        if m <= 0:
            raise ValueError(f"`m` must be positive, got m={m}.")
        if m > dim:
            raise ValueError(f"`m` must be <= dim, got m={m}, dim={dim}.")
        if alpha_0 <= 0:
            raise ValueError(f"`alpha_0` must be positive, got alpha_0={alpha_0}.")

        self.dim = dim
        self.m = m
        self.eps = eps

        self.alpha = torch.tensor(alpha_0, dtype=torch.float32)
        self.Z = torch.zeros(2 * m, dim, dtype=torch.float32)
        self.H = torch.empty(0, 0, dtype=torch.float32)
        self.rank = 0

        self.append_time = 0.0
        self.update_time = 0.0
        self.solve_time = 0.0

        self.append_calls = 0
        self.update_calls = 0
        self.solve_calls = 0

        self.blocks_seen = 0
        self.rows_seen = 0

        self.appended_rows = 0
        self.max_append_block_size = 0

    @property
    def active_Z(self) -> torch.Tensor:
        return self.Z[: self.rank]

    @torch.no_grad()
    def extend(self, rows: torch.Tensor) -> None:
        rows = rows.detach().to(device=self.Z.device, dtype=self.Z.dtype)

        if rows.ndim == 1:
            rows = rows.unsqueeze(0)
        if rows.ndim != 2:
            raise ValueError(f"`rows` must have shape ({self.dim},) or " f"(n, {self.dim}), got {tuple(rows.shape)}.")
        if rows.shape[1] != self.dim:
            raise ValueError(f"`rows` must have last dimension {self.dim}, " f"got {tuple(rows.shape)}.")

        self.blocks_seen += 1
        self.rows_seen += rows.shape[0]

        start = 0

        while start < rows.shape[0]:
            free = 2 * self.m - self.rank
            take = min(free, rows.shape[0] - start)

            block = rows[start : start + take]

            will_shrink = self.rank + take == 2 * self.m

            if not will_shrink:
                append_start = time.perf_counter()

                self.append_block(block)

                self.append_time += time.perf_counter() - append_start
                self.append_calls += 1
                self.appended_rows += take
                self.max_append_block_size = max(self.max_append_block_size, take)

            else:
                self.Z[self.rank : self.rank + take].copy_(block)
                self.rank += take

            start += take

            if self.rank == 2 * self.m:
                update_start = time.perf_counter()
                self.update()
                self.update_time += time.perf_counter() - update_start
                self.update_calls += 1

    @torch.no_grad()
    def append_block(self, rows: torch.Tensor) -> None:
        k = self.rank
        r = rows.shape[0]

        if r == 0:
            return

        if k + r > 2 * self.m:
            raise ValueError(f"Not enough space in sketch: rank={k}, " f"new_rows={r}, capacity={2 * self.m}.")

        if k == 0:
            D = torch.matmul(rows, rows.T)
            D.diagonal().add_(self.alpha)
            H_new = self.safe_inverse(D)

        else:
            Z = self.active_Z
            B = torch.matmul(Z, rows.T)  # (k, r)
            D = torch.matmul(rows, rows.T)  # (r, r)
            D.diagonal().add_(self.alpha)
            P = torch.matmul(self.H, B)  # (k, r)
            S = D - torch.matmul(B.T, P)
            S = 0.5 * (S + S.T)
            S_inv = self.safe_inverse(S)

            H_new = torch.empty(k + r, k + r, dtype=self.Z.dtype, device=self.Z.device)

            H_new[:k, :k] = self.H + torch.matmul(P, torch.matmul(S_inv, P.T))
            H_new[:k, k:] = -torch.matmul(P, S_inv)
            H_new[k:, :k] = H_new[:k, k:].T
            H_new[k:, k:] = S_inv

        self.Z[k : k + r].copy_(rows)
        self.rank = k + r
        self.H = H_new

    @torch.no_grad()
    def update(self) -> None:
        if self.rank < 2 * self.m:
            return

        Z = self.active_Z

        _, singular_values, Vh = torch.linalg.svd(Z, full_matrices=False)
        delta = singular_values[self.m - 1].square()
        self.alpha.add_(delta)

        shrunk_squared = torch.clamp(singular_values.square() - delta, min=0.0)

        singular_values_new = torch.sqrt(shrunk_squared)

        mask = singular_values_new > self.eps
        singular_values_new = singular_values_new[mask]
        Vh = Vh[mask]

        new_rank = singular_values_new.numel()

        self.Z.zero_()

        if new_rank > 0:
            self.Z[:new_rank].copy_(singular_values_new.unsqueeze(1) * Vh)
            Z_new = self.Z[:new_rank]
            G = torch.matmul(Z_new, Z_new.T)
            G.diagonal().add_(self.alpha)
            self.H = self.safe_inverse(G)

        else:
            self.H = torch.empty(0, 0, dtype=self.Z.dtype, device=self.Z.device)

        self.rank = new_rank

    @torch.no_grad()
    def solve(self, g: torch.Tensor) -> torch.Tensor:
        solve_start = time.perf_counter()

        g = g.detach().to(device=self.Z.device, dtype=self.Z.dtype)

        if g.ndim != 1:
            raise ValueError(f"`g` must be a vector, got shape {tuple(g.shape)}.")
        if g.shape[0] != self.dim:
            raise ValueError(f"`g` must have shape ({self.dim},), " f"got {tuple(g.shape)}.")
        if self.rank == 0:
            result = g / self.alpha

        else:
            Z = self.active_Z
            Zg = torch.matmul(Z, g)
            HZg = torch.matmul(self.H, Zg)
            result = (g - torch.matmul(Z.T, HZg)) / self.alpha

        self.solve_time += time.perf_counter() - solve_start
        self.solve_calls += 1
        return result

    @torch.no_grad()
    def inv(self) -> torch.Tensor:
        "DEBUG ONLY"
        identity = torch.eye(self.dim, device=self.Z.device, dtype=self.Z.dtype)

        if self.rank == 0:
            return identity / self.alpha

        Z = self.active_Z
        return (identity - torch.matmul(Z.T, torch.matmul(self.H, Z))) / self.alpha

    @torch.no_grad()
    def safe_inverse(self, x: torch.Tensor) -> torch.Tensor:
        x = 0.5 * (x + x.T)
        try:
            return torch.linalg.inv(x)
        except RuntimeError:
            x = x.clone()
            x.diagonal().add_(self.eps)
            return torch.linalg.inv(x)

    def profiling_stats(self) -> dict:
        return {
            "append_time": self.append_time,
            "update_time": self.update_time,
            "solve_time": self.solve_time,
            "append_calls": self.append_calls,
            "update_calls": self.update_calls,
            "solve_calls": self.solve_calls,
            "blocks_seen": self.blocks_seen,
            "rows_seen": self.rows_seen,
            "mean_input_block_size": (self.rows_seen / max(self.blocks_seen, 1)),
            "mean_append_block_size": (self.appended_rows / max(self.append_calls, 1)),
            "max_append_block_size": self.max_append_block_size,
            "final_rank": self.rank,
            "final_alpha": float(self.alpha.item()),
        }
