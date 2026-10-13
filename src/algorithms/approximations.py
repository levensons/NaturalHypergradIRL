import time

import torch


class CBSCFD:
    def __init__(self, dim: int, m: int, reg: float, dtype: torch.dtype = torch.float32):
        if m <= 0:
            raise ValueError(f"`m` must be positive, got m={m}.")

        if 2 * m > dim:
            raise ValueError(f"`m` must satisfy 2 * m <= dim, got m={m}, dim={dim}.")

        if reg <= 0:
            raise ValueError(f"`reg` must be positive, got reg={reg}.")

        self.dim = dim
        self.m = m
        self.dtype = dtype

        self.alpha = torch.tensor(reg, dtype=self.dtype)
        self.Z = torch.zeros(2 * m, dim, dtype=self.dtype)
        self.H = torch.full(size=(m,), fill_value=1 / reg, dtype=self.dtype)

        self.ptr = m

        self.extend_time = 0.0
        self.svd_time = 0.0
        self.svd_count = 0
        self.solve_time = 0.0
        self.rows_processed = 0

    @torch.no_grad()
    def extend(self, rows: torch.Tensor):
        start_extend = time.perf_counter()

        if rows.ndim == 1:
            rows = rows.unsqueeze(0)
        if rows.ndim != 2:
            raise ValueError(f"`rows` must have shape ({self.dim},) or (n, {self.dim}), got {tuple(rows.shape)}.")
        if rows.shape[1] != self.dim:
            raise ValueError(f"`rows` must have last dimension {self.dim}, got {tuple(rows.shape)}.")
        
        # rows: (batch_size, dim)
        rows = rows.to(device=self.Z.device, dtype=self.Z.dtype)

        self.rows_processed += rows.shape[0]

        start = 0
        while start < rows.shape[0]:
            take = min(2 * self.m - self.ptr, rows.shape[0] - start)

            self.Z[self.ptr : self.ptr + take].copy_(rows[start : start + take])
            self.ptr += take

            if self.ptr == 2 * self.m:
                start_svd = time.perf_counter()
                _, S, Vh = torch.linalg.svd(self.Z, full_matrices=False)
                self.svd_time += time.perf_counter() - start_svd
                self.svd_count += 1

                delta = S[self.m - 1].square()
                self.alpha.add_(delta)
                S.square_().sub_(delta).clamp_(min=0.0).sqrt_()

                Vh.mul_(S.unsqueeze(1))
                self.Z.copy_(Vh)
                self.Z[self.m:].zero_()

                self.H.copy_(1 / (S[:self.m].square() + self.alpha))

                del Vh, S

                self.ptr = self.m

            start += take

        self.extend_time += time.perf_counter() - start_extend

    @torch.no_grad()
    def get_Z(self):
        return self.Z[:self.ptr]

    @torch.no_grad()
    def get_H(self, cholesky: bool = True):
        if self.ptr == self.m:
            return torch.diag(self.H)
        
        Z = self.Z[:self.m]
        B = self.Z[self.m : self.ptr]

        I_b = torch.eye(B.shape[0], device=B.device, dtype=B.dtype)
        
        C = Z @ B.T
        P = self.H.reshape(-1, 1) * C

        K = B @ B.T - C.T @ P + self.alpha * I_b
        K = 0.5 * (K + K.T)

        del C

        if cholesky:
            L = torch.linalg.cholesky(K)
            K_inv = torch.cholesky_solve(I_b, L)
            del L
        else:
            K_inv = torch.linalg.inv(K)

        del K, I_b

        P_K_inv = P @ K_inv

        H_new = torch.empty(size=(self.ptr, self.ptr), device=Z.device, dtype=Z.dtype)

        H_new[:self.m, :self.m] = torch.diag(self.H) + P_K_inv @ P.T
        H_new[:self.m, self.m:] = -P_K_inv
        H_new[self.m:, :self.m] = -P_K_inv.T
        H_new[self.m:, self.m:] = K_inv

        H_new = 0.5 * (H_new + H_new.T)

        return H_new

    @torch.no_grad()
    def solve(self, g: torch.Tensor, cholesky: bool = True):
        start_solve = time.perf_counter()

        if g.ndim != 1 or g.shape[0] != self.dim:
            raise ValueError(f"g must have shape [{self.dim}], got {g.shape}.")

        H = self.get_H(cholesky=cholesky)
        Z = self.get_Z()

        v = Z @ g
        v = H @ v
        v = Z.T @ v
        out = (g - v) / self.alpha

        self.solve_time += time.perf_counter() - start_solve
        return out

    @property
    def profiling(self) -> dict:
        mean_svd_time = self.svd_time / self.svd_count if self.svd_count > 0 else 0.0
        svd_fraction = self.svd_time / self.extend_time if self.extend_time > 0 else 0.0

        return {
            "rows_processed": self.rows_processed,
            "svd_count": self.svd_count,
            "extend_time": self.extend_time,
            "svd_time": self.svd_time,
            "mean_svd_time": mean_svd_time,
            "svd_fraction": svd_fraction,
            "solve_time": self.solve_time,
            "total_time": self.extend_time + self.solve_time,
        }
