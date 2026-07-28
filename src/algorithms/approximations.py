import torch


class CBSCFD:
    def __init__(self, dim: int, m: int, reg: float):
        self.dim = dim
        self.m = m

        self.alpha = torch.tensor(reg, dtype=torch.float32)
        self.Z = torch.zeros(2 * m, dim, dtype=torch.float32)
        self.H = torch.full(size=(m,), fill_value=1 / reg, dtype=torch.float32)

        self.ptr = m

    @torch.no_grad()
    def extend(self, rows: torch.Tensor):
        if rows.ndim == 1:
            rows = rows.unsqueeze(0)
        if rows.ndim != 2:
            raise ValueError(f"`rows` must have shape ({self.dim},) or (n, {self.dim}), got {tuple(rows.shape)}.")
        if rows.shape[1] != self.dim:
            raise ValueError(f"`rows` must have last dimension {self.dim}, got {tuple(rows.shape)}.")
        
        # rows: (batch_size, dim)
        rows = rows.to(device=self.Z.device, dtype=self.Z.dtype)

        start = 0
        while start < rows.shape[0]:
            take = min(2 * self.m - self.ptr, rows.shape[0] - start)

            self.Z[self.ptr : self.ptr + take].copy_(rows[start : start + take])
            self.ptr += take

            if self.ptr == 2 * self.m:
                _, S, Vh = torch.linalg.svd(self.Z, full_matrices=False)

                delta = S[self.m - 1].square()
                self.alpha.add_(delta)
                S = torch.sqrt(torch.clamp(S.square() - delta, min=0.0))

                self.Z.copy_(S.reshape(-1, 1) * Vh)
                self.Z[self.m:].zero_()

                self.H.copy_(1 / (S[:self.m].square() + self.alpha))

                self.ptr = self.m

            start += take

    @torch.no_grad()
    def get_Z(self):
        return self.Z[:self.ptr]

    @torch.no_grad()
    def get_H(self):
        if self.ptr == self.m:
            return torch.diag(self.H)
        
        Z = self.Z[:self.m]
        B = self.Z[self.m : self.ptr]

        I_b = torch.eye(B.shape[0], dtype=torch.float32)
        
        C = Z @ B.T
        P = self.H.reshape(-1, 1) * C
        K = B @ B.T - C.T @ P + self.alpha * I_b
        K = 0.5 * (K + K.T)

        K_inv = torch.linalg.inv(K)
        P_K_inv = P @ K_inv

        H_new = torch.empty(size=(self.ptr, self.ptr), dtype=torch.float32)

        H_new[:self.m, :self.m] = torch.diag(self.H) + P_K_inv @ P.T
        H_new[:self.m, self.m:] = -P_K_inv
        H_new[self.m:, :self.m] = -P_K_inv.T
        H_new[self.m:, self.m:] = K_inv

        H_new = 0.5 * (H_new + H_new.T)

        return H_new

    @torch.no_grad()
    def solve(self, g: torch.Tensor):
        if g.ndim != 1 or g.shape[0] != self.dim:
            raise ValueError(f"g must have shape [{self.dim}], got {g.shape}.")

        H = self.get_H()
        Z = self.get_Z()

        v = Z @ g
        v = H @ v
        v = Z.T @ v
        return (g - v) / self.alpha


# class CBSCFD:
#     def __init__(self, dim: int, m: int, alpha: float, eps: float = 1e-6):
#         self.dim = dim
#         self.m = m
#         self.eps = eps

#         self.alpha = torch.tensor(alpha, dtype=torch.float32)
#         self.Z = torch.zeros(2 * m, dim, dtype=torch.float32)
#         self.H = torch.empty(0, 0, dtype=torch.float32)
#         self.rank = 0

#     @property
#     def active_Z(self) -> torch.Tensor:
#         return self.Z[: self.rank]

#     @torch.no_grad()
#     def extend(self, rows: torch.Tensor) -> None:
#         rows = rows.detach().to(device=self.Z.device, dtype=self.Z.dtype)

#         if rows.ndim == 1:
#             rows = rows.unsqueeze(0)
#         if rows.ndim != 2:
#             raise ValueError(f"`rows` must have shape ({self.dim},) or " f"(n, {self.dim}), got {tuple(rows.shape)}.")
#         if rows.shape[1] != self.dim:
#             raise ValueError(f"`rows` must have last dimension {self.dim}, " f"got {tuple(rows.shape)}.")

#         start = 0

#         while start < rows.shape[0]:
#             free = 2 * self.m - self.rank
#             take = min(free, rows.shape[0] - start)

#             block = rows[start : start + take]

#             will_shrink = self.rank + take == 2 * self.m

#             if not will_shrink:
#                 self.append_block(block)

#             else:
#                 self.Z[self.rank : self.rank + take].copy_(block)
#                 self.rank += take

#             start += take

#             if self.rank == 2 * self.m:
#                 self.update()

#     @torch.no_grad()
#     def append_block(self, rows: torch.Tensor) -> None:
#         k = self.rank
#         r = rows.shape[0]

#         if r == 0:
#             return

#         if k + r > 2 * self.m:
#             raise ValueError(f"Not enough space in sketch: rank={k}, " f"new_rows={r}, capacity={2 * self.m}.")

#         if k == 0:
#             D = torch.matmul(rows, rows.T)
#             D.diagonal().add_(self.alpha)
#             H_new = self.safe_inverse(D)

#         else:
#             Z = self.active_Z
#             B = torch.matmul(Z, rows.T)  # (k, r)
#             D = torch.matmul(rows, rows.T)  # (r, r)
#             D.diagonal().add_(self.alpha)
#             P = torch.matmul(self.H, B)  # (k, r)
#             S = D - torch.matmul(B.T, P)
#             S = 0.5 * (S + S.T)
#             S_inv = self.safe_inverse(S)

#             H_new = torch.empty(k + r, k + r, dtype=self.Z.dtype, device=self.Z.device)

#             H_new[:k, :k] = self.H + torch.matmul(P, torch.matmul(S_inv, P.T))
#             H_new[:k, k:] = -torch.matmul(P, S_inv)
#             H_new[k:, :k] = H_new[:k, k:].T
#             H_new[k:, k:] = S_inv

#         self.Z[k : k + r].copy_(rows)
#         self.rank = k + r
#         self.H = H_new

#     @torch.no_grad()
#     def update(self) -> None:
#         if self.rank < 2 * self.m:
#             return

#         Z = self.active_Z

#         _, singular_values, Vh = torch.linalg.svd(Z, full_matrices=False)
#         delta = singular_values[self.m - 1].square()
#         self.alpha.add_(delta)

#         shrunk_squared = torch.clamp(singular_values.square() - delta, min=0.0)

#         singular_values_new = torch.sqrt(shrunk_squared)

#         mask = singular_values_new > self.eps
#         singular_values_new = singular_values_new[mask]
#         Vh = Vh[mask]

#         new_rank = singular_values_new.numel()

#         self.Z.zero_()

#         if new_rank > 0:
#             self.Z[:new_rank].copy_(singular_values_new.unsqueeze(1) * Vh)
#             Z_new = self.Z[:new_rank]
#             G = torch.matmul(Z_new, Z_new.T)
#             G.diagonal().add_(self.alpha)
#             self.H = self.safe_inverse(G)

#         else:
#             self.H = torch.empty(0, 0, dtype=self.Z.dtype, device=self.Z.device)

#         self.rank = new_rank

#     @torch.no_grad()
#     def solve(self, g: torch.Tensor) -> torch.Tensor:
#         g = g.to(device=self.Z.device, dtype=self.Z.dtype)

#         if self.rank == 0:
#             result = g / self.alpha

#         else:
#             Z = self.active_Z
#             Zg = torch.matmul(Z, g)
#             HZg = torch.matmul(self.H, Zg)
#             result = (g - torch.matmul(Z.T, HZg)) / self.alpha

#         return result

#     @torch.no_grad()
#     def safe_inverse(self, x: torch.Tensor) -> torch.Tensor:
#         x = 0.5 * (x + x.T)
#         try:
#             return torch.linalg.inv(x)
#         except RuntimeError:
#             x = x.clone()
#             x.diagonal().add_(self.eps)
#             return torch.linalg.inv(x)
