import torch


class CBSCFD:
    def __init__(self, dim: int, m: int, reg: float, dtype: torch.dtype = torch.float32):
        self.dim = dim
        self.m = m
        self.dtype = dtype

        self.alpha = torch.tensor(reg, dtype=self.dtype)
        self.Z = torch.zeros(2 * m, dim, dtype=self.dtype)
        self.H = torch.full(size=(m,), fill_value=1 / reg, dtype=self.dtype)

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

        if cholesky:
            L = torch.linalg.cholesky(K)
            K_inv = torch.cholesky_solve(I_b, L)
        else:
            K_inv = torch.linalg.inv(K)

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
        if g.ndim != 1 or g.shape[0] != self.dim:
            raise ValueError(f"g must have shape [{self.dim}], got {g.shape}.")

        H = self.get_H(cholesky=cholesky)
        Z = self.get_Z()

        v = Z @ g
        v = H @ v
        v = Z.T @ v
        return (g - v) / self.alpha
