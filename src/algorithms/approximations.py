import torch


class SCFD:
    def __init__(self, dim: int, m: int, alpha_0: float, eps: float = 1e-12):
        super().__init__()

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
        self.Z = torch.empty((0, self.dim), dtype=torch.float32)  # (r, d)
        self.H = torch.empty((0, 0), dtype=torch.float32)  # (r, r) = (Z Z^T + alpha I)^{-1}

    @torch.no_grad()
    def extend(self, x: torch.Tensor):
        x = x.detach().to(device=self.Z.device, dtype=self.Z.dtype)

        if x.ndim != 1:
            raise ValueError(f"`x` must be a vector, got shape {tuple(x.shape)}.")
        if x.shape[0] != self.dim:
            raise ValueError(f"`x` must have shape ({self.dim},), got {tuple(x.shape)}.")

        will_shrink = (self.Z.shape[0] + 1) >= 2 * self.m

        if not will_shrink:
            self.border_update_H(x)

        self.Z = torch.cat([self.Z, x.reshape(1, -1)], dim=0)  # (r + 1, dim)

        if will_shrink:
            self.update()

    @torch.no_grad()
    def border_update_H(self, z: torch.Tensor):
        r = self.Z.shape[0]

        if r == 0:
            c = self.alpha + z @ z
            self.H = (1.0 / c).reshape(1, 1)
            return

        b = torch.matmul(self.Z, z)
        u = torch.matmul(self.H, b)
        c = self.alpha + torch.dot(z, z)
        s = c - torch.dot(b, u)

        r = self.H.shape[0]
        H_new = torch.empty(r + 1, r + 1, dtype=self.H.dtype, device=self.H.device)

        H_new[:r, :r] = self.H + torch.outer(u, u) / s
        H_new[:r, r] = -u / s
        H_new[r, :r] = -u / s
        H_new[r, r] = 1.0 / s

        self.H = H_new

    @torch.no_grad()
    def update(self):
        if self.Z.shape[0] < 2 * self.m:
            return

        _, S, Vh = torch.linalg.svd(self.Z, full_matrices=False)
        delta = S[self.m - 1].pow(2)
        self.alpha = self.alpha + delta

        S_new = torch.sqrt(torch.clamp(S.pow(2) - delta, min=0.0))
        self.Z = S_new.reshape(-1, 1) * Vh

        mask = S_new > self.eps
        self.Z = self.Z[mask]
        S_new = S_new[mask]

        self.H = torch.diag(1.0 / (S_new.pow(2) + self.alpha))

    @torch.no_grad()
    def solve(self, g: torch.Tensor) -> torch.Tensor:
        g = g.detach().to(device=self.Z.device, dtype=self.Z.dtype)

        if g.ndim != 1:
            raise ValueError(f"`g` must be a vector, got shape {tuple(g.shape)}.")
        if g.shape[0] != self.dim:
            raise ValueError(f"`g` must have shape ({self.dim},), got {tuple(g.shape)}.")

        if self.Z.shape[0] == 0:
            return g / self.alpha

        Zg = torch.matmul(self.Z, g)  # (r,)
        HZg = torch.matmul(self.H, Zg)  # (r,)
        return (g - torch.matmul(self.Z.T, HZg)) / self.alpha

    @torch.no_grad()
    def inv(self) -> torch.Tensor:
        """DEBUG ONLY"""
        d = self.dim
        I_d = torch.eye(d, device=self.Z.device, dtype=self.Z.dtype)

        if self.Z.shape[0] == 0:
            return I_d / self.alpha

        return (I_d - torch.einsum("rd,rs,se->de", self.Z, self.H, self.Z)) / self.alpha
