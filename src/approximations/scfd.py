import torch

class SCFD:
    def __init__(self, dim: int, m: int, alpha_0: float, eps: float = 1e-12):
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
        self.Z = torch.empty((0, self.dim), dtype=torch.float32)

    @torch.no_grad()
    def extend(self, x: torch.Tensor):
        x = x.detach().to(device=self.Z.device, dtype=self.Z.dtype)

        if x.ndim != 1:
            raise ValueError(f"`x` must be a vector, got shape {tuple(x.shape)}.")

        if x.shape[0] != self.dim:
            raise ValueError(f"`x` must have shape ({self.dim},), got {tuple(x.shape)}.")

        self.Z = torch.cat([self.Z, x.reshape(1, -1)], dim=0) # (r + 1, dim)

        if self.Z.shape[0] >= 2 * self.m:
            self.update()

    @torch.no_grad()
    def update(self):
        if self.Z.shape[0] < 2 * self.m:
            return
        
        _, S, Vh = torch.linalg.svd(self.Z, full_matrices=False)
        delta = S[self.m - 1].pow(2)
        self.alpha = self.alpha + delta
        S_new = torch.sqrt(torch.clamp(S.pow(2) - delta, min=0.0))
        self.Z = S_new.reshape(-1, 1) * Vh

        mask = (S_new > self.eps)
        self.Z = self.Z[mask]

    @torch.no_grad()
    def solve(self, g: torch.Tensor) -> torch.Tensor:
        g = g.detach().to(device=self.Z.device, dtype=self.Z.dtype)

        if g.ndim != 1:
            raise ValueError(f"`g` must be a vector, got shape {tuple(g.shape)}.")

        if g.shape[0] != self.dim:
            raise ValueError(f"`g` must have shape ({self.dim},), got {tuple(g.shape)}.")

        if self.Z.shape[0] == 0:
            return g / self.alpha
        
        r = self.Z.shape[0]
        I = torch.eye(r, device=self.Z.device, dtype=self.Z.dtype)

        q = torch.einsum("rd,d->r", self.Z, g)
        small = torch.einsum("rd,sd->rs", self.Z, self.Z) + self.alpha * I
        u = torch.linalg.solve(small, q)
        v = (g - torch.einsum("rd,r->d", self.Z, u)) / self.alpha
        return v
    
    @torch.no_grad()
    def inv(self):
        if self.Z.shape[0] == 0:
            I_d = torch.eye(self.dim, device=self.Z.device, dtype=self.Z.dtype)
            return I_d / self.alpha
        
        r = self.Z.shape[0]
        d = self.Z.shape[1]

        I_r = torch.eye(r, device=self.Z.device, dtype=self.Z.dtype)
        I_d = torch.eye(d, device=self.Z.device, dtype=self.Z.dtype)

        small = torch.einsum("rd,sd->rs", self.Z, self.Z) + self.alpha * I_r
        H = torch.linalg.inv(small)
        return (I_d - torch.einsum("rd,rs,se->de", self.Z, H, self.Z)) / self.alpha
    