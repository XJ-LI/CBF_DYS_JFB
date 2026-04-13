import torch
import torch.nn as nn
import numpy as np

# DYS Projection 
class DYSProjector(nn.Module):
    """
    Differentiable per-sample projector:
        u*_j = argmin_u  ½‖u − u_nom_j‖²   s.t.  A_j u ≤ b_j   ∀ j in batch

    A and b are passed at call time so the same instance can be reused
    across time steps with changing constraints.

    ── Fixed-point equation ───────────────────────────────────────────────────
        z* = T(z*; u_nom, A, b)    where T is one DYS step (see apply_T)

    ── JFB forward pass ───────────────────────────────────────────────────────
        1. Run  z ← T(z)  to convergence inside torch.no_grad()
        2. Apply one final T WITH gradient  (JFB approximation)
    """
    def __init__(self, alpha: float = 0.5, sv_thresh: float = 1e-6, grad_mode: str = "jfb"):
        super().__init__()
        self.alpha     = alpha
        self.sv_thresh = sv_thresh
        self.grad_mode = grad_mode
    
    def _build_matrices(self, A, b):
        batch, m, n = A.shape
        I_m   = torch.eye(m, dtype=A.dtype, device=A.device).unsqueeze(0).expand(batch, -1, -1)
        A_std = torch.cat([A, -A, I_m], dim=2)
        M = torch.cat([
             torch.eye(n,  dtype=A.dtype, device=A.device),
            -torch.eye(n,  dtype=A.dtype, device=A.device),
             torch.zeros(n, m, dtype=A.dtype, device=A.device),
        ], dim=1)
        AAt      = A_std @ A_std.transpose(-2, -1)
        U, S, _  = torch.linalg.svd(AAt)
        S_inv    = torch.where(S > self.sv_thresh, 1.0 / S, torch.zeros_like(S))
        AAt_pinv = (U * S_inv.unsqueeze(-2)) @ U.transpose(-2, -1)
        P_perp   = A_std.transpose(-2, -1) @ AAt_pinv
        return A_std, M, P_perp
    
    def proj_C1(self, v, A_std, P_perp, b):
        """Project onto the affine solution set  A_std x = b."""
        resid = A_std @ v.unsqueeze(-1) - b
        return v - (P_perp @ resid).squeeze(-1)
    def proj_C2(self, z):
        """Project onto the non-negativity set: clamp slack variable to be >= 0."""
        return torch.clamp(z, min=0)
    
    def apply_T(self, z, u_nom, A_std, M, P_perp, b):
        x    = self.proj_C2(z)
        grad = (x @ M.T - u_nom) @ M
        v    = 2 * x - z - self.alpha * grad
        y    = self.proj_C1(v, A_std, P_perp, b)
        return z + y - x
    
    def forward(self, u_nom, A, b, z0=None,
                max_iter=2000, tol=1e-2, verbose=False, n_grad_iters=1):
        """
        Args:
            u_nom    : (batch, n)      nominal control
            A        : (batch, m, n)   per-sample constraint matrix
            b        : (batch, m, 1)   per-sample RHS
            z0       : (batch, N)      optional warm start
            max_iter : int
            tol      : float
            verbose  : bool
            n_grad_iters : int        number of gradient iterations (JFB equivalent)
        Returns:
            u_star : (batch, n)
            z_star : (batch, N)
            info   : dict
        """
        batch, m, n = A.shape
        N = 2 * n + m
        A_std, M, P_perp = self._build_matrices(A, b)
        z = (z0 if z0 is not None else torch.zeros(batch, N, dtype=u_nom.dtype, device=u_nom.device))
        residuals = []
        converged = False
        k = 0
        if self.grad_mode == "jfb":
            with torch.no_grad():
                for k in range(max_iter - 1):
                    z_new = self.apply_T(z, u_nom, A_std, M, P_perp, b)
                    res   = (z_new - z).abs().max().item()
                    residuals.append(res)
                    z = z_new
                    if verbose:
                        print(f"  iter {k+1:4d}  |  max residual = {res:.2e}")
                    if res < tol:
                        converged = True
                        break
            # JFB step
            for _ in range(n_grad_iters):
                z = self.apply_T(z, u_nom, A_std, M, P_perp, b)
        elif self.grad_mode == "ad":
            for k in range(max_iter):
                z_new = self.apply_T(z, u_nom, A_std, M, P_perp, b)
                res = (z_new - z).abs().max().item()
                residuals.append(res)
                z = z_new
                if verbose:
                    print(f"  iter {k+1:4d}  |  max residual = {res:.2e}")
                if res < tol:
                    converged = True
                    break
        else:
            raise ValueError(f"Unknown grad_mode: {self.grad_mode}")
        z_star = z
        u_star = z_star @ M.T
        info = dict(iters=k + 1, converged=converged,
                    final_residual=residuals[-1] if residuals else 0.0,
                    residuals=residuals)
        return u_star, z_star, info
        

def euler_step(z, u, ti, h, f):
    return z + h * f(z, u, ti) 
        
def rk4_step(z, u, ti, h, f):
    """One RK4 step with control u held constant over [ti, ti+h]."""
    k1 = f(z, u, ti)
    k2 = f(z + (h/2) * k1,  u, ti + h/2)
    k3 = f(z + (h/2) * k2,  u, ti + h/2)
    k4 = f(z + h * k3, u, ti + h)
    return z + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
    
#### Neural Network Definition
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.net(x))

class ControlNet(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, output_dim=2, n_blocks=3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks     = nn.Sequential(*[ResBlock(hidden_dim) for _ in range(n_blocks)])
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, t):
        t_batch = t.view(-1, 1).expand(x.shape[0], -1)   # (batch, 1)
        xt = torch.cat([x, t_batch], dim=-1)              # (batch, input_dim)
        h  = self.input_proj(xt)                          # (batch, hidden_dim)
        h  = self.blocks(h)                               # (batch, hidden_dim)
        return self.output_proj(h)                        # (batch, output_dim)