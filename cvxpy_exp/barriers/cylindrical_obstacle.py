"""
Cylindrical obstacle barrier for 3D multi-agent single integrator.

Barrier acts only in the XY plane (cylinder extends infinitely in Z):
    h_i(p) = (p_ix - cx)² + (p_iy - cy)² - r² - ε²

Safe set: {p : h_i(p) ≥ 0}  (outside cylinder of radius r)

For 3D single integrator (ẋ = u, u ∈ R³):
    Lf h = 0  (no drift)
    Lg h = ∂h/∂p = [2(p_x-cx), 2(p_y-cy), 0]  (z has no effect)

CBF constraint per agent i:
    2(p_i - c)_xy · u_i_xy ≥ -α h_i
"""

import torch
from .base import RelativeDegree1Barrier


class MultiAgentCylindricalObstacle1(RelativeDegree1Barrier):
    """
    Cylindrical obstacle for N agents running 3D single integrator dynamics.

    State layout (SingleIntegrator(dim=N*3)):
        x = [p1, p2, ..., pN]   each pi ∈ R³

    Barrier acts in XY plane only (z component of gradient is zero).
    Epsilon convention matches reference: uses ε² (not ε) for numerical margin.
    """

    def __init__(self, center_xy, radius, epsilon=0.1, n_agent=50, spatial_dim=3):
        super().__init__()
        center = torch.as_tensor(center_xy, dtype=torch.float64)
        self.center = center[:2]  # only xy
        self.radius = radius
        self.epsilon = epsilon
        self.n_agent = n_agent
        self.spatial_dim = spatial_dim  # 3 for 3D

    def h(self, x, dynamics=None):
        """Barrier values per agent: (batch, n_agent)."""
        N, sdim = self.n_agent, self.spatial_dim
        pos    = x[:, :N * sdim].reshape(x.shape[0], N, sdim)
        pos_xy = pos[:, :, :2]
        center = self.center.to(x.device).to(x.dtype).view(1, 1, 2)
        dxy    = pos_xy - center
        return (dxy ** 2).sum(-1) - self.radius ** 2 - self.epsilon ** 2

    def compute_cbf_constraint(self, x, dynamics, alpha):
        """
        Returns:
            A_cbf: (batch, n_agent, n_agent*spatial_dim) — block-diagonal, sparse
            b_cbf: (batch, n_agent)
        """
        batch      = x.shape[0]
        N, sdim    = self.n_agent, self.spatial_dim
        alpha_val  = float(alpha[0]) if isinstance(alpha, (tuple, list)) else float(alpha)

        pos    = x[:, :N * sdim].reshape(batch, N, sdim)
        pos_xy = pos[:, :, :2]
        center = self.center.to(x.device).to(x.dtype).view(1, 1, 2)
        dxy    = pos_xy - center                                      # (batch, N, 2)
        h      = (dxy ** 2).sum(-1) - self.radius ** 2 - self.epsilon ** 2  # (batch, N)

        # Block-diagonal A: agent i's row is non-zero only at its own control cols
        A_cbf = torch.zeros(batch, N, N * sdim, dtype=x.dtype, device=x.device)
        for i in range(N):
            A_cbf[:, i, i * sdim    ] = 2.0 * dxy[:, i, 0]   # ∂h/∂p_x
            A_cbf[:, i, i * sdim + 1] = 2.0 * dxy[:, i, 1]   # ∂h/∂p_y
            # z component = 0 (cylinder extends infinitely in z)

        b_cbf = -alpha_val * h   # (batch, N)
        return A_cbf, b_cbf

    def __repr__(self):
        return (f"MultiAgentCylindricalObstacle1(center={self.center.tolist()}, "
                f"radius={self.radius}, epsilon={self.epsilon}, n_agent={self.n_agent})")
