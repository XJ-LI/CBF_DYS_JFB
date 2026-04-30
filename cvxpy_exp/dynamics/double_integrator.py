"""
Double Integrator Dynamics: ẍ = a

State: x = [p, v] ∈ R^(2n)
    p ∈ R^n: position
    v ∈ R^n: velocity
Control: a ∈ R^n (acceleration)

Control-affine form:
    ẋ = f(x) + g(x)a

    f(x) = [v]    g(x) = [0]
           [0]           [I]

Relative degree for position barriers: 2 (requires HOCBF)
    - h₀(p) = barrier on position
    - ḣ₀ = ∇h₀ · v (no control yet)
    - ḧ₀ = ∇h₀ · a + v^T H_h₀ v (control appears in second derivative)
    - Need Higher-Order CBF (HOCBF)

HOCBF formulation:
    h₀(p) = ||p - c||² - r²          (position barrier)
    h₁ = ḣ₀ + α₁(h₀)                 (auxiliary barrier)
    ḧ₀ + α₂(h₁) ≥ 0                  (HOCBF condition with control)
"""

import torch
from .base import ControlAffineDynamics


class DoubleIntegrator(ControlAffineDynamics):
    """
    Double integrator dynamics: ẍ = a

    State x = [p, v] where:
        - p: position (first half of state)
        - v: velocity (second half of state)

    Requires HOCBF (Higher-Order CBF) with relative degree 2.
    """

    def __init__(self, dim=2):
        """
        Initialize double integrator.

        Args:
            dim: Dimension of position/velocity space (default: 2 for planar motion)
        """
        super().__init__(
            state_dim=2 * dim,  # [position, velocity]
            control_dim=dim,     # acceleration
            relative_degree=2    # HOCBF required
        )
        self.dim = dim  # dimension of position/velocity/control

    def f(self, x, t=0.0):
        """
        Drift dynamics: f(x, t) = [v, 0]  (time-invariant)

        Args:
            x: State [p, v] (batch_size, 2*dim)
            t: Current time (unused; accepted for interface consistency)

        Returns:
            f: Drift vector (batch_size, 2*dim)
        """
        batch_size = x.shape[0]
        p, v = self.split_state(x)

        # ṗ = v, v̇ = 0 (without control)
        f_p = v
        f_v = torch.zeros(batch_size, self.dim, dtype=x.dtype, device=x.device)

        return torch.cat([f_p, f_v], dim=-1)

    def g(self, x, t=0.0):
        """
        Control matrix: g(x, t) = [0, I]^T  (time-invariant)

        Args:
            x: State [p, v] (batch_size, 2*dim)
            t: Current time (unused; accepted for interface consistency)

        Returns:
            g: Control matrix (batch_size, 2*dim, dim)
        """
        batch_size = x.shape[0]

        # ṗ = v (no control), v̇ = a (control)
        g_p = torch.zeros(batch_size, self.dim, self.dim, dtype=x.dtype, device=x.device)
        g_v = torch.eye(self.dim, dtype=x.dtype, device=x.device).unsqueeze(0).expand(batch_size, -1, -1)

        # Stack: g = [g_p; g_v] has shape (batch, 2*dim, dim)
        return torch.cat([g_p, g_v], dim=1)

    def step(self, x, u, dt, t=0.0):
        """
        RK4 integration (constant u over interval).

        For double integrator with constant u:
            v_next = v + dt * u
            p_next = p + dt * v + 0.5 * dt^2 * u  (exact ZOH)

        Args:
            x: Current state [p, v] (batch_size, 2*dim)
            u: Acceleration control (batch_size, dim)
            dt: Time step
            t: Current time (unused; accepted for interface consistency)

        Returns:
            x_next: Next state [p_next, v_next] (batch_size, 2*dim)
        """
        def dynamics(xi):
            f_x = self.f(xi)
            g_x = self.g(xi)
            return f_x + (g_x @ u.unsqueeze(-1)).squeeze(-1)

        k1 = dynamics(x)
        k2 = dynamics(x + 0.5 * dt * k1)
        k3 = dynamics(x + 0.5 * dt * k2)
        k4 = dynamics(x + dt * k3)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def split_state(self, x):
        """
        Split state into position and velocity.

        Args:
            x: State (batch_size, 2*dim)

        Returns:
            p: Position (batch_size, dim)
            v: Velocity (batch_size, dim)
        """
        p = x[..., :self.dim]
        v = x[..., self.dim:]
        return p, v

    def position(self, x):
        """
        Extract position from state.

        Args:
            x: State [p, v] (batch_size, 2*dim)

        Returns:
            p: Position (batch_size, dim)
        """
        return x[..., :self.dim]

    def velocity(self, x):
        """
        Extract velocity from state.

        Args:
            x: State [p, v] (batch_size, 2*dim)

        Returns:
            v: Velocity (batch_size, dim)
        """
        return x[..., self.dim:]
