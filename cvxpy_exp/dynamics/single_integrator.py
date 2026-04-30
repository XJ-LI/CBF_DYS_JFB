"""
Single Integrator Dynamics: ẋ = u

State: x ∈ R^n (position)
Control: u ∈ R^n (velocity)

Control-affine form:
    f(x) = 0
    g(x) = I

Relative degree for position barriers: 1
    - h(x) = barrier on position
    - ḣ = ∇h · ẋ = ∇h · u (control appears in first derivative)
    - Standard CBF applies
"""

import torch
from .base import ControlAffineDynamics


class SingleIntegrator(ControlAffineDynamics):
    """
    Single integrator dynamics: ẋ = u

    Simplest control-affine system with relative degree 1.
    """

    def __init__(self, dim=2):
        """
        Initialize single integrator.

        Args:
            dim: Dimension of position space (default: 2 for planar motion)
        """
        super().__init__(
            state_dim=dim,
            control_dim=dim,
            relative_degree=1  # Standard CBF
        )
        self.dim = dim

    def f(self, x, t=0.0):
        """
        Drift dynamics: f(x, t) = 0  (time-invariant)

        Args:
            x: State (batch_size, dim)
            t: Current time (unused; accepted for interface consistency)

        Returns:
            f: Zero drift (batch_size, dim)
        """
        batch_size = x.shape[0]
        return torch.zeros(batch_size, self.dim, dtype=x.dtype, device=x.device)

    def g(self, x, t=0.0):
        """
        Control matrix: g(x, t) = I  (time-invariant)

        Args:
            x: State (batch_size, dim)
            t: Current time (unused; accepted for interface consistency)

        Returns:
            g: Identity matrix (batch_size, dim, dim)
        """
        batch_size = x.shape[0]
        I = torch.eye(self.dim, dtype=x.dtype, device=x.device)
        return I.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, dim, dim)

    def step(self, x, u, dt, t=0.0):
        """
        RK4 integration (matches reference rk4_step in utils.py).

        For single integrator f(x)=0, g(x)=I so all RK4 stages equal u,
        giving x_next = x + dt*u — same result as Euler, but kept as RK4
        for consistency with the reference code.

        Args:
            x: Current position (batch_size, dim)
            u: Control velocity (batch_size, dim)
            dt: Time step
            t: Current time (unused; accepted for interface consistency)

        Returns:
            x_next: Next position (batch_size, dim)
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

    def position(self, x):
        """
        Extract position from state (for single integrator, state = position).

        Args:
            x: State (batch_size, dim)

        Returns:
            p: Position (batch_size, dim)
        """
        return x
