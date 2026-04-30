"""
3D Spherical obstacle barrier function.

Barrier: h(x) = ||p - c||² - r² - ε²

where:
    p = position (x, y, z)
    c = obstacle center
    r = obstacle radius
    ε = safety margin

Safe set boundary is at ||p - c|| = sqrt(r² + ε²), matching the reference
quadcopter_multi.py formulation.  Note: this differs slightly from ||p-c|| = r+ε
(which would use -(r+ε)²), but both are valid barrier functions.

For multi-agent systems with n agents:
    - Each agent has its own position p_i
    - One barrier per agent per obstacle
"""

import torch
from .base import BarrierFunction


class SphericalObstacle(BarrierFunction):
    """
    3D spherical obstacle barrier.

    For multi-agent quadrotors:
    - State x contains [agent1_state, agent2_state, ...]
    - Each agent_state = [pos, angles, vel, ang_vel] (12D)
    - Extracts position (first 3D) for each agent
    """

    def __init__(self, center, radius, epsilon, dynamics, n_agent=1):
        """
        Initialize spherical obstacle.

        Args:
            center: Obstacle center [x, y, z]
            radius: Obstacle radius
            epsilon: Safety margin
            dynamics: Quadrotor dynamics object
            n_agent: Number of agents
        """
        # Relative degree 2 for position-based barriers with quadrotor
        super().__init__(relative_degree=dynamics.relative_degree)

        self.center = torch.tensor(center, dtype=torch.float64)
        self.radius = radius
        self.epsilon = epsilon
        self.dynamics = dynamics
        self.n_agent = n_agent

    def extract_positions(self, x):
        """
        Extract positions from multi-agent state.

        Args:
            x: (batch, 12*n_agent)

        Returns:
            positions: (batch, n_agent, 3)
        """
        batch = x.shape[0]
        x_reshaped = x.reshape(batch, self.n_agent, 12)
        return x_reshaped[:, :, 0:3]  # Extract [x, y, z] for each agent

    def extract_velocities(self, x):
        """
        Extract linear velocities from multi-agent state.

        Args:
            x: (batch, 12*n_agent)

        Returns:
            velocities: (batch, n_agent, 3)
        """
        batch = x.shape[0]
        x_reshaped = x.reshape(batch, self.n_agent, 12)
        return x_reshaped[:, :, 6:9]  # Extract [vx, vy, vz] for each agent

    def h(self, x, dynamics=None):
        """
        Barrier function: h(x) = ||p - c||² - (r + ε)²

        Args:
            x: State (batch, 12*n_agent)

        Returns:
            h: Barrier values (batch, n_agent) - one per agent
        """
        positions = self.extract_positions(x)  # (batch, n_agent, 3)
        center = self.center.to(x.device).to(x.dtype)  # (3,)

        # Distance squared from center for each agent
        dp = positions - center.view(1, 1, 3)  # (batch, n_agent, 3)
        dist_sq = (dp ** 2).sum(dim=-1)  # (batch, n_agent)

        # Barrier: positive inside safe region
        # Match reference: h = ||p-c||² - r² - ε²  (NOT -(r+ε)²)
        h_vals = dist_sq - self.radius ** 2 - self.epsilon ** 2

        return h_vals

    def compute_cbf_constraint(self, x, dynamics, alpha):
        """
        Compute CBF constraint parameters for HOCBF (relative degree 2).

        For relative degree 2 barriers:
            ψ₁ = Lf h + α₁ h
            constraint: Lg Lf h · u ≥ -Lf² h - α₂ ψ₁

        Multi-agent: Returns constraints for ALL agents for this obstacle.

        Args:
            x: State (batch, 12*n_agent)
            dynamics: Quadrotor dynamics
            alpha: CBF parameters (alpha1, alpha2) or scalar

        Returns:
            A_cbf: (batch, n_agent, 4*n_agent) - gradient for each agent
            b_cbf: (batch, n_agent) - RHS for each agent
        """
        batch = x.shape[0]

        # Handle alpha parameters
        if isinstance(alpha, (tuple, list)):
            alpha1, alpha2 = alpha
        else:
            alpha1 = alpha2 = alpha

        # Get positions and velocities
        positions = self.extract_positions(x)  # (batch, n_agent, 3)
        velocities = self.extract_velocities(x)  # (batch, n_agent, 3)
        center = self.center.to(x.device).to(x.dtype)

        # Compute h
        dp = positions - center.view(1, 1, 3)
        h_vals = (dp ** 2).sum(dim=-1) - self.radius ** 2 - self.epsilon ** 2

        # Compute Lf h = ∂h/∂x · f(x)
        # ∂h/∂p = 2(p - c), f(x) gives velocity
        Lf_h = 2 * (dp * velocities).sum(dim=-1)  # (batch, n_agent)

        # Compute ψ₁
        psi1 = Lf_h + alpha1 * h_vals  # (batch, n_agent)

        # For Lf² h, we need second derivative
        # Lf² h = 2||v||² + 2(p-c) · a
        # where a = f_drift_accel - g*e3
        vel_sq = (velocities ** 2).sum(dim=-1)  # (batch, n_agent)
        Lf2_h = 2 * vel_sq - 2 * self.dynamics.gravity * dp[:, :, 2]  # (batch, n_agent)

        # Compute Lg Lf h (how control affects Lf h)
        # Lg Lf h = ∂(Lf h)/∂x · g(x)
        # For quadrotor: thrust control affects acceleration
        # Lg Lf h · u = 2(p-c) · ((thrust/m) * thrust_direction)

        # Get thrust directions
        angles = x.reshape(batch, self.n_agent, 12)[:, :, 3:6]  # (batch, n_agent, 3)
        thrust_dir = self.dynamics.thrust_direction(angles)  # (batch, n_agent, 3)

        # Build A_cbf: coefficient of control in constraint
        # Only thrust (first control) matters for position
        A_cbf = torch.zeros(batch, self.n_agent, 4*self.n_agent,
                           device=x.device, dtype=x.dtype)

        for i in range(self.n_agent):
            # Thrust control effect: 2(p-c) · (thrust_dir/m)
            A_cbf[:, i, 4*i] = (2.0 / self.dynamics.mass) * \
                              (dp[:, i, :] * thrust_dir[:, i, :]).sum(dim=-1)

        # RHS: -Lf² h - α₁ Lf h - α₂ ψ₁
        # Full HOCBF derivation: ψ̇₁ + α₂ψ₁ ≥ 0
        #   ψ̇₁ = Lf²h + LgLf h·u + α₁·Lf h   (since Lg h = 0 for rel-deg-2)
        #   → LgLf h·u ≥ -(Lf²h + α₁·Lf h + α₂·ψ₁)
        b_cbf = -Lf2_h - alpha1 * Lf_h - alpha2 * psi1  # (batch, n_agent)

        return A_cbf, b_cbf

    def to(self, device):
        """Move obstacle to device."""
        self.center = self.center.to(device)
        return self

    def __repr__(self):
        return (f"SphericalObstacle(center={self.center.tolist()}, "
                f"radius={self.radius}, epsilon={self.epsilon}, "
                f"n_agent={self.n_agent})")
