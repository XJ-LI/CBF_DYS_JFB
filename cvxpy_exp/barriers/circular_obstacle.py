"""
Circular obstacle barrier function.

Works for both single and double integrator systems by automatically
adapting to the relative degree of the dynamics.

Barrier function:
    h(p) = ||p - center||² - r² - ε

Safe set: {x : h(p) ≥ 0} (stay outside circle of radius r around center)

For single integrator (relative degree 1):
    - Standard CBF constraint

For double integrator (relative degree 2):
    - HOCBF constraint with auxiliary barrier
"""

import torch
from .base import RelativeDegree1Barrier, RelativeDegree2Barrier


class CircularObstacle:
    """
    Factory for circular obstacle barriers.

    Automatically creates the correct barrier type based on dynamics and n_agent.
    """

    def __new__(cls, center, radius, epsilon=0.1, dynamics=None, n_agent=1):
        """
        Create circular obstacle barrier matching dynamics relative degree.

        Args:
            center: Obstacle center (tensor or list) — always 2D spatial coords
            radius: Obstacle radius (including safety margin)
            epsilon: Small constant for numerical stability
            dynamics: Dynamics object (determines relative degree)
            n_agent: Number of agents (>1 returns multi-agent variant)

        Returns:
            CircularObstacle1, CircularObstacle2, or MultiAgentCircularObstacle2 instance
        """
        if dynamics is None:
            return CircularObstacle1(center, radius, epsilon)

        if dynamics.relative_degree == 1:
            if n_agent > 1:
                return MultiAgentCircularObstacle1(center, radius, epsilon, n_agent=n_agent)
            return CircularObstacle1(center, radius, epsilon)
        elif dynamics.relative_degree == 2:
            if n_agent > 1:
                return MultiAgentCircularObstacle2(center, radius, epsilon, n_agent=n_agent)
            return CircularObstacle2(center, radius, epsilon)
        else:
            raise ValueError(f"Unsupported relative degree: {dynamics.relative_degree}")


class CircularObstacle1(RelativeDegree1Barrier):
    """
    Circular obstacle for relative degree 1 (single integrator).

    h(x) = ||x - center||² - r² - ε
    ∇h = 2(x - center)

    For single integrator: ẋ = u
        Lf h = 0 (no drift)
        Lg h = ∇h = 2(x - center)

    CBF constraint: 2(x - center) · u ≥ -α h
    """

    def __init__(self, center, radius, epsilon=0.1):
        """
        Initialize circular obstacle barrier.

        Args:
            center: Obstacle center position (tensor or list)
            radius: Total clearance radius (obstacle + safety margin)
            epsilon: Numerical stability constant
        """
        super().__init__()
        self.center = torch.as_tensor(center) if not isinstance(center, torch.Tensor) else center
        self.radius = radius
        self.epsilon = epsilon

    def h(self, x, dynamics=None):
        """
        Evaluate barrier function.

        Args:
            x: State (batch_size, state_dim)
            dynamics: Optional (not used for single integrator)

        Returns:
            h: Barrier value (batch_size,)
        """
        # Ensure center is on same device
        center = self.center.to(x.device).to(x.dtype)

        dist_sq = torch.sum((x - center)**2, dim=-1)
        return dist_sq - self.radius**2 - self.epsilon

    def compute_lie_derivatives(self, x, dynamics):
        """
        Compute Lie derivatives for single integrator.

        Args:
            x: State (batch_size, state_dim)
            dynamics: SingleIntegrator object

        Returns:
            Lf_h: Zero (batch_size,)
            Lg_h: 2(x - center) (batch_size, control_dim)
            h_val: Barrier value (batch_size,)
        """
        center = self.center.to(x.device).to(x.dtype)

        # Barrier value
        h_val = self.h(x, dynamics)

        # Gradient: ∇h = 2(x - center)
        grad_h = 2.0 * (x - center)

        # For single integrator: f(x) = 0, g(x) = I
        Lf_h = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        Lg_h = grad_h  # ∇h · I = ∇h

        return Lf_h, Lg_h, h_val

    def __repr__(self):
        return (f"CircularObstacle1(center={self.center.tolist()}, "
                f"radius={self.radius}, epsilon={self.epsilon})")


class CircularObstacle2(RelativeDegree2Barrier):
    """
    Circular obstacle for relative degree 2 (double integrator).

    h₀(p) = ||p - center||² - r² - ε

    For double integrator x = [p, v], ẍ = a:
        ḣ₀ = 2(p - center) · v
        ḧ₀ = 2||v||² + 2(p - center) · a

    Auxiliary barrier:
        h₁ = ḣ₀ + α₁ h₀

    HOCBF constraint:
        2(p - center) · a ≥ -2||v||² - α₂(ḣ₀ + α₁ h₀)
    """

    def __init__(self, center, radius, epsilon=0.1):
        """
        Initialize circular obstacle barrier for double integrator.

        Args:
            center: Obstacle center position (tensor or list)
            radius: Total clearance radius (obstacle + safety margin)
            epsilon: Numerical stability constant
        """
        super().__init__()
        self.center = torch.as_tensor(center) if not isinstance(center, torch.Tensor) else center
        self.radius = radius
        self.epsilon = epsilon

    def h(self, x, dynamics):
        """
        Evaluate barrier function on position.

        Args:
            x: State [p, v] (batch_size, 2*dim)
            dynamics: DoubleIntegrator object

        Returns:
            h: Barrier value (batch_size,)
        """
        # Extract position
        p = dynamics.position(x)

        # Ensure center is on same device
        center = self.center.to(x.device).to(x.dtype)

        dist_sq = torch.sum((p - center)**2, dim=-1)
        return dist_sq - self.radius**2 - self.epsilon

    def compute_hocbf_terms(self, x, dynamics):
        """
        Compute HOCBF terms for double integrator.

        For h₀(p) = ||p - c||² - r² - ε:
            ∇h₀ = 2(p - c)
            ḣ₀ = 2(p - c) · v
            ḧ₀ = 2||v||² + 2(p - c) · a

        Args:
            x: State [p, v] (batch_size, 2*dim)
            dynamics: DoubleIntegrator object

        Returns:
            h0: Barrier value (batch_size,)
            h0_dot: First derivative (batch_size,)
            Lf2_h0: Second Lie derivative without control (batch_size,)
            Lg_Lf_h0: Control gradient (batch_size, control_dim)
        """
        p = dynamics.position(x)
        v = dynamics.velocity(x)

        center = self.center.to(x.device).to(x.dtype)

        # h₀(p)
        dist_sq = torch.sum((p - center)**2, dim=-1)
        h0 = dist_sq - self.radius**2 - self.epsilon

        # Gradient ∇h₀ = 2(p - center)
        grad_h0 = 2.0 * (p - center)

        # ḣ₀ = ∇h₀ · v
        h0_dot = torch.sum(grad_h0 * v, dim=-1)

        # ḧ₀ = 2||v||² + 2(p - center) · a
        #    = Lf²h₀ + Lg Lf h₀ · a

        # Lf²h₀ = 2||v||² (drift contribution to second derivative)
        Lf2_h0 = 2.0 * torch.sum(v**2, dim=-1)

        # Lg Lf h₀ = 2(p - center) (control gradient)
        Lg_Lf_h0 = grad_h0

        return h0, h0_dot, Lf2_h0, Lg_Lf_h0

    def __repr__(self):
        return (f"CircularObstacle2(center={self.center.tolist()}, "
                f"radius={self.radius}, epsilon={self.epsilon})")


class MultiAgentCircularObstacle1(RelativeDegree1Barrier):
    """
    Circular obstacle for N agents each running single integrator dynamics.

    State layout (SingleIntegrator(dim=N*spatial_dim)):
        x = [p1, p2, ..., pN]   each pi ∈ R^spatial_dim

    Control layout:
        u = [u1, u2, ..., uN]   each ui ∈ R^spatial_dim

    For each agent i:
        h_i = ||p_i - c||² - r² - ε
        CBF constraint: 2(p_i - c) · u_i ≥ -α h_i

    A_cbf[i] is zero everywhere except at agent i control slots (block-diagonal).
    Total constraints = n_agent (one per agent per obstacle).
    """

    def __init__(self, center, radius, epsilon=0.1, n_agent=2, spatial_dim=2):
        super().__init__()
        self.center = torch.as_tensor(center) if not isinstance(center, torch.Tensor) else center
        self.radius = radius
        self.epsilon = epsilon
        self.n_agent = n_agent
        self.spatial_dim = spatial_dim

    def h(self, x, dynamics=None):
        """Barrier values per agent: (batch, n_agent)."""
        N, sdim = self.n_agent, self.spatial_dim
        pos = x[:, :N * sdim].reshape(x.shape[0], N, sdim)
        center = self.center.to(x.device).to(x.dtype).view(1, 1, sdim)
        return ((pos - center) ** 2).sum(-1) - self.radius ** 2 - self.epsilon

    def compute_cbf_constraint(self, x, dynamics, alpha):
        """
        Returns:
            A_cbf: (batch, n_agent, n_agent*spatial_dim) — block-diagonal
            b_cbf: (batch, n_agent)
        """
        batch = x.shape[0]
        N, sdim = self.n_agent, self.spatial_dim

        alpha_val = float(alpha[0]) if isinstance(alpha, (tuple, list)) else float(alpha)

        pos = x[:, :N * sdim].reshape(batch, N, sdim)
        center = self.center.to(x.device).to(x.dtype).view(1, 1, sdim)
        dp = pos - center

        h = (dp ** 2).sum(-1) - self.radius ** 2 - self.epsilon  # (batch, N)

        # Block-diagonal A_cbf: agent i's row is non-zero only at its own control cols
        A_cbf = torch.zeros(batch, N, N * sdim, dtype=x.dtype, device=x.device)
        for i in range(N):
            A_cbf[:, i, i * sdim:(i + 1) * sdim] = 2.0 * dp[:, i, :]

        b_cbf = -alpha_val * h  # (batch, N)

        return A_cbf, b_cbf

    def __repr__(self):
        return (f"MultiAgentCircularObstacle1(center={self.center.tolist()}, "
                f"radius={self.radius}, epsilon={self.epsilon}, "
                f"n_agent={self.n_agent}, spatial_dim={self.spatial_dim})")


class MultiAgentCircularObstacle2(RelativeDegree2Barrier):
    """
    Circular obstacle for N agents each running double integrator dynamics.

    State layout (from DoubleIntegrator(dim=N*2)):
        x = [p1, p2, ..., pN,  v1, v2, ..., vN]
            each pi, vi ∈ R^spatial_dim (default 2)

    Control layout:
        u = [a1, a2, ..., aN]
            each ai ∈ R^spatial_dim

    For each agent i:
        h_i  = ||p_i - c||² - r² - ε
        Lf h_i  = 2(p_i - c) · v_i
        Lf²h_i  = 2||v_i||²                   (no drift acceleration)
        LgLf h_i = 2(p_i - c)  w.r.t. a_i only

    A_cbf[i] is zero everywhere except at the agent i control slots:
        A_cbf[i, i*sdim:(i+1)*sdim] = 2(p_i - c)
    """

    def __init__(self, center, radius, epsilon=0.1, n_agent=2, spatial_dim=2):
        super().__init__()
        self.center = torch.as_tensor(center) if not isinstance(center, torch.Tensor) else center
        self.radius = radius
        self.epsilon = epsilon
        self.n_agent = n_agent
        self.spatial_dim = spatial_dim  # spatial dims per agent (2 for 2D)

    def _extract(self, x):
        """Split flat state into per-agent positions and velocities."""
        sdim = self.spatial_dim
        N = self.n_agent
        # x: (batch, 2*N*sdim)
        # positions: columns 0 : N*sdim, shape (batch, N, sdim)
        pos = x[:, :N * sdim].reshape(x.shape[0], N, sdim)
        # velocities: columns N*sdim : end, shape (batch, N, sdim)
        vel = x[:, N * sdim:].reshape(x.shape[0], N, sdim)
        return pos, vel

    def h(self, x, dynamics=None):
        """Barrier values per agent: (batch, n_agent)."""
        pos, _ = self._extract(x)
        center = self.center.to(x.device).to(x.dtype).view(1, 1, self.spatial_dim)
        dp = pos - center
        return (dp ** 2).sum(dim=-1) - self.radius ** 2 - self.epsilon

    # compute_hocbf_terms is not used: this class overrides compute_cbf_constraint
    # directly (block-diagonal structure) rather than going through the base class
    # machinery that would call compute_hocbf_terms per-agent sequentially.

    def compute_cbf_constraint(self, x, dynamics, alpha):
        """
        Returns:
            A_cbf: (batch, n_agent, n_agent*spatial_dim)  — block-diagonal
            b_cbf: (batch, n_agent)
        """
        batch = x.shape[0]
        N = self.n_agent
        sdim = self.spatial_dim
        control_dim = N * sdim

        if isinstance(alpha, (int, float)):
            alpha1 = alpha2 = float(alpha)
        else:
            alpha1, alpha2 = alpha

        pos, vel = self._extract(x)   # (batch, N, sdim) each
        center = self.center.to(x.device).to(x.dtype).view(1, 1, sdim)

        dp = pos - center                                   # (batch, N, sdim)
        h0 = (dp ** 2).sum(dim=-1) - self.radius ** 2 - self.epsilon  # (batch, N)

        Lf_h = 2.0 * (dp * vel).sum(dim=-1)               # (batch, N)
        psi1 = Lf_h + alpha1 * h0                          # (batch, N)
        Lf2_h = 2.0 * (vel ** 2).sum(dim=-1)              # (batch, N)  no drift term

        # A_cbf: block-diagonal — agent i's constraint uses only agent i's control
        A_cbf = torch.zeros(batch, N, control_dim, dtype=x.dtype, device=x.device)
        for i in range(N):
            A_cbf[:, i, i * sdim:(i + 1) * sdim] = 2.0 * dp[:, i, :]  # (batch, sdim)

        b_cbf = -(Lf2_h + alpha1 * Lf_h + alpha2 * psi1)  # (batch, N)

        return A_cbf, b_cbf

    def __repr__(self):
        return (f"MultiAgentCircularObstacle2(center={self.center.tolist()}, "
                f"radius={self.radius}, epsilon={self.epsilon}, "
                f"n_agent={self.n_agent}, spatial_dim={self.spatial_dim})")
