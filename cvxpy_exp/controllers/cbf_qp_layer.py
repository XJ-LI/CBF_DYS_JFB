"""
Differentiable CBF-QP layer using CVXPyLayers.

Two modes selectable via ``use_slack``:

Soft (default, use_slack=True):
    minimize:    ||u - u_des||² + p·||s||²
    subject to:  A_i @ u + s_i ≥ b_i  for all obstacles i
                 s_i ≥ 0

Hard (use_slack=False):
    minimize:    ||u - u_des||²
    subject to:  A_i @ u ≥ b_i  for all obstacles i

Works for both:
    - Standard CBF (relative degree 1)
    - HOCBF (relative degree 2)

Slack variables make the QP always feasible. Hard constraints are recovered
in the limit p → ∞ (default p=1e4 is sufficient for typical CBF problems).
Use hard constraints only when you are confident the problem is always feasible
(e.g. single obstacle, well-conditioned A_cbf).
"""

import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


def create_cbf_qp_layer(control_dim, num_constraints, slack_penalty=1e4, use_slack=True):
    """
    Create differentiable CBF-QP layer.

    Soft QP (use_slack=True, default):
        minimize:    ||u - u_desired||² + slack_penalty * ||s||²
        subject to:  A_i @ u + s_i ≥ b_i  for i = 1, ..., num_constraints
                     s_i ≥ 0

    Hard QP (use_slack=False):
        minimize:    ||u - u_desired||²
        subject to:  A_i @ u ≥ b_i  for i = 1, ..., num_constraints

    Args:
        control_dim: Dimension of control input
        num_constraints: Total number of CBF constraints (obstacles × agents)
        slack_penalty: Penalty weight on constraint violation (default 1e4, ignored if use_slack=False)
        use_slack: If True (default), add slack variables for guaranteed feasibility.
                   If False, use hard constraints — will fail if QP is infeasible.

    Returns:
        CvxpyLayer: Differentiable QP solver
    """
    # Decision variables
    u = cp.Variable(control_dim)

    # Parameters
    u_desired = cp.Parameter(control_dim)

    # CBF constraint parameters (one per constraint)
    A_cbf_list = [cp.Parameter(control_dim) for _ in range(num_constraints)]
    b_cbf_list = [cp.Parameter() for _ in range(num_constraints)]

    if use_slack:
        slack = cp.Variable(num_constraints, nonneg=True)
        objective = cp.Minimize(
            cp.sum_squares(u) - 2 * (u_desired @ u)
            + slack_penalty * cp.sum_squares(slack)
        )
        constraints = [
            A_cbf_list[i] @ u + slack[i] >= b_cbf_list[i]
            for i in range(num_constraints)
        ]
        variables = [u, slack]
    else:
        objective = cp.Minimize(
            cp.sum_squares(u) - 2 * (u_desired @ u)
        )
        constraints = [
            A_cbf_list[i] @ u >= b_cbf_list[i]
            for i in range(num_constraints)
        ]
        variables = [u]

    # Create problem
    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp(), "Problem must be DPP (Disciplined Parametrized Programming)"

    # Parameters: [u_desired, A_cbf_1, b_cbf_1, ...]
    parameters = [u_desired] + [param for pair in zip(A_cbf_list, b_cbf_list) for param in pair]

    layer = CvxpyLayer(
        problem,
        parameters=parameters,
        variables=variables,
    )

    return layer


class CBFQPController:
    """
    CBF-QP safety filter for multiple obstacles.

    Takes desired control and projects it to safe control satisfying
    all CBF constraints.
    """

    def __init__(self, dynamics, obstacles, alpha, verbose=False, use_slack=True, slack_penalty=1e4):
        """
        Initialize CBF-QP controller.

        Args:
            dynamics: Dynamics object (provides control_dim, relative_degree)
            obstacles: List of BarrierFunction objects
            alpha: CBF class-K parameter(s)
                  - For relative degree 1: scalar
                  - For relative degree 2: tuple (alpha1, alpha2) or scalar
            use_slack: If True (default), use soft constraints (always feasible).
                       If False, use hard constraints — may raise if QP is infeasible.
            slack_penalty: Penalty weight on slack (ignored when use_slack=False).
        """
        self.dynamics = dynamics
        self.obstacles = obstacles
        self.alpha = alpha
        self.num_obstacles = len(obstacles)
        self.verbose = verbose
        self.use_slack = use_slack

        # Determine total number of constraints (for multi-agent, multiply by n_agent)
        # Check if obstacles have n_agent attribute (multi-agent case)
        if hasattr(obstacles[0], 'n_agent'):
            self.n_agent = obstacles[0].n_agent
            self.total_constraints = self.num_obstacles * self.n_agent
        else:
            self.n_agent = 1
            self.total_constraints = self.num_obstacles

        # Create QP layer
        self.qp_layer = create_cbf_qp_layer(
            dynamics.control_dim, self.total_constraints,
            slack_penalty=slack_penalty, use_slack=use_slack,
        )

        # Verify all obstacles have same relative degree as dynamics
        for obs in obstacles:
            if obs.relative_degree != dynamics.relative_degree:
                raise ValueError(
                    f"Obstacle relative degree {obs.relative_degree} "
                    f"does not match dynamics relative degree {dynamics.relative_degree}"
                )

    def filter_control(self, x, u_desired):
        """
        Filter desired control through CBF safety constraints.

        Args:
            x: Current state (batch_size, state_dim)
            u_desired: Desired control from policy (batch_size, control_dim)

        Returns:
            u_safe: Safe control (batch_size, control_dim)
        """
        batch_size = x.shape[0]

        # IMPORTANT: CVXPyLayers differentiation ONLY works with SCS solver!
        # eps=1e-3: SCS convergence is the bottleneck, not gradient accuracy (IFT
        #   is already an approximation).  1e-3 avoids "Solved/Inaccurate" reliably.
        # max_iters=10000: sufficient for 20-var/15-constraint QP at eps=1e-3.
        solver_args = {
            "solve_method": "SCS",
            "eps": 1e-3,
            "max_iters": 1000,
            "acceleration_lookback": 0,
        }
        # Compute CBF constraint parameters for each obstacle
        A_cbf_list = []
        b_cbf_list = []

        for obstacle in self.obstacles:
            A_cbf, b_cbf = obstacle.compute_cbf_constraint(x, self.dynamics, self.alpha)

            # Handle multi-agent constraints: flatten (batch, n_agent, ...) into separate constraints
            if A_cbf.dim() == 3: 
                # Multi-agent: unpack per-agent rows
                for i in range(A_cbf.shape[1]):
                    A_i = A_cbf[:, i, :]  # (batch, 4*n_agent)
                    b_i = b_cbf[:, i]     # (batch,)
 
                    # ── Degeneracy warning ──────────────────────────────────
                    if self.verbose:
                        # Check the diagonal block for agent i
                        ctrl_per_agent = A_cbf.shape[2] // A_cbf.shape[1]
                        coeff = A_cbf[:, i, i * ctrl_per_agent:(i + 1) * ctrl_per_agent]
                        if coeff.abs().max().item() < 1e-4:
                            print(
                                f"[CBF] WARNING: A_cbf degenerate for agent {i}, "
                                f"obstacle {obstacle}. max|coeff|="
                                f"{coeff.abs().max().item():.2e}"
                            )
 
                    A_cbf_list.append(A_i)
                    b_cbf_list.append(b_i)
            else:  # Single-agent case: (batch, control_dim)
                A_cbf_list.append(A_cbf)
                b_cbf_list.append(b_cbf)

        # Do NOT manually normalize constraints — dividing b by a near-zero A_norm
        # (clamped to 1e-4) turns benign slack constraints (b ≈ −55) into extreme
        # values (b ≈ −550,000).  SCS then sees RHS values spanning six orders of
        # magnitude, fails to converge, and returns "Solved/Inaccurate" with garbage
        # u_safe.  SCS has its own internal scaling; pass raw constraints.
        device = u_desired.device
        qp_params = [u_desired.cpu()]
        for A, b in zip(A_cbf_list, b_cbf_list):
            qp_params.append(A.cpu())
            qp_params.append(b.cpu())

        # Solve QP (on CPU)
        result = self.qp_layer(*qp_params, solver_args=solver_args)
        u_safe = result[0].to(device)   # result[0] = u; result[1] = slack (soft mode only)

        return u_safe

    def __repr__(self):
        if self.n_agent > 1:
            return (f"CBFQPController("
                    f"dynamics={self.dynamics.__class__.__name__}, "
                    f"n_agent={self.n_agent}, "
                    f"num_obstacles={self.num_obstacles}, "
                    f"total_constraints={self.total_constraints}, "
                    f"alpha={self.alpha})")
        else:
            return (f"CBFQPController("
                    f"dynamics={self.dynamics.__class__.__name__}, "
                    f"num_obstacles={self.num_obstacles}, "
                    f"alpha={self.alpha})")
