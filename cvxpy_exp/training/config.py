"""
Unified configuration system for CBF training.

Works for any combination of dynamics and obstacles — simple integrators
or complex multi-agent systems like quadrotors.
"""

import torch
from dataclasses import dataclass, field
from typing import List, Optional, Any


@dataclass
class TrainingConfig:
    """
    Configuration for CBF-based control training.

    Two usage modes:

    Simple (integrators): pass dynamics_type string; trainer builds everything.

        config = TrainingConfig(dynamics_type='double_integrator', ...)

    Advanced (quadrotor, etc.): pass pre-built objects and optional hooks.

        config = TrainingConfig(
            dynamics=my_dynamics,
            obstacles_list=my_obstacles,
            output_transform=my_transform_fn,
            cost_fn=my_cost_fn,
            ...
        )
    """

    # ========================================================================
    # Problem Setup
    # ========================================================================
    initial_state: List[float] = field(default_factory=lambda: [0.0, 0.0])
    target_state: List[float] = field(default_factory=lambda: [2.0, 2.0])

    # ========================================================================
    # Dynamics — pass object OR type string (object takes priority)
    # ========================================================================
    dynamics: Optional[Any] = None          # pre-built dynamics object
    dynamics_type: Optional[str] = None     # 'single_integrator' or 'double_integrator'
    position_dim: int = 2

    # ========================================================================
    # Obstacles — pass objects OR dicts (objects take priority)
    # ========================================================================
    obstacles_list: Optional[List[Any]] = None  # pre-built obstacle objects
    obstacles: List[dict] = field(default_factory=lambda: [
        {'center': [1.0, 1.0], 'radius': 0.6, 'epsilon': 0.1}
    ])

    # ========================================================================
    # Hooks — callables for system-specific behavior
    # ========================================================================
    # fn(u_raw) -> u_desired  (e.g. tanh scaling + hover bias for quadrotor)
    output_transform: Optional[Any] = None

    # fn(states, controls, alpha_terminal) -> (total_loss, running, terminal)
    # states: (batch, num_steps+1, state_dim), controls: (batch, num_steps, control_dim)
    # If None, uses default: control_penalty*||u||^2 + alpha_terminal*||x_T - x*||^2
    cost_fn: Optional[Any] = None

    # fn(batch_size) -> z0 tensor (batch_size, state_dim)
    # If None, uses default: initial_state + Gaussian noise
    sample_fn: Optional[Any] = None

    # Double-integrator-specific: velocity penalty term in default cost_fn
    velocity_penalty: bool = True

    # ========================================================================
    # Time Parameters
    # ========================================================================
    T: float = 2.0
    dt: float = 0.05

    # ========================================================================
    # CBF Parameters
    # ========================================================================
    cbf_alpha: float = 10.0   # scalar or tuple (alpha1, alpha2) for HOCBF
    use_slack: bool = True         # soft constraints (always feasible); False = hard QP
    slack_penalty: float = 1e4     # penalty on slack variables (ignored if use_slack=False)

    # ========================================================================
    # Cost Weights
    # ========================================================================
    control_penalty: float = 1.0
    terminal_cost_weight: float = 500.0

    # Alpha-terminal scheduling: anneal terminal_cost_weight upward during training
    alpha_terminal_final: Optional[float] = None  # ceiling value (None = no scheduling)
    alpha_terminal_step: float = 0.0              # increment per event
    alpha_terminal_every: int = 0                 # epochs between increments (0 = off)

    # ========================================================================
    # Network Architecture
    # ========================================================================
    hidden_dim: int = 64
    num_hidden_layers: int = 3
    activation: str = 'relu'
    use_time: bool = False   # if True, append t/T as extra input to policy

    # ========================================================================
    # Training Parameters
    # ========================================================================
    num_epochs: int = 200
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    initial_state_std: float = 0.3
    grad_clip_norm: float = 1.0    # 0 = no clipping
    lr_decay_every: int = 0        # epochs between LR halving (0 = off)
    lr_decay_factor: float = 0.5

    # ========================================================================
    # Logging / Plotting
    # ========================================================================
    log_barrier: bool = False      # compute and log min barrier value each epoch
    log_grad_norm: bool = False    # log gradient norm each epoch
    log_every: int = 1             # print every N epochs

    # fn(trainer, epoch, states) called every plot_every epochs (0 = off)
    # states: (1, num_steps+1, state_dim) from a no-grad single-trajectory rollout
    plot_fn: Optional[Any] = None
    plot_every: int = 0

    # ========================================================================
    # Computation
    # ========================================================================
    device: Optional[str] = None
    use_double_precision: bool = True

    # ========================================================================
    # Model Saving
    # ========================================================================
    save_path: str = './models/cbf_controller.pth'

    # ========================================================================
    # Output Files  (None = auto-derive from save_path directory)
    # ========================================================================
    csv_path: Optional[str] = None   # e.g. './models/training_history.csv'
    log_path: Optional[str] = None   # e.g. './models/training_log.txt'

    def __post_init__(self):
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.dtype = torch.float64 if self.use_double_precision else torch.float32
        self.num_steps = int(self.T / self.dt)

        # Determine state/control dims
        if self.dynamics is not None:
            self.state_dim = self.dynamics.state_dim
            self.control_dim = self.dynamics.control_dim
        elif self.dynamics_type == 'single_integrator':
            self.state_dim = self.position_dim
            self.control_dim = self.position_dim
        elif self.dynamics_type == 'double_integrator':
            self.state_dim = 2 * self.position_dim
            self.control_dim = self.position_dim
        else:
            raise ValueError(
                "Provide either a 'dynamics' object or 'dynamics_type' string "
                "('single_integrator' or 'double_integrator')"
            )

        self.initial_state_tensor = torch.tensor(
            self.initial_state, dtype=self.dtype, device=self.device
        )
        self.target_state_tensor = torch.tensor(
            self.target_state, dtype=self.dtype, device=self.device
        )

        if len(self.initial_state) != self.state_dim:
            raise ValueError(
                f"initial_state length {len(self.initial_state)} != state_dim {self.state_dim}"
            )
        if len(self.target_state) != self.state_dim:
            raise ValueError(
                f"target_state length {len(self.target_state)} != state_dim {self.state_dim}"
            )

    def get_num_obstacles(self):
        if self.obstacles_list is not None:
            return len(self.obstacles_list)
        return len(self.obstacles)

    def __repr__(self):

        lines = ["=" * 70, "CBF Training Configuration", "=" * 70]
        if self.dynamics is not None:
            lines.append(f"Dynamics:      {self.dynamics}")
        else:
            lines.append(f"Dynamics:      {self.dynamics_type}")
        lines.append(f"State/control: {self.state_dim} / {self.control_dim}")
        lines.append(f"Time:          T={self.T}s, dt={self.dt}s, steps={self.num_steps}")
        lines.append(f"Obstacles:     {self.get_num_obstacles()}")
        lines.append(f"CBF alpha:     {self.cbf_alpha}")
        lines.append(f"Cost:          control={self.control_penalty}, terminal={self.terminal_cost_weight}")
        if self.alpha_terminal_final is not None:
            lines.append(
                f"Alpha sched:   +{self.alpha_terminal_step} every {self.alpha_terminal_every} "
                f"epochs → {self.alpha_terminal_final}"
            )
        time_suffix = "+t" if self.use_time else ""
        lines.append(
            f"Network:       {self.state_dim}{time_suffix}→{self.hidden_dim}x{self.num_hidden_layers}→{self.control_dim}"
        )
        lines.append(
            f"Training:      epochs={self.num_epochs}, batch={self.batch_size}, "
            f"lr={self.learning_rate}, wd={self.weight_decay}"
        )
        lines.append(f"Device:        {self.device}, dtype={self.dtype}")
        lines.append("=" * 70)
        return "\n".join(lines)

