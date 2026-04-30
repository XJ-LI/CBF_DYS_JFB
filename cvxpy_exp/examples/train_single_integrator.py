"""
Train CBF controller for 50-agent 3D single integrator swarm.

Matches reference train.py --problem single_integrator_swarm exactly:
  - 50 agents, 3D single integrator (ẋ = u, state ∈ R^150)
  - 2 cylindrical obstacles (infinite in Z)
  - Initial: two rows at z=0 and z=3, x∈[0,5], y=0
  - Target: all agents → [2.5, 5.0, 1.5]
  - Running cost: dt * 0.5 * mean(||u||²)  (dt-weighted lagrangian)
  - Terminal cost: 0.5 * mean(||pos - target||²)
  - Alpha annealing: starts 20, +5 every 20 epochs, capped at 500
  - T=10.0, dt=0.2, alpha_cbf=1.0, lr=1e-3, weight_decay=1e-3
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import random
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from training import TrainingConfig, CBFTrainer
from dynamics import SingleIntegrator
from barriers import MultiAgentCylindricalObstacle1


# ─────────────────────────────────────────────────────────────────────────────
# Hooks
# ─────────────────────────────────────────────────────────────────────────────

def make_cost_fn(p_target, n_agent, dt):
    """
    Matches reference compute_loss (single_integrator_swarm.py):
      running  = dt * 0.5 * mean_batch(sum_t ||u_t||²)   [lagrangian, dt-weighted]
      terminal = 0.5 * mean_batch(sum_agents ||pos_final - target||²)
      total    = running + alpha_terminal * terminal
    """
    def cost_fn(states, controls, alpha_terminal):
        batch = states.shape[0]

        # Running: dt * 0.5 * ||u||² per step, summed over T, mean over batch
        running = dt * 0.5 * (controls.pow(2).sum(dim=-1).sum(dim=-1)).mean()

        # Terminal: 0.5 * mean(sum_agents ||pos - target||²)
        pos_final = states[:, -1, :].reshape(batch, n_agent, 3)
        terminal  = 0.5 * ((pos_final - p_target.unsqueeze(0)) ** 2).sum(dim=-1).sum(dim=-1).mean()

        total = running + alpha_terminal * terminal
        return total, running, terminal

    return cost_fn


def make_sample_fn(n_agent, z0_std, device, dtype):
    """
    Initial condition matching reference single_integrator_swarm.sample_initial_condition:
      - 25 agents: x∈[0,5], y=0, z=0
      - 25 agents: x∈[0,5], y=0, z=3
      - Plus Gaussian noise std=z0_std
    """
    n_per_row  = n_agent // 2
    x_row      = torch.linspace(0.0, 5.0, n_per_row, device=device, dtype=dtype)
    x_positions = torch.cat([x_row, x_row], dim=0)               # (n_agent,)
    z_centers   = torch.cat([
        torch.zeros(n_per_row, device=device, dtype=dtype),
        3.0 * torch.ones(n_per_row, device=device, dtype=dtype),
    ], dim=0)                                                     # (n_agent,)

    def sample_fn(batch_size):
        z0      = torch.zeros(batch_size, 3 * n_agent, device=device, dtype=dtype)
        z0_view = z0.view(batch_size, n_agent, 3)
        z0_view[:, :, 0] = x_positions   # x
        z0_view[:, :, 1] = 0.0           # y
        z0_view[:, :, 2] = z_centers     # z (0 or 3)
        z0_view += z0_std * torch.randn(batch_size, n_agent, 3, device=device, dtype=dtype)
        return z0

    return sample_fn


def make_plot_fn(obstacles, n_agent, p_target, plot_dir):
    """3D + bird's-eye trajectory plot, matching reference plot_trajectory style."""
    colors = plt.cm.tab20(np.linspace(0, 1, n_agent))

    def plot_fn(_, epoch, states):
        traj = states.squeeze(0).cpu().numpy()   # (T+1, n_agent*3)

        fig = plt.figure(figsize=(14, 6))
        ax  = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122)

        for a in range(n_agent):
            x = traj[:, 3*a];  y = traj[:, 3*a+1];  z = traj[:, 3*a+2]
            ax.plot(x, y, z, color=colors[a], linewidth=1, alpha=0.6)
            ax.scatter(x[0], y[0], z[0], color=colors[a], s=10, marker='o')
            ax2.plot(x, y, color=colors[a], linewidth=1, alpha=0.6)
            ax2.scatter(x[0], y[0], color=colors[a], s=10, marker='o')

        # Obstacles
        theta = np.linspace(0, 2 * np.pi, 40)
        for obs in obstacles:
            c = (obs.center_xy if hasattr(obs, "center_xy") else obs.center).cpu().numpy()
            r = obs.radius
            ax2.add_patch(plt.Circle(c, r, color='blue', alpha=0.25))
            ax2.add_patch(plt.Circle(c, r + obs.epsilon, color='k',
                                     fill=False, linestyle='--', linewidth=1))
            # 3D cylinder sketch
            z_cyl = np.linspace(-0.5, 4.0, 10)
            T_cyl, Z_cyl = np.meshgrid(theta, z_cyl)
            ax.plot_surface(c[0] + r*np.cos(T_cyl), c[1] + r*np.sin(T_cyl), Z_cyl,
                            alpha=0.15, color='blue')

        # Targets
        tgt = p_target.cpu().numpy()
        ax.scatter(tgt[:, 0], tgt[:, 1], tgt[:, 2], c='green', s=40, marker='X')
        ax2.scatter(tgt[:, 0], tgt[:, 1], c='green', s=40, marker='X', label='Target')

        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(f'Epoch {epoch} — 3D Swarm')
        ax2.set_xlabel('X'); ax2.set_ylabel('Y')
        ax2.set_aspect('equal'); ax2.grid(True, alpha=0.3); ax2.legend(fontsize=8)
        ax2.set_title("Bird's-eye view")

        plt.tight_layout()
        path = plot_dir / f"traj_epoch_{epoch:04d}.png"
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        tqdm.write(f"  Saved plot: {path}")

    return plot_fn


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",        type=int,   default=3500)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--lr_decay",      type=int,   default=800)
    parser.add_argument("--hidden_dim",    type=int,   default=192)
    parser.add_argument("--n_blocks",      type=int,   default=4)
    parser.add_argument("--n_agent",       type=int,   default=50)
    parser.add_argument("--seed",          type=int,   default=0)
    parser.add_argument("--use_slack",     action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--slack_penalty", type=float, default=1e4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device      = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype       = torch.float64
    n_agent     = args.n_agent
    spatial_dim = 3   # 3D single integrator

    # ── Directories ───────────────────────────────────────────────────────────
    script_name = Path(__file__).stem
    save_dir    = Path(__file__).parent.parent / "models" / script_name
    plot_dir    = save_dir / "plots"
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(exist_ok=True)

    # ── Dynamics — 3D single integrator, 50 agents → 150-D state ─────────────
    dynamics = SingleIntegrator(dim=n_agent * spatial_dim)

    # ── Obstacles — 2 cylinders, matching reference single_integrator_swarm ───
    cyl1_center_xy = [1.5, 3.5];  cyl1_radius = 0.5
    cyl2_center_xy = [4.0, 2.0];  cyl2_radius = 0.7
    eps_safe       = 0.1

    obstacles = [
        MultiAgentCylindricalObstacle1(
            center_xy=cyl1_center_xy, radius=cyl1_radius, epsilon=eps_safe,
            n_agent=n_agent, spatial_dim=spatial_dim,
        ),
        MultiAgentCylindricalObstacle1(
            center_xy=cyl2_center_xy, radius=cyl2_radius, epsilon=eps_safe,
            n_agent=n_agent, spatial_dim=spatial_dim,
        ),
    ]

    # ── Target — all agents converge to [2.5, 5.0, 1.5] ─────────────────────
    target_center = torch.tensor([2.5, 5.0, 1.5], device=device, dtype=dtype)
    p_target      = target_center.unsqueeze(0).repeat(n_agent, 1)   # (n_agent, 3)
    target_state  = p_target.reshape(-1).tolist()                    # 150D

    # Nominal initial state (noise added by sample_fn)
    initial_state = [0.0] * (n_agent * spatial_dim)

    # ── Training config ───────────────────────────────────────────────────────
    T, dt = 10.0, 0.2   # 50 steps — matching reference

    config = TrainingConfig(
        dynamics       = dynamics,
        obstacles_list = obstacles,
        initial_state  = initial_state,
        target_state   = target_state,

        # Time — matching reference
        T  = T,
        dt = dt,

        # CBF alpha — matching reference gamma(x) = 1.0 * x
        cbf_alpha     = 1.0,
        use_slack     = args.use_slack,
        slack_penalty = args.slack_penalty,
        velocity_penalty=False,

        # Cost annealing — matching reference: start 20, +5 every 20 epochs, cap 500
        control_penalty      = 1.0,
        terminal_cost_weight = 20.0,
        alpha_terminal_final = 500.0,
        alpha_terminal_step  = 5.0,
        alpha_terminal_every = 20,

        # Network — matches reference ControlNet (ResBlock + SiLU)
        hidden_dim        = args.hidden_dim,
        num_hidden_layers = args.n_blocks,
        activation        = 'resnet',
        use_time          = True,

        # Optimiser — matching reference: lr=1e-3, wd=1e-3, lr_decay
        num_epochs      = args.epochs,
        batch_size      = 32,
        learning_rate   = args.lr,
        weight_decay    = 1e-3,
        grad_clip_norm  = 0.0,
        lr_decay_every  = args.lr_decay,
        lr_decay_factor = 0.5,

        # Logging
        log_barrier   = True,
        log_grad_norm = True,
        log_every     = 1,

        # Saving
        device               = device,
        use_double_precision = True,
        save_path            = str(save_dir / "best_model.pth"),
        csv_path             = str(save_dir / "training_history.csv"),
        log_path             = str(save_dir / "training_log.txt"),
    )

    trainer = CBFTrainer(config)

    # ── Hooks ─────────────────────────────────────────────────────────────────
    config.cost_fn   = make_cost_fn(p_target.to(device), n_agent, dt)
    config.sample_fn = make_sample_fn(n_agent, z0_std=5e-2, device=device, dtype=dtype)
    config.plot_fn   = make_plot_fn(obstacles, n_agent, p_target.to(device), plot_dir)
    config.plot_every = 50

    # ── Run ───────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"TRAINING: {n_agent}-agent 3D Single Integrator Swarm (CBF)")
    print(f"  state_dim={dynamics.state_dim}  control_dim={dynamics.control_dim}")
    print(f"  constraints={n_agent * len(obstacles)}  (2 cylinders × {n_agent} agents)")
    print(f"  T={T}, dt={dt}, steps={config.num_steps}")
    print(f"  alpha_cbf=1.0  alpha_terminal: 20 → 500 (+5/20 epochs)")
    print(f"  Obstacles: cyl1={cyl1_center_xy}  r={cyl1_radius}")
    print(f"             cyl2={cyl2_center_xy}  r={cyl2_radius}")
    print(f"  Target: [2.5, 5.0, 1.5] for all agents")
    print(f"  save_dir: {save_dir}")
    print("=" * 70)

    trainer.train(verbose=True)

    print(f"\nDone!  Best loss: {trainer.best_loss:.4e}")
    print(f"Model: {config.save_path}")


if __name__ == "__main__":
    main()
