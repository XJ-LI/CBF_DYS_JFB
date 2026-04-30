"""
Train CBF controller for single-agent 2D double integrator.

Matches reference train.py --problem double_integrator_single exactly:
  - 1 agent, 2D double integrator (ẍ = a, state ∈ R^4: [px, py, vx, vy])
  - 3 circular obstacles at [0.4,1.0], [2.2,2.2], [2.4,0.6], r=0.3, eps=0.1
  - Initial: [0, 0, 0, 0] (origin, zero velocity)
  - Target: position [3, 3]
  - Running cost: dt * 0.5 * ||u||²  (no velocity term)
  - Terminal cost: 0.5 * ||pos_final - target||²  (position only)
  - Alpha annealing: starts 20, +5 every 10 epochs, NO cap
  - T=10.0, dt=0.2, cbf_alpha=(1,1), lr=1e-3, weight_decay=1e-4
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import math
import random
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path
from tqdm import tqdm

from training import TrainingConfig, CBFTrainer
from dynamics import DoubleIntegrator
from barriers import CircularObstacle


# ─────────────────────────────────────────────────────────────────────────────
# Hooks
# ─────────────────────────────────────────────────────────────────────────────

def make_cost_fn(p_target, dt):
    """
    Matches reference double_integrator_single.py compute_loss:
      running  = dt * 0.5 * mean(sum_t ||u_t||²)   (no velocity term)
      terminal = 0.5 * mean(||pos_final - target||²) (position only)
      total    = running + alpha_terminal * terminal
    """
    def cost_fn(states, controls, alpha_terminal):
        # Running: dt * 0.5 * ||u||² per step, summed over T, mean over batch
        running = dt * 0.5 * (controls.pow(2).sum(dim=-1).sum(dim=-1)).mean()

        # Terminal: 0.5 * ||pos_final - target||²  (position = first 2 dims)
        pos_final = states[:, -1, :2]                              # (batch, 2)
        terminal  = 0.5 * ((pos_final - p_target) ** 2).sum(dim=-1).mean()

        total = running + alpha_terminal * terminal
        return total, running, terminal

    return cost_fn


def make_sample_fn(z0_base, z0_std, device, dtype):
    """
    Initial condition matching reference double_integrator_single.sample_initial_condition:
      - z0 = [0, 0, 0, 0]  (origin, zero velocity)
      - Gaussian noise std=z0_std on positions only (not velocity)
    """
    def sample_fn(batch_size):
        z0 = z0_base.unsqueeze(0).repeat(batch_size, 1).clone()
        z0[:, :2] += z0_std * torch.randn(batch_size, 2, device=device, dtype=dtype)
        return z0
    return sample_fn


def make_plot_fn(obstacles, p_target, plot_dir):
    """2D trajectory plot with obstacles and target."""
    def plot_fn(_, epoch, states):
        traj = states.squeeze(0).cpu().numpy()   # (T+1, 4)

        fig, ax = plt.subplots(figsize=(7, 7))

        for k, obs in enumerate(obstacles):
            c = obs.center.cpu().numpy()
            safe_r = math.sqrt(obs.radius ** 2 + obs.epsilon)
            ax.add_patch(Circle(c, obs.radius,
                                color='red', alpha=0.5,
                                label='Obstacle' if k == 0 else None))
            ax.add_patch(Circle(c, safe_r,
                                color='red', alpha=0.2, fill=False,
                                linestyle='--',
                                label='Safety boundary' if k == 0 else None))

        ax.plot(traj[:, 0], traj[:, 1], '-b', linewidth=2, label='Trajectory')
        ax.plot(traj[0, 0],  traj[0, 1],  'go', markersize=8, label='Start')
        ax.plot(traj[-1, 0], traj[-1, 1], 'bs', markersize=8, label='End')

        tgt = p_target.cpu().numpy()
        ax.plot(tgt[0], tgt[1], 'g*', markersize=14, label='Target')

        pad = 0.5
        all_x = traj[:, 0];  all_y = traj[:, 1]
        ax.set_xlim(min(all_x.min(), tgt[0]) - pad, max(all_x.max(), tgt[0]) + pad)
        ax.set_ylim(min(all_y.min(), tgt[1]) - pad, max(all_y.max(), tgt[1]) + pad)
        ax.set_aspect('equal');  ax.grid(True, alpha=0.3);  ax.legend(fontsize=9)
        ax.set_title(f'Epoch {epoch} — Single-Agent Double Integrator (HOCBF)')
        ax.set_xlabel('X');  ax.set_ylabel('Y')

        path = plot_dir / f"traj_epoch_{epoch:04d}.png"
        plt.savefig(path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        tqdm.write(f"  Saved plot: {path}")

    return plot_fn


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",        type=int,   default=1000)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--lr_decay",      type=int,   default=400)
    parser.add_argument("--hidden_dim",    type=int,   default=64)
    parser.add_argument("--n_blocks",      type=int,   default=3)
    parser.add_argument("--seed",          type=int,   default=0)
    parser.add_argument("--use_slack",     action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--slack_penalty", type=float, default=1e4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype  = torch.float64

    # ── Directories ───────────────────────────────────────────────────────────
    script_name = Path(__file__).stem
    save_dir    = Path(__file__).parent.parent / "models" / script_name
    plot_dir    = save_dir / "plots"
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(exist_ok=True)

    # ── Dynamics — 2D double integrator, 1 agent → 4-D state ─────────────────
    dynamics = DoubleIntegrator(dim=2)   # state=[px,py,vx,vy], control=[ax,ay]

    # ── Obstacles — 3 circles, matching reference double_integrator_single ────
    obstacle_defs = [
        {'center': [0.4, 1.0],  'radius': 0.3, 'epsilon': 0.1},
        {'center': [2.2, 2.2],  'radius': 0.3, 'epsilon': 0.1},
        {'center': [2.4, 0.6],  'radius': 0.3, 'epsilon': 0.1},
    ]
    obstacles = [
        CircularObstacle(
            center=obs['center'], radius=obs['radius'], epsilon=obs['epsilon'],
            dynamics=dynamics, n_agent=1,
        )
        for obs in obstacle_defs
    ]

    # ── Target — position [3, 3], matching reference ──────────────────────────
    p_target = torch.tensor([3.0, 3.0], device=device, dtype=dtype)   # (2,)
    # Full state target (velocity = 0)
    target_state = [3.0, 3.0, 0.0, 0.0]

    # ── Initial state — origin, zero velocity ─────────────────────────────────
    initial_state = [0.0, 0.0, 0.0, 0.0]

    # ── Training config ───────────────────────────────────────────────────────
    T, dt = 10.0, 0.2

    config = TrainingConfig(
        dynamics       = dynamics,
        obstacles_list = obstacles,
        initial_state  = initial_state,
        target_state   = target_state,

        # Time
        T  = T,
        dt = dt,

        # HOCBF — (1,1) matches reference gamma(x)=x
        cbf_alpha     = (1.0, 1.0),
        use_slack     = args.use_slack,
        slack_penalty = args.slack_penalty,

        # Cost — no cap on alpha_terminal (reference single has no if-guard)
        control_penalty      = 1.0,
        terminal_cost_weight = 20.0,
        alpha_terminal_final = float('inf'),   # no cap — matches reference train.py:140-142
        alpha_terminal_step  = 5.0,
        alpha_terminal_every = 10,

        # Network — 'resnet' matches reference ControlNet (ResBlock + SiLU)
        hidden_dim        = args.hidden_dim,
        num_hidden_layers = args.n_blocks,
        activation        = 'resnet',
        use_time          = True,

        # Optimiser — matching reference: lr=1e-3, wd=1e-4
        num_epochs      = args.epochs,
        batch_size      = 32,
        learning_rate   = args.lr,
        weight_decay    = 1e-4,
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
    z0_base = torch.tensor(initial_state, device=device, dtype=dtype)
    config.cost_fn   = make_cost_fn(p_target, dt)
    config.sample_fn = make_sample_fn(z0_base, z0_std=1e-1, device=device, dtype=dtype)
    config.plot_fn   = make_plot_fn(obstacles, p_target, plot_dir)
    config.plot_every = 50

    # ── Run ───────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TRAINING: Single-Agent 2D Double Integrator (HOCBF)")
    print(f"  state_dim={dynamics.state_dim}  control_dim={dynamics.control_dim}")
    print(f"  constraints={len(obstacles)}  (3 circular obstacles × 1 agent)")
    print(f"  T={T}, dt={dt}, steps={config.num_steps}")
    print(f"  cbf_alpha=(1,1)  alpha_terminal: 20 → ∞ (+5/10 epochs, no cap)")
    print(f"  Obstacles: {[(o['center'], o['radius']) for o in obstacle_defs]}")
    print(f"  Initial: {initial_state}  Target pos: {p_target.tolist()}")
    print(f"  save_dir: {save_dir}")
    print("=" * 70)

    trainer.train(verbose=True)

    print(f"\nDone!  Best loss: {trainer.best_loss:.4e}")
    print(f"Model: {config.save_path}")


if __name__ == "__main__":
    main()
