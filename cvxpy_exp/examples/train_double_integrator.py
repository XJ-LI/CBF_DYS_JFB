"""
Train CBF controller for multi-agent double integrator with 3 obstacles.

System
------
N agents each running double integrator dynamics: ẍ = a (HOCBF, relative degree 2)

State layout  (DoubleIntegrator(dim=N*2)):
    x = [p1, p2, ..., pN,  v1, v2, ..., vN]   each pi, vi ∈ R²

Control layout:
    u = [a1, a2, ..., aN]                       each ai ∈ R²

Barrier
-------
MultiAgentCircularObstacle2 — one HOCBF constraint per (agent, obstacle) pair.
Block-diagonal A_cbf: agent i's row is non-zero only at its own control columns.
Total constraints = n_agent × n_obstacles.

To switch to single agent, set n_agent = 1.
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

from training   import TrainingConfig, CBFTrainer
from visualization import plot_trajectories
from dynamics   import DoubleIntegrator
from barriers   import CircularObstacle


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def make_plot_fn(obstacles, n_agent, spatial_dim, target_positions, plot_dir):
    """
    Returns a plot_fn(trainer, epoch, states) hook.
    Draws all agents' 2-D trajectories on one axes.
    states: (1, T+1, state_dim)  from a no-grad single rollout.
    """
    colors = plt.cm.tab10.colors

    def plot_fn(_, epoch, states):
        traj = states.squeeze(0).cpu().numpy()   # (T+1, state_dim)

        fig, ax = plt.subplots(figsize=(7, 7))

        # Obstacles
        for k, obs in enumerate(obstacles):
            c = obs.center.cpu().numpy()
            ax.add_patch(Circle(c, obs.radius, color='red', alpha=0.6,
                                label='Obstacle' if k == 0 else None))
            safe_r = math.sqrt(obs.radius ** 2 + obs.epsilon)
            ax.add_patch(Circle(c, safe_r, color='red', alpha=0.2,
                                linestyle='--', fill=False,
                                label='Safety boundary' if k == 0 else None))

        # Per-agent trajectories
        for i in range(n_agent):
            col = colors[i % len(colors)]
            # positions of agent i: columns [i*sdim : (i+1)*sdim]
            pos = traj[:, i * spatial_dim:(i + 1) * spatial_dim]
            ax.plot(pos[:, 0], pos[:, 1], '-', color=col, linewidth=2,
                    label=f'Agent {i}')
            ax.plot(pos[0,  0], pos[0,  1], 'o', color=col, markersize=8)
            ax.plot(pos[-1, 0], pos[-1, 1], 's', color=col, markersize=8)
            ax.plot(target_positions[i][0], target_positions[i][1],
                    '*', color=col, markersize=12)

        # Compute limits from all agent trajectories with padding
        all_x = traj[:, 0::spatial_dim].flatten()
        all_y = traj[:, 1::spatial_dim].flatten()
        pad = 0.5
        ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
        ax.set_ylim(all_y.min() - pad, all_y.max() + pad)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        ax.set_title(
            f'Epoch {epoch} — {n_agent}-agent Double Integrator (HOCBF)',
            fontsize=12
        )

        path = plot_dir / f"traj_epoch_{epoch:04d}.png"
        plt.savefig(path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        tqdm.write(f"  Saved plot: {path}")

    return plot_fn


# ─────────────────────────────────────────────────────────────────────────────
# Cost function
# ─────────────────────────────────────────────────────────────────────────────

def make_cost_fn(target_state_tensor, n_agent, spatial_dim, dt,
                 control_penalty=1.0, velocity_penalty=True):
    """
    Custom cost matching the reference double_integrator_multi.py:

    Running:  dt * 0.5*||u||²  [+ dt * ||v||²  if velocity_penalty=True]
    Terminal: 0.5*||pos - pos_target||²  +  0.5*||vel||²
    """
    pos_dim = n_agent * spatial_dim   # first half of state = positions

    def cost_fn(states, controls, alpha_terminal):
        # Running cost — dt-weighted, matching reference compute_loss
        run = dt * 0.5 * (controls ** 2).sum(dim=-1).sum(dim=-1)   # (batch,)
        if velocity_penalty:
            # velocities live in state[pos_dim : 2*pos_dim]
            vel = states[:, 1:, pos_dim:2*pos_dim]                  # (batch, T, vel_dim)
            run = run + dt * vel.pow(2).sum(dim=-1).sum(dim=-1)     # (batch,)
        running_cost = control_penalty * run.mean()

        # Terminal cost — position error + velocity penalty (matches reference G)
        final = states[:, -1, :]
        pos_err = final[:, :pos_dim] - target_state_tensor[:pos_dim].unsqueeze(0)
        vel_fin = final[:, pos_dim:2*pos_dim]
        terminal_cost = (
            0.5 * pos_err.pow(2).sum(dim=-1) +
            0.5 * vel_fin.pow(2).sum(dim=-1)
        ).mean()

        total = running_cost + alpha_terminal * terminal_cost
        return total, running_cost, terminal_cost

    return cost_fn


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",        type=int,   default=5000)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--lr_decay",      type=int,   default=800)
    parser.add_argument("--hidden_dim",    type=int,   default=128)
    parser.add_argument("--n_blocks",      type=int,   default=3)
    parser.add_argument("--n_agent",       type=int,   default=6)
    parser.add_argument("--seed",          type=int,   default=0)
    parser.add_argument("--use_slack",     action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--slack_penalty", type=float, default=1e4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Problem size ──────────────────────────────────────────────────────────
    n_agent     = args.n_agent
    spatial_dim = 2   # 2-D plane

    # ── Directories ───────────────────────────────────────────────────────────
    script_name = Path(__file__).stem
    save_dir    = Path(__file__).parent.parent / "models" / script_name
    plot_dir    = save_dir / "plots"
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(exist_ok=True)

    # ── Dynamics ──────────────────────────────────────────────────────────────
    # state_dim  = 2 * n_agent * spatial_dim  (positions then velocities)
    # control_dim =     n_agent * spatial_dim  (accelerations)
    dynamics = DoubleIntegrator(dim=n_agent * spatial_dim)

    # ── Obstacles — matching reference double_integrator_multi / train.py ─────
    obstacle_defs = [
        {'center': [0.63, 1.0],  'radius': 0.35, 'epsilon': 0.15},
        {'center': [1.5,  2.5],  'radius': 0.35, 'epsilon': 0.15},
        {'center': [2.37, 1.0],  'radius': 0.35, 'epsilon': 0.15},
    ]
    obstacles = [
        CircularObstacle(
            center=obs['center'], radius=obs['radius'], epsilon=obs['epsilon'],
            dynamics=dynamics, n_agent=n_agent,
        )
        for obs in obstacle_defs
    ]

    # ── Initial & target states — matching reference ───────────────────────────
    # Agents equally spaced clockwise on a circle around [1.5, 1.5].
    # Initial: large circle (radius 2√2 ≈ 2.83), agents spread out
    # Target:  small circle (radius 0.2√2 ≈ 0.28), agents converge to centre
    # Layout: [p1, p2, ..., pN,  v1, v2, ..., vN]  each in R^spatial_dim
    scene_center  = np.array([1.5, 1.5])
    r_initial     = (2 * 2.0 ** 2) ** 0.5          # = 2√2
    r_target      = (2 * 0.2 ** 2) ** 0.5          # = 0.2√2
    angles        = -2 * np.pi * np.arange(n_agent) / n_agent   # clockwise

    initial_positions = [
        (scene_center + r_initial * np.array([np.cos(a), np.sin(a)])).tolist()
        for a in angles
    ]
    target_positions = [
        (scene_center + r_target  * np.array([np.cos(a), np.sin(a)])).tolist()
        for a in angles
    ]

    initial_state = [v for p in initial_positions for v in p] \
                  + [0.0] * (n_agent * spatial_dim)
    target_state  = [v for p in target_positions  for v in p] \
                  + [0.0] * (n_agent * spatial_dim)

    # ── Training config ───────────────────────────────────────────────────────
    T, dt = 10.0, 0.2

    config = TrainingConfig(
        dynamics      = dynamics,
        obstacles_list = obstacles,
        initial_state  = initial_state,
        target_state   = target_state,

        # Time
        T  = T,
        dt = dt,

        # HOCBF parameters (alpha1, alpha2)
        cbf_alpha = (1.0, 1.0),
        use_slack     = args.use_slack,
        slack_penalty = args.slack_penalty,

        # Cost — anneal terminal weight upward during training
        control_penalty      = 1.0,
        terminal_cost_weight = 20.0,
        alpha_terminal_final = 500.0,
        alpha_terminal_step  = 5.0,
        alpha_terminal_every = 10,
        velocity_penalty = True,   # set False to drop the per-step ||v||² term

        # Network — matches reference double_integrator_multi
        hidden_dim        = args.hidden_dim,
        num_hidden_layers = args.n_blocks,
        activation        = 'tanh',
        use_time          = True,   # append t/T to policy input

        # Optimiser — matches reference: epochs=5000, lr_decay=800
        num_epochs     = args.epochs,
        batch_size     = 32,
        learning_rate  = args.lr,
        weight_decay   = 1e-3,
        grad_clip_norm = 0.0,
        lr_decay_every = args.lr_decay,
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

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = CBFTrainer(config)

    # Custom cost matching reference (velocity_penalty flag controls per-step ||v||²)
    target_tensor = trainer.config.target_state_tensor
    config.cost_fn = make_cost_fn(
        target_tensor, n_agent, spatial_dim, dt,
        control_penalty=config.control_penalty,
        velocity_penalty=config.velocity_penalty,
    )

    # Plot hook — drawn every plot_every epochs
    config.plot_fn   = make_plot_fn(trainer.obstacles, n_agent, spatial_dim,
                                    target_positions, plot_dir)
    config.plot_every = 50

    # Custom sampler: perturb all agents' start positions, keep velocities zero
    z0_base  = torch.tensor(initial_state, dtype=torch.float64, device=device)
    pos_std  = 0.05   # matches reference z0_std
    pos_dim  = n_agent * spatial_dim

    def sample_fn(batch_size):
        z0 = z0_base.unsqueeze(0).repeat(batch_size, 1).clone()
        z0[:, :pos_dim] += pos_std * torch.randn(
            batch_size, pos_dim, dtype=torch.float64, device=device
        )
        return z0

    config.sample_fn = sample_fn

    # ── Run ───────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"TRAINING: {n_agent}-agent Double Integrator + {len(obstacles)} Obstacles (HOCBF)")
    print(f"  state_dim={dynamics.state_dim}  control_dim={dynamics.control_dim}"
          f"  constraints={n_agent * len(obstacles)}")
    print(f"  save_dir : {save_dir}")
    print(f"  plot_dir : {plot_dir}")
    print("=" * 70)

    trainer.train(verbose=True)


    # ── Final plot ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL TRAJECTORY PLOT")
    print("=" * 70)
    plot_trajectories(
        trainer,
        num_trajectories = 10,
        show_velocity    = False,
        save_path        = str(save_dir / "final_trajectories.png"),
    )

    print(f"\nDone!  Best loss: {trainer.best_loss:.4e}")
    print(f"Model: {config.save_path}")


if __name__ == "__main__":
    main()
