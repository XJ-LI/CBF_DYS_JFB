"""
Example: Train CBF controller for multi-quadrotor system using CVXPyLayers.

This demonstrates:
    - Multi-agent quadrotor dynamics (12-state per agent)
    - HOCBF (Higher-Order CBF, relative degree 2)
    - 3D spherical obstacles
    - CVXPyLayers-based differentiable QP solver
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import argparse
import json
from pathlib import Path

from dynamics import Quadrotor
from barriers import SphericalObstacle
from training import TrainingConfig, CBFTrainer


# ============================================================================
# Quadrotor-specific hooks
# ============================================================================

def make_output_transform(n_agent, T_hover, T_dev_scale, tau_scale, device, dtype):
    """Returns fn(u_raw) -> u_desired with tanh scaling and hover bias."""
    hover_bias = torch.zeros(n_agent * 4, device=device, dtype=dtype)
    hover_bias[::4] = T_hover

    def output_transform(u_raw):
        batch = u_raw.shape[0]
        u = u_raw.reshape(batch, n_agent, 4)
        u_scaled = torch.cat([
            T_dev_scale * torch.tanh(u[:, :, 0:1]),   # thrust deviation
            tau_scale   * torch.tanh(u[:, :, 1:4]),   # torques
        ], dim=-1).reshape(batch, n_agent * 4)
        return u_scaled + hover_bias

    return output_transform


def make_cost_fn(p_target, alpha_running, n_agent, dt=0.2):
    """
    Returns fn(states, controls, alpha_terminal) -> (total, running, terminal).

    Running cost: control effort + velocity penalty (dt-weighted, matching reference).
    Terminal cost: position + velocity + angles + angular velocity.
    """
    def cost_fn(states, controls, alpha_terminal):
        # states: (batch, T+1, state_dim), controls: (batch, T, control_dim)
        batch = states.shape[0]

        # Running: dt-weighted sum over time steps (matches reference compute_loss)
        vel_penalty = (
            states[:, 1:, :]
            .reshape(batch, -1, n_agent, 12)[:, :, :, 6:9]
            .pow(2).sum(dim=-1).sum(dim=-1)  # (batch, T)
        )
        run_control = dt * 0.5 * (controls ** 2).sum(dim=-1).sum(dim=-1)  # (batch,)
        run_vel = dt * vel_penalty.sum(dim=-1)                              # (batch,)
        running_cost = (run_control + run_vel).mean()

        # Terminal: per-component breakdown
        x_final = states[:, -1, :].reshape(batch, n_agent, 12)
        pos     = x_final[:, :, 0:3]
        angles  = x_final[:, :, 3:6]
        vel     = x_final[:, :, 6:9]
        ang_vel = x_final[:, :, 9:12]

        terminal_cost = (
            0.5 * ((pos - p_target.unsqueeze(0)) ** 2).sum(dim=-1).sum(dim=-1) +
            0.5 * vel.pow(2).sum(dim=-1).sum(dim=-1) +
            0.5 * angles.pow(2).sum(dim=-1).sum(dim=-1) +
            0.5 * ang_vel.pow(2).sum(dim=-1).sum(dim=-1)
        ).mean()

        total = alpha_running * running_cost + alpha_terminal * terminal_cost
        return total, running_cost, terminal_cost

    return cost_fn


def make_sample_fn(initial_state, n_agent, z0_std, device, dtype):
    """Returns fn(batch_size) -> z0 with per-agent xy noise."""
    def sample_fn(batch_size):
        z0 = initial_state.unsqueeze(0).repeat(batch_size, 1)
        for i in range(n_agent):
            z0[:, 12*i:12*i+2] += z0_std * torch.randn(batch_size, 2, device=device, dtype=dtype)
        return z0
    return sample_fn


def make_plot_fn(dynamics, obstacles, p_target, plot_dir):
    """Returns fn(trainer, epoch, states) that saves a trajectory plot."""
    def plot_fn(_trainer, epoch, states):
        # states: (1, T+1, state_dim) — transpose to (1, state_dim, T+1) for dynamics.plot
        traj = states.permute(0, 2, 1)
        path = plot_dir / f"traj_epoch_{epoch:04d}.png"
        dynamics.plot_trajectory(
            traj=traj,
            obstacles=obstacles,
            p_target=p_target,
            save_path=str(path),
            title=f"Epoch {epoch} - Quadrotor Trajectories"
        )
        tqdm_write = __import__('tqdm').tqdm.write
        tqdm_write(f"  Saved trajectory plot: {path}")
    return plot_fn


# ============================================================================
# Main
# ============================================================================

def train_quadrotor(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float64
    torch.manual_seed(args.seed)

    # --- System parameters ---
    n_agent = 5
    mass    = 0.5
    gravity = 1.0
    T_hover = mass * gravity
    T_dev_scale = 0.5
    tau_scale   = 0.1
    alpha_running = 1.0

    # --- Dynamics ---
    dynamics = Quadrotor(n_agent=n_agent, mass=mass, gravity=gravity)

    # --- Obstacles ---
    obstacle_configs = [
        {'center': [0.63, 1.0, 1.0], 'radius': 0.35, 'epsilon': 0.15},
        {'center': [1.5,  2.5, 0.8], 'radius': 0.35, 'epsilon': 0.15},
        {'center': [2.37, 1.0, 1.0], 'radius': 0.35, 'epsilon': 0.15},
    ]
    obstacles = [
        SphericalObstacle(
            center=obs['center'], radius=obs['radius'], epsilon=obs['epsilon'],
            dynamics=dynamics, n_agent=n_agent
        ).to(device)
        for obs in obstacle_configs
    ]

    # --- Target positions ---
    x_positions = torch.linspace(1.1, 1.9, n_agent, device=device, dtype=dtype)
    p_target = torch.zeros(n_agent, 3, device=device, dtype=dtype)
    p_target[:, 0] = x_positions
    p_target[:, 1] = 3.5
    p_target[:, 2] = 1.0

    # --- Initial state ---
    initial_state = torch.zeros(dynamics.state_dim, device=device, dtype=dtype)
    for i in range(n_agent):
        initial_state[12*i]   = x_positions[i]
        initial_state[12*i+1] = -0.5
        initial_state[12*i+2] = 1.0

    # --- Output directory ---
    script_name = Path(__file__).stem
    save_dir = Path(__file__).parent.parent / "models" / script_name
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = save_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    # --- Config ---
    lr = args.lr  # use as-is (default 0.001 = 1e-3, matching reference)
    T, dt = 10.0, 0.2

    config = TrainingConfig(
        # Problem
        dynamics=dynamics,
        obstacles_list=obstacles,
        initial_state=initial_state.tolist(),
        target_state=[0.0] * dynamics.state_dim,  # unused (custom cost_fn)

        # Time
        T=T,
        dt=dt,

        # CBF — (1,1) matches reference gamma(x)=x convention exactly
        cbf_alpha=(1.0, 1.0),

        # Cost
        control_penalty=1.0,
        terminal_cost_weight=20.0,        # match reference (starts at 20, not 100)
        alpha_terminal_final=200.0,       # match reference cap
        alpha_terminal_step=5.0,
        alpha_terminal_every=20,          # match reference (every 20 epochs)

        # Network — 'resnet' matches reference ControlNet (ResBlock + SiLU)
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.n_blocks,
        activation='resnet',

        # Training
        num_epochs=args.epochs,
        batch_size=32,            # match reference
        learning_rate=lr,
        weight_decay=1e-3,
        initial_state_std=0.0,        # handled by sample_fn
        grad_clip_norm=100.0 if args.grad_clip else 0.0,
        lr_decay_every=args.lr_decay,
        lr_decay_factor=0.5,
        use_time=True,

        # Logging
        log_barrier=True,
        log_grad_norm=True,
        log_every=1,

        # Hooks
        output_transform=make_output_transform(
            n_agent, T_hover, T_dev_scale, tau_scale, device, dtype
        ),
        cost_fn=make_cost_fn(p_target, alpha_running, n_agent, dt=dt),
        sample_fn=make_sample_fn(initial_state, n_agent, z0_std=4e-2, device=device, dtype=dtype),
        plot_fn=make_plot_fn(dynamics, obstacles, p_target, plot_dir),
        plot_every=50,

        # Saving
        device=device,
        use_double_precision=True,
        save_path=str(save_dir / "best_model_cvxpy.pth"),
        csv_path=str(save_dir / "training_history_cvxpy.csv"),
        log_path=str(save_dir / "training_log_cvxpy.txt"),
    )

    print("\n" + "=" * 70)
    print("TRAINING: Multi-Quadrotor + 3D Obstacles (CVXPyLayers + HOCBF)")
    print("=" * 70)
    print(f"Agents: {n_agent}  |  mass={mass}  |  gravity={gravity}  |  T_hover={T_hover}")
    print(f"Obstacles: {len(obstacles)}")
    for i, obs in enumerate(obstacles):
        print(f"  [{i+1}] {obs}")
    print(f"Target: x∈[1.1,1.9], y=3.5, z=1.0")
    print(f"Save dir: {save_dir}")

    # Save configuration
    config_dict = {
        'n_agent': n_agent, 'mass': mass, 'gravity': gravity,
        'state_dim': dynamics.state_dim, 'control_dim': dynamics.control_dim,
        'T': config.T, 'dt': config.dt, 'num_steps': config.num_steps,
        'obstacles': obstacle_configs,
        'p_target': p_target.tolist(),
        'cbf_alpha': config.cbf_alpha,
        'alpha_running': alpha_running,
        'alpha_terminal_initial': config.terminal_cost_weight,
        'alpha_terminal_final': config.alpha_terminal_final,
        'weight_decay': config.weight_decay,
        'num_epochs': config.num_epochs,
        'learning_rate': config.learning_rate,
        'lr_decay_every': config.lr_decay_every,
        'batch_size': config.batch_size,
        'z0_std': 4e-2,
        'grad_clip_norm': config.grad_clip_norm,
        'hidden_dim': config.hidden_dim,
        'num_hidden_layers': config.num_hidden_layers,
        'seed': args.seed,
        'framework': 'CVXPyLayers',
    }
    with open(save_dir / "config_cvxpy.json", 'w') as f:
        json.dump(config_dict, f, indent=2)

    # --- Train ---
    trainer = CBFTrainer(config)
    trainer.train(verbose=True)

    # --- Save final model ---
    import torch as _torch
    _torch.save(trainer.policy.state_dict(), save_dir / "final_model_cvxpy.pth")
    print(f"Final model: {save_dir / 'final_model_cvxpy.pth'}")
    print(f"Best model:  {save_dir / 'best_model_cvxpy.pth'}  (loss={trainer.best_loss:.4e})")


    # --- Final trajectory plot ---
    with torch.no_grad():
        z0 = initial_state.unsqueeze(0)
        final_states, _ = trainer.rollout(z0)
    traj_final = final_states.permute(0, 2, 1)  # (1, state_dim, T+1)
    dynamics.plot_trajectory(
        traj=traj_final,
        obstacles=obstacles,
        p_target=p_target,
        save_path=str(save_dir / "final_trajectory.png"),
        title="Final Trajectory - Best Model"
    )
    print(f"Final plot:  {save_dir / 'final_trajectory.png'}")
    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=2800)
    parser.add_argument("--lr",         type=float, default=0.001)
    parser.add_argument("--lr_decay",   type=int,   default=600)
    parser.add_argument("--hidden_dim", type=int,   default=128)
    parser.add_argument("--n_blocks",   type=int,   default=3)
    parser.add_argument("--seed",       type=int,   default=0)
    parser.add_argument("--grad_clip",  action='store_true', default=False)
    args = parser.parse_args()

    train_quadrotor(args)
