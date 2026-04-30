"""
Visualization utilities for CBF trajectories.

Works for both single and double integrator systems.
"""

import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection


def plot_trajectories(
    trainer,
    num_trajectories=10,
    num_steps=None,
    figsize=(10, 10),
    show_velocity=False,
    save_path=None
):
    """
    Visualize safe trajectories.

    Args:
        trainer: CBFTrainer object (contains dynamics, obstacles, policy)
        num_trajectories: Number of trajectories to plot
        num_steps: Number of steps (uses config if None)
        figsize: Figure size
        show_velocity: If True, show velocity field for double integrator
        save_path: Path to save figure (shows if None)
    """
    if num_steps is None:
        num_steps = trainer.config.num_steps

    config = trainer.config
    dynamics = trainer.dynamics
    obstacles = trainer.obstacles
    policy = trainer.policy
    cbf_controller = trainer.cbf_controller

    # Set policy to eval mode
    policy.eval()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot obstacles
    for i, obstacle in enumerate(obstacles):
        # Inner circle (obstacle)
        circle_inner = Circle(
            obstacle.center.cpu().numpy(),
            obstacle.radius,
            color='red',
            alpha=0.6,
            label='Obstacle' if i == 0 else None
        )
        ax.add_patch(circle_inner)

        # Outer circle (safety boundary)
        # h = ||p-c||² - r² - ε ≥ 0  →  safe radius = sqrt(r² + ε)
        safe_radius = math.sqrt(obstacle.radius ** 2 + obstacle.epsilon)
        circle_outer = Circle(
            obstacle.center.cpu().numpy(),
            safe_radius,
            color='red',
            alpha=0.2,
            linestyle='--',
            fill=False,
            label='Safety Boundary' if i == 0 else None
        )
        ax.add_patch(circle_outer)

    # Plot target
    target_pos = config.target_state_tensor[:config.position_dim].cpu().numpy()
    ax.plot(
        target_pos[0], target_pos[1],
        'g*', markersize=20, label='Target'
    )

    # Generate and plot trajectories
    colors = plt.cm.viridis(np.linspace(0, 1, num_trajectories))

    with torch.no_grad():
        for i in range(num_trajectories):
            # Sample initial state
            z0 = trainer.sample_initial_states(1)

            # Simulate trajectory
            states, controls = simulate_trajectory(
                z0, policy, cbf_controller, dynamics, num_steps, config.dt
            )

            # Extract position
            if dynamics.relative_degree == 1:
                # Single integrator: state = position
                trajectory = states.squeeze(0).cpu().numpy()  # (num_steps+1, 2)
            else:
                # Double integrator: extract position from [p, v]
                trajectory = states.squeeze(0)[:, :config.position_dim].cpu().numpy()

            # Plot trajectory
            ax.plot(
                trajectory[:, 0], trajectory[:, 1],
                color=colors[i], alpha=0.7, linewidth=2
            )

            # Plot start and end points
            ax.plot(
                trajectory[0, 0], trajectory[0, 1],
                'o', color=colors[i], markersize=8,
                label='Start' if i == 0 else None
            )
            ax.plot(
                trajectory[-1, 0], trajectory[-1, 1],
                's', color=colors[i], markersize=8,
                label='End' if i == 0 else None
            )

            # Optionally show velocity arrows for double integrator
            if show_velocity and dynamics.relative_degree == 2:
                plot_velocity_arrows(ax, states.squeeze(0), config, colors[i])

    # Formatting
    ax.set_xlabel('X Position', fontsize=14)
    ax.set_ylabel('Y Position', fontsize=14)

    dynamics_label = config.dynamics_type.replace("_", " ").title() if config.dynamics_type else type(dynamics).__name__
    title = f'Safe Trajectories - {dynamics_label}'
    if dynamics.relative_degree == 2:
        title += ' (HOCBF)'
    ax.set_title(title, fontsize=16, fontweight='bold')

    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    else:
        plt.show()

    return fig, ax


def simulate_trajectory(z0, policy, cbf_controller, dynamics, num_steps, dt, T=None):
    """
    Simulate a single trajectory.

    Args:
        z0: Initial state (1, state_dim)
        policy: Policy network
        cbf_controller: CBF-QP controller
        dynamics: Dynamics object
        num_steps: Number of steps
        dt: Time step

    Returns:
        states: State trajectory (1, num_steps+1, state_dim)
        controls: Control trajectory (1, num_steps, control_dim)
    """
    states = [z0]
    controls = []

    z = z0
    t = 0.0
    t_horizon = T if T is not None else num_steps * dt
    for _ in range(num_steps):
        # Get desired control
        u_desired = policy(z, t / t_horizon)

        # Apply CBF filter
        u_safe = cbf_controller.filter_control(z, u_desired)

        # Step dynamics
        z = dynamics.step(z, u_safe, dt)
        t += dt

        states.append(z)
        controls.append(u_safe)

    states = torch.stack(states, dim=1)
    controls = torch.stack(controls, dim=1)

    return states, controls


def plot_velocity_arrows(ax, states, config, color, skip=5):
    """
    Plot velocity arrows for double integrator trajectories.

    Args:
        ax: Matplotlib axis
        states: State trajectory (num_steps+1, state_dim)
        config: TrainingConfig
        color: Arrow color
        skip: Plot every skip-th arrow
    """
    states_np = states.cpu().numpy()
    positions = states_np[:, :config.position_dim]  # (num_steps+1, 2)
    velocities = states_np[:, config.position_dim:]  # (num_steps+1, 2)

    # Subsample for clarity
    positions = positions[::skip]
    velocities = velocities[::skip]

    # Plot arrows
    ax.quiver(
        positions[:, 0], positions[:, 1],
        velocities[:, 0], velocities[:, 1],
        color=color, alpha=0.3, scale=20, width=0.003
    )


def plot_training_curves(losses, save_path=None):
    """
    Plot training loss curves.

    Args:
        losses: List of (epoch, loss, running, terminal) tuples
        save_path: Path to save figure (shows if None)
    """
    epochs, total_losses, running_costs, terminal_costs = zip(*losses)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Total loss
    ax1.plot(epochs, total_losses, linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Total Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Running vs terminal costs
    ax2.plot(epochs, running_costs, label='Running Cost', linewidth=2)
    ax2.plot(epochs, terminal_costs, label='Terminal Cost', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Cost', fontsize=12)
    ax2.set_title('Cost Breakdown', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    else:
        plt.show()

    return fig, (ax1, ax2)
