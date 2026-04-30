"""
Quadrotor dynamics with Euler angles.

State: [x, y, z, ψ, θ, φ, vx, vy, vz, vψ, vθ, vφ]  (12D per agent)
Control: [thrust, τψ, τθ, τφ]  (4D per agent)

Dynamics:
    ṗ = v
    [ψ̇, θ̇, φ̇] = [vψ, vθ, vφ]
    v̇ = (thrust/m) * thrust_direction(ψ, θ, φ) - g*e3
    [v̇ψ, v̇θ, v̇φ] = [τψ, τθ, τφ]

For CBF purposes, we focus on position dynamics:
    ẍ = (thrust/m) * thrust_direction(ψ, θ, φ)_x - 0
    ÿ = (thrust/m) * thrust_direction(ψ, θ, φ)_y - 0
    z̈ = (thrust/m) * thrust_direction(ψ, θ, φ)_z - g

This gives RELATIVE DEGREE 2 for position-based barriers.
"""

import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from .base import ControlAffineDynamics


class Quadrotor(ControlAffineDynamics):
    """
    Quadrotor dynamics with Euler angles.

    Multi-agent support: n_agent quadrotors
    Total state: 12 * n_agent
    Total control: 4 * n_agent
    """

    def __init__(self, n_agent=1, mass=0.5, gravity=1.0):
        """
        Initialize quadrotor dynamics.

        Args:
            n_agent: Number of quadrotors
            mass: Mass of each quadrotor
            gravity: Gravity constant
        """
        state_dim = 12 * n_agent
        control_dim = 4 * n_agent
        relative_degree = 2  # Position barriers have relative degree 2

        super().__init__(state_dim, control_dim, relative_degree)

        self.n_agent = n_agent
        self.mass = mass
        self.gravity = gravity
        self.T_hover = mass * gravity

    def thrust_direction(self, angles):
        """
        Compute thrust direction from Euler angles.

        Args:
            angles: (batch, n_agent, 3) with [ψ, θ, φ]

        Returns:
            thrust_dir: (batch, n_agent, 3) unit direction
        """
        psi, theta, phi = angles[..., 0], angles[..., 1], angles[..., 2]

        sps, cps = torch.sin(psi), torch.cos(psi)
        sth, cth = torch.sin(theta), torch.cos(theta)
        sph, cph = torch.sin(phi), torch.cos(phi)

        # Thrust direction in world frame
        tx = sps * sph + cps * sth * cph
        ty = -cps * sph + sps * sth * cph
        tz = cth * cph

        return torch.stack([tx, ty, tz], dim=-1)

    def split_state(self, x):
        """
        Split state into components.

        Args:
            x: (batch, 12*n_agent)

        Returns:
            pos: (batch, n_agent, 3) position
            angles: (batch, n_agent, 3) Euler angles
            vel: (batch, n_agent, 3) linear velocity
            ang_vel: (batch, n_agent, 3) angular velocity
        """
        batch = x.shape[0]
        x_reshaped = x.reshape(batch, self.n_agent, 12)
        return (x_reshaped[..., 0:3], x_reshaped[..., 3:6],
                x_reshaped[..., 6:9], x_reshaped[..., 9:12])

    def f(self, x, t=0.0):
        """
        Drift dynamics f(x, t).  (time-invariant)

        Args:
            x: (batch, 12*n_agent)
            t: Current time (unused; accepted for interface consistency)

        Returns:
            drift: (batch, 12*n_agent)
        """
        batch = x.shape[0]
        pos, angles, vel, ang_vel = self.split_state(x)

        # Drift dynamics
        dpos = vel  # (batch, n_agent, 3)
        dangles = ang_vel  # (batch, n_agent, 3)

        # Gravity effect on velocity: broadcast to (batch, n_agent, 3)
        e3 = torch.tensor([0., 0., 1.], device=x.device, dtype=x.dtype)
        dvel = -self.gravity * e3.view(1, 1, 3).expand(batch, self.n_agent, 3)  # (batch, n_agent, 3)

        dang_vel = torch.zeros_like(ang_vel)  # (batch, n_agent, 3)

        drift = torch.cat([dpos, dangles, dvel, dang_vel], dim=-1)  # (batch, n_agent, 12)
        return drift.reshape(batch, self.state_dim)

    def g(self, x, t=0.0):
        """
        Control matrix g(x, t).  (time-invariant)

        For quadrotor:
        - Thrust affects linear acceleration via thrust direction
        - Angular accelerations directly control angular velocity

        Args:
            x: (batch, 12*n_agent)
            t: Current time (unused; accepted for interface consistency)

        Returns:
            G: (batch, 12*n_agent, 4*n_agent)
        """
        batch = x.shape[0]
        pos, angles, vel, ang_vel = self.split_state(x)

        # Compute thrust direction
        thrust_dir = self.thrust_direction(angles)  # (batch, n_agent, 3)

        # Build control matrix
        G = torch.zeros(batch, self.n_agent, 12, 4, device=x.device, dtype=x.dtype)

        for i in range(self.n_agent):
            # Thrust affects velocity (indices 6:9)
            G[:, i, 6:9, 0] = thrust_dir[:, i, :] / self.mass

            # Angular accelerations affect angular velocities (indices 9:12)
            G[:, i, 9:12, 1:4] = torch.eye(3, device=x.device, dtype=x.dtype)

        # Reshape to (batch, 12*n_agent, 4*n_agent)
        G_full = torch.zeros(batch, 12*self.n_agent, 4*self.n_agent,
                            device=x.device, dtype=x.dtype)
        for i in range(self.n_agent):
            G_full[:, 12*i:12*(i+1), 4*i:4*(i+1)] = G[:, i, :, :]

        return G_full

    def step(self, x, u, dt, t=0.0):
        """
        RK4 integration step, passing time to each sub-step.

        Args:
            x: (batch, 12*n_agent)
            u: (batch, 4*n_agent)
            dt: time step
            t: Current time. Passed to f and g at each RK4 sub-step,
               matching the reference rk4_step(z, u, ti, h, f) interface.

        Returns:
            x_next: (batch, 12*n_agent)
        """
        def dynamics(x_temp, u_temp, t_temp):
            f_x = self.f(x_temp, t_temp)
            g_x = self.g(x_temp, t_temp)
            # g_x @ u: (batch, 12*n_agent, 4*n_agent) @ (batch, 4*n_agent, 1)
            return f_x + (g_x @ u_temp.unsqueeze(-1)).squeeze(-1)

        k1 = dynamics(x,              u, t)
        k2 = dynamics(x + dt/2 * k1, u, t + dt/2)
        k3 = dynamics(x + dt/2 * k2, u, t + dt/2)
        k4 = dynamics(x + dt   * k3, u, t + dt)

        return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    def plot_trajectory(self, traj, obstacles, p_target, save_path="trajs.png", title="Quadrotor Trajectories"):
        """
        Plot 3D trajectory visualization with obstacles.

        Args:
            traj: (batch, state_dim, num_steps+1) trajectory tensor
            obstacles: list of SphericalObstacle objects
            p_target: (n_agent, 3) target positions
            save_path: where to save the plot
            title: plot title
        """
        fig = plt.figure(figsize=(16, 8))
        ax  = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122)

        colors = plt.cm.tab10(range(self.n_agent))

        # Convert to numpy for plotting (use first trajectory in batch)
        traj_np = traj[0].detach().cpu().numpy()  # (state_dim, num_steps+1)

        # Plot each agent's trajectory
        for a in range(self.n_agent):
            b = 12 * a  # Each agent has 12 states: [x, y, z, psi, theta, phi, vx, vy, vz, vpsi, vtheta, vphi]

            # 3D plot
            ax.plot(traj_np[b,   :],
                    traj_np[b+1, :],
                    traj_np[b+2, :],
                    color=colors[a], alpha=0.6, linewidth=1)
            # Starting point
            ax.scatter(traj_np[b, 0], traj_np[b+1, 0], traj_np[b+2, 0],
                       color=colors[a], s=20, marker='o')

            # 2D bird's-eye view
            ax2.plot(traj_np[b,   :],
                     traj_np[b+1, :],
                     color=colors[a], alpha=0.6, linewidth=1)
            ax2.scatter(traj_np[b, 0], traj_np[b+1, 0],
                        color=colors[a], s=20, marker='o')

        # Plot obstacles
        u_s = np.linspace(0, 2 * np.pi, 20)
        v_s = np.linspace(0,     np.pi, 20)
        xs  = np.outer(np.cos(u_s), np.sin(v_s))
        ys  = np.outer(np.sin(u_s), np.sin(v_s))
        zs  = np.outer(np.ones_like(u_s), np.cos(v_s))

        for obs in obstacles:
            center = obs.center.detach().cpu().numpy()  # (3,)
            cx, cy, cz = center[0], center[1], center[2]
            r = obs.radius
            eps = obs.epsilon

            # 3D obstacle sphere
            ax.plot_wireframe(cx + r * xs,
                              cy + r * ys,
                              cz + r * zs,
                              color='red', alpha=0.25, linewidth=0.5)
            # Safety margin
            ax.plot_wireframe(cx + (r + eps) * xs,
                              cy + (r + eps) * ys,
                              cz + (r + eps) * zs,
                              color='black', alpha=0.10, linewidth=0.5)

            # 2D circles (bird's-eye view)
            circle = plt.Circle((cx, cy), r, color='red', alpha=0.25)
            safe_circle = plt.Circle((cx, cy), r + eps,
                                     color='black', fill=False, alpha=0.4, linewidth=1)
            ax2.add_patch(circle)
            ax2.add_patch(safe_circle)

        # Plot targets
        tgt = p_target.detach().cpu().numpy()  # (n_agent, 3)
        ax.scatter(tgt[:, 0], tgt[:, 1], tgt[:, 2],
                   c='green', s=100, marker='X', label='Target', zorder=5)
        ax2.scatter(tgt[:, 0], tgt[:, 1],
                    c='green', s=100, marker='X', label='Target', zorder=5)

        # Formatting
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(title)
        ax.legend()
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(-1, 4)
        ax.set_zlim(0, 3)
        ax.view_init(azim=45)

        ax2.set_xlabel('X'); ax2.set_ylabel('Y')
        ax2.set_title("Bird's-eye view")
        ax2.set_aspect('equal')
        ax2.legend()
        ax2.set_xlim(-0.5, 3.5)
        ax2.set_ylim(-1, 4)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=400)
        plt.close()

    def __repr__(self):
        return (f"Quadrotor(n_agent={self.n_agent}, mass={self.mass}, "
                f"gravity={self.gravity}, state_dim={self.state_dim}, "
                f"control_dim={self.control_dim})")
