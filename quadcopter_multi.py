import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from utils import DYSProjector, rk4_step, ControlNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

n_agent = globals().get("n_agent", 5)

m = 0.5    # mass 
g = 1.0    # normalised gravity 

T_hover = m * g

STATE_DIM   = 12 * n_agent
CONTROL_DIM =  4 * n_agent

def split_state(z):
    """z : (B, 12·N)  →  pos, angles, vel, ang_vel  each  (B, N, 3).
    """
    B = z.shape[0]
    v = z.reshape(B, n_agent, 12)
    return v[:, :, 0:3], v[:, :, 3:6], v[:, :, 6:9], v[:, :, 9:12]

def thrust_direction(psi, theta, phi):
    sps, cps = torch.sin(psi),   torch.cos(psi)
    sth, cth = torch.sin(theta), torch.cos(theta)
    sph, cph = torch.sin(phi),   torch.cos(phi)

    f7 =  sps * sph + cps * sth * cph   # sin(ψ)sin(φ) + cos(ψ)sin(θ)cos(φ)
    f8 = -cps * sph + sps * sth * cph   # −cos(ψ)sin(φ) + sin(ψ)sin(θ)cos(φ)
    f9 =  cth * cph                      # cos(θ)cos(φ)

    return torch.stack([f7, f8, f9], dim=-1)   # (B, N, 3)

def f(z, u, t=None):
    B = z.shape[0]
    z_v = z.reshape(B, n_agent, 12)
    u_v = u.reshape(B, n_agent, 4)

    angles  = z_v[:, :, 3:6]    # (B, N, 3): [ψ, θ, φ]
    vel     = z_v[:, :, 6:9]    # (B, N, 3): [vx, vy, vz]
    ang_vel = z_v[:, :, 9:12]   # (B, N, 3): [vψ, vθ, vφ]

    u_thrust = u_v[:, :, 0]     # (B, N)    collective thrust
    tau      = u_v[:, :, 1:4]   # (B, N, 3) angular accelerations [τψ, τθ, τφ]

    psi, theta, phi = angles[:, :, 0], angles[:, :, 1], angles[:, :, 2]
    td = thrust_direction(psi, theta, phi)   # (B, N, 3)

    e3 = torch.tensor([0., 0., 1.], device=z.device, dtype=z.dtype)
    dp       = vel                                          # ṗ = v
    dangles  = ang_vel                                      # [ψ̇, θ̇, φ̇] = [vψ, vθ, vφ]
    dv       = (u_thrust / m).unsqueeze(-1) * td - g * e3  # v̇ = (u/m)td − g e3
    dang_vel = tau                                          # [v̇ψ, v̇θ, v̇φ] = [τψ, τθ, τφ]

    dz = torch.cat([dp, dangles, dv, dang_vel], dim=-1)    # (B, N, 12)
    return dz.reshape(B, STATE_DIM)

def lagrangian(u):
    return 0.5 * torch.norm(u, dim=-1) ** 2

def G(z, p_target):
    """
    Terminal cost: position error + velocity + attitude + angular-rate penalty.
    """
    B = z.shape[0]
    v = z.reshape(B, n_agent, 12)
    pos     = v[:, :, 0:3]
    angles  = v[:, :, 3:6]
    vel     = v[:, :, 6:9]
    ang_vel = v[:, :, 9:12]

    cost = (0.5 * ((pos - p_target.unsqueeze(0)) ** 2).sum(dim=-1).sum(dim=-1)
            + 0.5 * vel.pow(2).sum(dim=-1).sum(dim=-1)
            + 0.5 * angles.pow(2).sum(dim=-1).sum(dim=-1)
            + 0.5 * ang_vel.pow(2).sum(dim=-1).sum(dim=-1))
    return cost.mean()

# HOCBF settings
def gamma(x):
    return 1.0 * x

def barrier_function(z, center, r_obs, eps):
    # ball barrier
    pos, _, _, _ = split_state(z)
    return ((pos - center) ** 2).sum(dim=-1) - r_obs ** 2 - eps ** 2

def evaluate_barriers(z, centers, r_obs, eps):
    return torch.stack([barrier_function(z, c, r_obs, eps) for c in centers], dim=-1)

def psi1_function(z, center, r_obs, eps):
    pos, _, vel, _ = split_state(z)
    dp = pos - center
    h  = dp.pow(2).sum(dim=-1) - r_obs ** 2 - eps ** 2
    return 2.0 * (dp * vel).sum(dim=-1) + gamma(h)

def evaluate_psi1(z, centers, r_obs, eps):
    return torch.stack([psi1_function(z, c, r_obs, eps) for c in centers], dim=-1)

def construct_cbf_constraints(z, centers, r_obs, eps):
    B     = z.shape[0]
    n_obs = len(centers)

    pos, angles, vel, _ = split_state(z)
    psi, theta, phi = angles[:, :, 0], angles[:, :, 1], angles[:, :, 2]
    td = thrust_direction(psi, theta, phi)   # (B, N, 3)

    K_list, d_list = [], []

    for center in centers:
        dp   = pos - center                                       # (B, N, 3)
        h    = dp.pow(2).sum(dim=-1) - r_obs**2 - eps**2         # (B, N)
        Lf_h = 2.0 * (dp * vel).sum(dim=-1)                      # (B, N)
        psi1 = Lf_h + gamma(h)                                   # (B, N)

        # d_i = 2‖v‖² − 2g·dp_z + γ(Lf_h) + γ(ψ₁)
        d_vals = (2.0 * vel.pow(2).sum(dim=-1)
                  - 2.0 * g * dp[:, :, 2]
                  + gamma(Lf_h)
                  + gamma(psi1))                                  # (B, N)

        for i in range(n_agent):
            K_row         = torch.zeros(B, CONTROL_DIM, device=z.device, dtype=z.dtype)
            K_row[:, 4*i] = -(2.0 / m) * (dp[:, i, :] * td[:, i, :]).sum(dim=-1)

            K_list.append(K_row.unsqueeze(1))          # (B, 1, 4N)
            d_list.append(d_vals[:, i].view(B, 1, 1))  # (B, 1, 1)

    K_cbf = torch.cat(K_list, dim=1)   # (B, N·n_obs, 4N)
    d_cbf = torch.cat(d_list, dim=1)   # (B, N·n_obs, 1)
    return K_cbf, d_cbf

def sample_initial_condition(batch_size=64, z0_std=0.02):
    if n_agent == 5:
        x_min, x_max = 1.1, 1.9
    else:
        x_min, x_max = 0.2, 2.8

    x_positions = torch.linspace(x_min, x_max, n_agent, device=device)

    z0 = torch.zeros(batch_size, STATE_DIM, device=device)
    z0_view = z0.view(batch_size, n_agent, 12)

    z0_view[:, :, 0] = x_positions
    z0_view[:, :, 1] = -0.5
    z0_view[:, :, 2] = 1.0
    z0_view[:, :, :2] += z0_std * torch.randn(batch_size, n_agent, 2, device=device)
    return z0


def compute_loss(u, z0, nt, f, p_target, obstacle_centers, r_obstacle, eps_safe, alpha_running, alpha_terminal, proj, verbose=False, return_traj=True, dt=None):
    running_cost = 0.0
    z  = z0
    h  = dt if dt is not None else 1.0 / nt
    ti = torch.zeros(1, device=z0.device)
    batch_size = z0.shape[0]

    # Hover-thrust bias: every 4th entry (thrust u_i) gets +m·g
    hover_bias      = torch.zeros(CONTROL_DIM, device=z0.device, dtype=z0.dtype)
    hover_bias[::4] = T_hover

    isprojected         = 0
    max_res_norm_array  = torch.zeros(nt)
    n_iters_array       = torch.zeros(nt)
    barrier_value_array = torch.zeros(nt)

    if return_traj:
        traj = torch.zeros(batch_size, STATE_DIM, nt + 1, device=z0.device)
        traj[:, :, 0] = z0.detach().clone()
    else:
        traj = None

    for i in range(nt):
        u_nom = u(z, ti) + hover_bias
        assert u_nom.shape == (batch_size, CONTROL_DIM)

        K_cbf, d_cbf = construct_cbf_constraints(
            z.detach(), obstacle_centers, r_obstacle, eps_safe)
        assert K_cbf.shape == (batch_size, n_agent * len(obstacle_centers), CONTROL_DIM)
        assert d_cbf.shape == (batch_size, n_agent * len(obstacle_centers), 1)

        psi2_nom   = d_cbf - K_cbf @ u_nom.unsqueeze(-1)
        hocbf_viol = (-psi2_nom).clamp(min=0).max().item()
        is_projected = hocbf_viol > 0

        if is_projected:
            current_u, _, info = proj(u_nom, K_cbf, d_cbf,
                                      max_iter=5000, tol=5e-3, verbose=verbose)
            isprojected           = 1
            n_iters_array[i]      = info['iters']
            max_res_norm_array[i] = info['final_residual']

            if verbose:
                psi2_safe = (d_cbf - K_cbf @ current_u.unsqueeze(-1)).min().item()
                print(f't {i*h:.3f}  CBF ACTIVE  '
                      f'ψ₂_nom_min={-hocbf_viol:.2e}  '
                      f'ψ₂_safe_min={psi2_safe:.2e}  '
                      f'iters={info["iters"]:4d}  '
                      f'converged={info["converged"]}')
        else:
            current_u             = u_nom
            n_iters_array[i]      = 0
            max_res_norm_array[i] = 0.0

        assert current_u.shape == (batch_size, CONTROL_DIM)
        z  = rk4_step(z, current_u, ti, h, f)
        ti = ti + h

        if return_traj:
            traj[:, :, i + 1] = z.detach().clone()

        barrier_value_array[i] = evaluate_barriers(
            z, obstacle_centers, r_obstacle, eps_safe).min().item()

        vel          = z.reshape(z.shape[0], n_agent, 12)[:, :, 6:9]
        vel_penalty  = 0.5 * vel.pow(2).sum(dim=(1, 2))
        running_cost = running_cost + h * lagrangian(current_u) + 2.0 * h * vel_penalty

    running_cost  = running_cost.mean()
    terminal_cost = G(z, p_target)
    total_cost    = alpha_running * running_cost + alpha_terminal * terminal_cost

    assert traj.shape == (batch_size, STATE_DIM, nt + 1) if return_traj else True

    return total_cost, running_cost, terminal_cost, isprojected, n_iters_array, max_res_norm_array, barrier_value_array, traj


# ── Plotting ───────────────────────────────────────────────────────────────────
def plot_trajectory(traj, obstacle_centers, obstacle_radius, p_target, title="Quadrotor Trajectories", eps_safe=0.1):
    fig = plt.figure(figsize=(16, 8))
    ax  = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)

    colors = plt.cm.tab10(range(n_agent))
    i = 0   # only visualize one trajectory
    for a in range(n_agent):
        b = 12 * a
        ax.plot(traj[i, b,   :],
                traj[i, b+1, :],
                traj[i, b+2, :],
                color=colors[a], alpha=0.6, linewidth=1)
        ax.scatter(traj[i, b, 0], traj[i, b+1, 0], traj[i, b+2, 0],
                   color=colors[a], s=20, marker='o')

        ax2.plot(traj[i, b,   :],
                 traj[i, b+1, :],
                 color=colors[a], alpha=0.6, linewidth=1)
        ax2.scatter(traj[i, b, 0], traj[i, b+1, 0],
                    color=colors[a], s=20, marker='o')

    # Obstacle spheres (wireframe)
    u_s = np.linspace(0, 2 * np.pi, 20)
    v_s = np.linspace(0,     np.pi, 20)
    xs  = np.outer(np.cos(u_s), np.sin(v_s))
    ys  = np.outer(np.sin(u_s), np.sin(v_s))
    zs  = np.outer(np.ones_like(u_s), np.cos(v_s))

    for center in obstacle_centers:
        cx, cy, cz = center[0, 0], center[0, 1], center[0, 2]
        ax.plot_wireframe(cx + obstacle_radius * xs,
                          cy + obstacle_radius * ys,
                          cz + obstacle_radius * zs,
                          color='red', alpha=0.25, linewidth=0.5)
        ax.plot_wireframe(cx + (obstacle_radius + eps_safe) * xs,
                          cy + (obstacle_radius + eps_safe) * ys,
                          cz + (obstacle_radius + eps_safe) * zs,
                          color='black', alpha=0.10, linewidth=0.5)

        circle = plt.Circle((cx, cy), obstacle_radius, color='red', alpha=0.25)
        safe_circle = plt.Circle((cx, cy), obstacle_radius + eps_safe,
                                 color='black', fill=False, alpha=0.4, linewidth=1)
        ax2.add_patch(circle)
        ax2.add_patch(safe_circle)

    tgt = p_target.detach().cpu().numpy()
    ax.scatter(tgt[:, 0], tgt[:, 1], tgt[:, 2],
               c='green', s=100, marker='X', label='Target', zorder=5)
    ax2.scatter(tgt[:, 0], tgt[:, 1],
                c='green', s=100, marker='X', label='Target', zorder=5)

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
    plt.savefig("trajs.png", bbox_inches="tight", dpi=400)
    plt.close()
