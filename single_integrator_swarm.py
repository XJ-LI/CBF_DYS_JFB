import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
from utils import DYSProjector, euler_step, rk4_step, ResBlock, ControlNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

n_agent = 50

A = torch.zeros((3, 3)).to(device)
B = torch.eye(3).to(device)

A = torch.block_diag(*([A] * n_agent))
B = torch.block_diag(*([B] * n_agent))


def f(z, u, t, A=A, B=B):
    """Dynamics: ż = Az + Bu"""
    output = (A @ z.T + B @ u.T).T
    assert output.shape == z.shape
    return output
    
def lagrangian(u):
    """Running cost"""
    return 0.5 * torch.norm(u, dim=-1)**2
    
def G(z, p_target):
    """Terminal cost"""
    # technically each agent can have different targets but here default we use a single shared target location
    pos = z.reshape(z.shape[0], n_agent, 3)   # (B, N, 3)
    output = 0.5 * ((pos - p_target.unsqueeze(0))**2).sum(dim=-1).sum(dim=-1)
    output = output.mean()
    assert output.shape == ()
    return output


#### CBF Settings for 3D single-integrator
def gamma(x):
    return 1.0 * x
    
def cylinder_barrier(z, cyl_center_xy, cyl_radius, eps):
    pos = z.reshape(z.shape[0], n_agent, 3)                          # (B, N, 3)
    dxy = pos[:, :, :2] - cyl_center_xy.view(1, 1, 2)                # (B, N, 2)
    h_cyl = (dxy ** 2).sum(dim=-1) - cyl_radius**2 - eps**2          # (B, N)
    return h_cyl   
    
def evaluate_barriers(z, cyl1_center_xy, cyl1_radius, cyl2_center_xy, cyl2_radius, eps):
    h_cyl1 = cylinder_barrier(z, cyl1_center_xy, cyl1_radius, eps)   # (B, N)
    h_cyl2 = cylinder_barrier(z, cyl2_center_xy, cyl2_radius, eps)   # (B, N)
    return torch.stack([h_cyl1, h_cyl2], dim=-1)                     # (B, N, 2)    
    
    
def construct_cbf_constraints(z, cyl1_center_xy, cyl1_radius, cyl2_center_xy, cyl2_radius, eps):
    batch = z.shape[0]
    pos = z.reshape(batch, n_agent, 3)                               # (B, N, 3)
    agent_idx = torch.arange(n_agent, device=z.device)

    # ----- Cylinder 1 -----
    dxy1 = pos[:, :, :2] - cyl1_center_xy.view(1, 1, 2)              # (B, N, 2)
    h_cyl1 = (dxy1 ** 2).sum(dim=-1) - cyl1_radius**2 - eps**2       # (B, N)

    K_cyl1 = torch.zeros(batch, n_agent, 3 * n_agent,
                         device=z.device, dtype=z.dtype)
    K_cyl1[:, agent_idx, 3 * agent_idx + 0] = -2.0 * dxy1[:, :, 0]
    K_cyl1[:, agent_idx, 3 * agent_idx + 1] = -2.0 * dxy1[:, :, 1]
    # z component stays zero
    d_cyl1 = gamma(h_cyl1).unsqueeze(-1)                             # (B, N, 1)

    # ----- Cylinder 2 -----
    dxy2 = pos[:, :, :2] - cyl2_center_xy.view(1, 1, 2)              # (B, N, 2)
    h_cyl2 = (dxy2 ** 2).sum(dim=-1) - cyl2_radius**2 - eps**2       # (B, N)

    K_cyl2 = torch.zeros(batch, n_agent, 3 * n_agent,
                         device=z.device, dtype=z.dtype)
    K_cyl2[:, agent_idx, 3 * agent_idx + 0] = -2.0 * dxy2[:, :, 0]
    K_cyl2[:, agent_idx, 3 * agent_idx + 1] = -2.0 * dxy2[:, :, 1]
    # z component stays zero
    d_cyl2 = gamma(h_cyl2).unsqueeze(-1)                             # (B, N, 1)

    K_cbf = torch.cat([K_cyl1, K_cyl2], dim=1)                       # (B, 2N, 3N)
    d_cbf = torch.cat([d_cyl1, d_cyl2], dim=1)                       # (B, 2N, 1)
    return K_cbf, d_cbf



def sample_initial_condition(batch_size=64, z0_std=0.1):
    """50 agents: x evenly spaced in [0,5], y=0, first 25 agents z=0, next 25 agents z=3"""
    # x positions equally spaced
    n_per_row = n_agent // 2
    x_row = torch.linspace(0.0, 5.0, n_per_row, device=device)

    # duplicate for the two z-layers
    x_positions = torch.cat([x_row, x_row], dim=0)

    # z layer assignment
    z_centers = torch.cat([torch.zeros(n_per_row, device=device),3.0 * torch.ones(n_per_row, device=device)], dim=0)

    z0 = torch.zeros(batch_size, 3 * n_agent, device=device)
    z0_view = z0.view(batch_size, n_agent, 3)

    z0_view[:, :, 0] = x_positions   # x
    z0_view[:, :, 1] = 0.0           # y
    z0_view[:, :, 2] = z_centers     # z
    z0_view += z0_std * torch.randn(batch_size, n_agent, 3, device=device)

    return z0


def compute_loss(u, z0, nt, f, p_target, cyl1_center_xy, cyl1_radius, cyl2_center_xy, cyl2_radius, eps_safe,
                 alpha_running, alpha_terminal, proj, verbose=False, return_traj=True, dt=None):

    running_cost = 0.
    z  = z0
    h  = dt if dt is not None else 1.0 / nt
    ti = torch.zeros(1, device=z0.device)
    batch_size = z0.shape[0]
    state_dim  = z0.shape[1]

    isprojected         = 0
    max_res_norm_array  = torch.zeros(nt)
    n_iters_array       = torch.zeros(nt)
    barrier_value_array = torch.zeros(nt)
    
    if return_traj:
        traj = torch.zeros(batch_size, state_dim, nt + 1, device=z0.device)
        traj[:, :, 0] = z0.detach().clone()
    else:
        traj = None
    
    for i in range(nt):
        u_nom = u(z, ti)
        assert u_nom.shape == (batch_size, 3 * n_agent)
        
        K_cbf, d_cbf = construct_cbf_constraints(z.detach(), cyl1_center_xy, cyl1_radius, cyl2_center_xy, cyl2_radius, eps_safe)
        assert K_cbf.shape == (batch_size, 2 * n_agent, 3 * n_agent)
        assert d_cbf.shape == (batch_size, 2 * n_agent, 1)
        
        # Check whether u_nom violates any CBF constraint
        cbf_margin = d_cbf - K_cbf @ u_nom.unsqueeze(-1)   # (batch, 2*n_agent, 1)
        cbf_viol = (-cbf_margin).clamp(min=0).max().item()
        is_projected = cbf_viol > 0
    

        if is_projected:
            current_u, _, info = proj(u_nom, K_cbf, d_cbf,
                                      max_iter=5000, tol=5e-3, verbose=verbose)
            isprojected           = 1
            n_iters_array[i]      = info['iters']
            max_res_norm_array[i] = info['final_residual']
    
            
            if verbose:
                cbf_safe = (d_cbf - K_cbf @ current_u.unsqueeze(-1)).min().item()
                print(f't {i*h:.3f}  CBF ACTIVE  '
                      f'cbf_nom_min={-cbf_viol:.2e}  '
                      f'cbf_safe_min={cbf_safe:.2e}  '
                      f'iters={info["iters"]:4d}  '
                      f'converged={info["converged"]}')
        else:
            current_u             = u_nom
            n_iters_array[i]      = 0
            max_res_norm_array[i] = 0.0

        assert current_u.shape == (batch_size, 3 * n_agent)
        z  = rk4_step(z, current_u, ti, h, f)
        ti = ti + h

        if return_traj:
            traj[:, :, i + 1] = z.detach().clone()
        
        barrier_value_array[i] = evaluate_barriers(z, cyl1_center_xy, cyl1_radius, cyl2_center_xy, cyl2_radius, eps_safe).min().item()
        running_cost = running_cost + h * lagrangian(current_u)

    running_cost  = running_cost.mean()
    terminal_cost = G(z, p_target)
    total_cost    = alpha_running * running_cost + alpha_terminal * terminal_cost

    assert traj.shape == (batch_size, state_dim, nt + 1) if return_traj else True
    return total_cost, running_cost, terminal_cost, isprojected, n_iters_array, max_res_norm_array, barrier_value_array, traj
    
    
def plot_trajectory(traj, cyl1_center_xy, cyl1_radius, cyl2_center_xy, cyl2_radius, p_target,
                    title="3D Trajectory with Obstacles", eps_safe=0.1):

    fig = plt.figure(figsize=(16, 8))  # slightly wider

    # --- NEW: 1x2 layout ---
    ax = fig.add_subplot(121, projection='3d')   # left: 3D
    ax2 = fig.add_subplot(122)                   # right: bird-eye (2D)

    n_agent = traj.shape[1] // 3
    colors = plt.cm.tab20(np.linspace(0, 1, n_agent))

    i = 0
    for a in range(n_agent):
        px, py, pz = 3*a, 3*a+1, 3*a+2

        x = traj[i, px, :].detach().cpu().numpy()
        y = traj[i, py, :].detach().cpu().numpy()
        z = traj[i, pz, :].detach().cpu().numpy()

        # ---- 3D plot ----
        ax.plot(x, y, z, color=colors[a], linewidth=1.5)
        ax.scatter(x[0], y[0], z[0], color=colors[a], s=20, marker='o')
        ax.scatter(x[-1], y[-1], z[-1], color=colors[a], s=30, marker='^')

        # ---- 2D bird-eye plot ----
        ax2.plot(x, y, color=colors[a], linewidth=1.5)
        ax2.scatter(x[0], y[0], color=colors[a], s=20, marker='o')
        ax2.scatter(x[-1], y[-1], color=colors[a], s=30, marker='^')
    
    z_vals = traj[i, 2::3, :].detach().cpu().numpy().reshape(-1)
    z_min = min(z_vals.min(), p_target[:, 2].detach().cpu().numpy().min()) - 1.0
    z_max = max(z_vals.max(), p_target[:, 2].detach().cpu().numpy().max()) + 1.0

    theta = np.linspace(0, 2 * np.pi, 40)
    z_cyl = np.linspace(z_min, z_max, 30)
    Theta, Z = np.meshgrid(theta, z_cyl)

    # ----- cylinder 1 -----
    cyl1_center_np = cyl1_center_xy.detach().cpu().numpy()

    for r, alpha, edge in [(cyl1_radius, 0.20, False), (cyl1_radius + eps_safe, 0.06, True)]:
        X = cyl1_center_np[0] + r * np.cos(Theta)
        Y = cyl1_center_np[1] + r * np.sin(Theta)

        if edge:
            ax.plot_wireframe(X, Y, Z, color='k', linewidth=0.5)
        else:
            ax.plot_surface(X, Y, Z, color='blue', alpha=alpha, linewidth=0)

    cyl1_circle = plt.Circle(cyl1_center_np, cyl1_radius, color='blue', alpha=0.2)
    cyl1_circle_safe = plt.Circle(cyl1_center_np, cyl1_radius + eps_safe,
                                  color='k', fill=False, linestyle='--', linewidth=1)
    ax2.add_patch(cyl1_circle)
    ax2.add_patch(cyl1_circle_safe)

    # ----- cylinder 2 -----
    cyl2_center_np = cyl2_center_xy.detach().cpu().numpy()

    for r, alpha, edge in [(cyl2_radius, 0.20, False), (cyl2_radius + eps_safe, 0.06, True)]:
        X = cyl2_center_np[0] + r * np.cos(Theta)
        Y = cyl2_center_np[1] + r * np.sin(Theta)

        if edge:
            ax.plot_wireframe(X, Y, Z, color='k', linewidth=0.5)
        else:
            ax.plot_surface(X, Y, Z, color='red', alpha=alpha, linewidth=0)

    cyl2_circle = plt.Circle(cyl2_center_np, cyl2_radius, color='red', alpha=0.2)
    cyl2_circle_safe = plt.Circle(cyl2_center_np, cyl2_radius + eps_safe,
                                  color='k', fill=False, linestyle='--', linewidth=1)
    ax2.add_patch(cyl2_circle)
    ax2.add_patch(cyl2_circle_safe)

    # ----- targets -----
    p_target_np = p_target.detach().cpu().numpy()

    ax.scatter(p_target_np[:, 0], p_target_np[:, 1], p_target_np[:, 2],
               color='green', s=60, marker='X', label='Target')

    ax2.scatter(p_target_np[:, 0], p_target_np[:, 1],
                color='green', s=60, marker='X', label='Target')

    # ----- labels -----
    ax.set_xlabel('Position X')
    ax.set_ylabel('Position Y')
    ax.set_zlabel('Position Z')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    ax2.set_xlabel('Position X')
    ax2.set_ylabel('Position Y')
    ax2.set_title("Bird's-eye view (XY)")
    ax2.set_aspect('equal')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("trajs.png", bbox_inches="tight", dpi=400)
    plt.close()
    