import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
from utils import DYSProjector, euler_step, rk4_step, ResBlock, ControlNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'


n_agent = 6

#### Settings for double integrator dynamics
A = torch.tensor([[0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0]]).to(device)
B = torch.tensor([[0.0, 0.0],
                  [0.0, 0.0],
                  [1.0, 0.0],
                  [0.0, 1.0]]).to(device)

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
    pos = z[:, :4*n_agent].reshape(z.shape[0], n_agent, 4)[:, :, :2]          # (B, N, 2)
    output = 0.5 * ((pos - p_target.unsqueeze(0))**2).sum(dim=-1).sum(dim=-1)              # (B,)
    vel = z[:, :4*n_agent].reshape(z.shape[0], n_agent, 4)[:, :, 2:]          # (B, N, 2) 
    output = output + 0.5 * (vel**2).sum(dim=-1).sum(dim=-1)                  # also add velocity penalty to stabilize behavior
    output = output.mean()
    assert output.shape == ()
    return output    
    
#### HOCBF Settings
def split_state(z):
    z_view = z.reshape(z.shape[0], n_agent, 4)   # (B, N, 4)
    pos = z_view[:, :, :2]                       # (B, N, 2)
    vel = z_view[:, :, 2:]                       # (B, N, 2)
    return pos, vel    

def barrier_function(z, center, r_obs, eps):
    pos, _ = split_state(z)                                  # (B, N, 2)
    return ((pos - center)**2).sum(dim=-1) - r_obs**2 - eps**2    
    
def evaluate_barriers(z, centers, r_obs, eps):
    return torch.stack([barrier_function(z, c, r_obs, eps) for c in centers],dim=-1)    
    
def gamma(x):
    return 1.0 * x    
    
def psi1_function(z, center, r_obs, eps):
    pos, vel = split_state(z)                                # (B, N, 2), (B, N, 2)
    dp = pos - center                                        # (B, N, 2)
    h = (dp**2).sum(dim=-1) - r_obs**2 - eps**2             # (B, N)
    return 2 * (dp * vel).sum(dim=-1) + gamma(h)    


def evaluate_psi1(z, centers, r_obs, eps):
    return torch.stack([psi1_function(z, c, r_obs, eps) for c in centers],dim=-1)

    
def construct_cbf_constraints(z, centers, r_obs, eps):
    batch = z.shape[0]
    n_obs = len(centers)

    # z -> (batch, n_agent, 4), with per-agent state [px, py, vx, vy]
    z_view = z.reshape(batch, n_agent, 4)
    pos = z_view[:, :, :2]   # (batch, n_agent, 2)
    vel = z_view[:, :, 2:]   # (batch, n_agent, 2)

    K_list = []
    d_list = []

    for center in centers:
        dp = pos - center                                      # (batch, n_agent, 2)
        h = dp.pow(2).sum(dim=-1) - r_obs**2 - eps**2         # (batch, n_agent)
        Lf_h = 2 * (dp * vel).sum(dim=-1)                     # (batch, n_agent)
        psi1 = Lf_h + gamma(h)                                # (batch, n_agent)

        d = (2 * vel.pow(2).sum(dim=-1)                       # (batch, n_agent)
             + gamma(Lf_h)
             + gamma(psi1))

        # Build one row per agent for this obstacle
        for i in range(n_agent):
            K_i = torch.zeros(batch, 2 * n_agent, device=z.device, dtype=z.dtype)
            K_i[:, 2*i:2*i+2] = -2 * dp[:, i, :]              # only agent i control block is active

            d_i = d[:, i]                                     # (batch,)

            K_list.append(K_i.unsqueeze(1))                   # (batch, 1, 2*n_agent)
            d_list.append(d_i.view(batch, 1, 1))             # (batch, 1, 1)

    K_cbf = torch.cat(K_list, dim=1)                          # (batch, n_agent*n_obs, 2*n_agent)
    d_cbf = torch.cat(d_list, dim=1)                          # (batch, n_agent*n_obs, 1)
    return K_cbf, d_cbf
    

def sample_initial_condition(batch_size=64, z0_std=0.1, mode="gaussian"):
    """6 agents equally spaced clockwise on a circle around [1.5, 1.5]."""
    center = torch.tensor([1.5, 1.5], device=device)
    radius = (2 * 2**2) ** 0.5
    angles = -2 * torch.pi * torch.arange(n_agent, device=device) / n_agent

    z0 = torch.zeros(batch_size, 4 * n_agent, device=device)
    z0_view = z0.view(batch_size, n_agent, 4)
    z0_view[:, :, 0] = center[0] + radius * torch.cos(angles)
    z0_view[:, :, 1] = center[1] + radius * torch.sin(angles)

    if mode == "gaussian":
        z0_view[:, :, :2] += z0_std * torch.randn(batch_size, n_agent, 2, device=device)
    elif mode == "uniform":
        z0_view[:, :, :2] += z0_std * (2 * torch.rand(batch_size, n_agent, 2, device=device) - 1)

    return z0   
    
    
def compute_loss(u, z0, nt, f, p_target, obstacle_centers, r_obstacle, eps_safe,
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
        assert u_nom.shape == (batch_size, 2 * n_agent)

        # Build HOCBF constraints K_cbf u ≤ d_cbf (ψ₂ ≥ 0) at current state
        K_cbf, d_cbf = construct_cbf_constraints(z.detach(), obstacle_centers, r_obstacle, eps_safe)
        assert K_cbf.shape == (batch_size, n_agent * len(obstacle_centers), 2 * n_agent)
        assert d_cbf.shape == (batch_size, n_agent * len(obstacle_centers), 1)

        # Check whether u_nom violates ψ₂ ≥ 0 for any obstacle / batch element
        psi2_nom   = d_cbf - K_cbf @ u_nom.unsqueeze(-1)   # (batch, n_agent * n_obs, 1)
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
        
        assert current_u.shape == (batch_size, 2 * n_agent)
        z  = rk4_step(z, current_u, ti, h, f)
        ti = ti + h

        if return_traj:
            traj[:, :, i + 1] = z.detach().clone()

        barrier_value_array[i] = evaluate_barriers(z, obstacle_centers, r_obstacle, eps_safe).min().item()
        running_cost = running_cost + h * lagrangian(current_u) + 2 * h * (0.5 * (z.view(z.shape[0], 6, 4)[:, :, 2:]**2).sum(dim=(1,2)))  # add velocity cost here as well to avoid weird behavior

    running_cost  = running_cost.mean()
    terminal_cost = G(z, p_target)
    total_cost    = alpha_running * running_cost + alpha_terminal * terminal_cost

    assert traj.shape == (batch_size, state_dim, nt + 1) if return_traj else True

    return total_cost, running_cost, terminal_cost, isprojected, n_iters_array, max_res_norm_array, barrier_value_array, traj
    

#### plotter
def plot_trajectory(traj, obstacle_centers, obstacle_radius, p_target, title="Trajectory with Obstacles", eps_safe = 0.1, n_traj=10):
    plt.figure(figsize=(8, 8))
    n_agent = traj.shape[1] // 4
    
    colors = plt.cm.tab10(range(n_agent))
    for i in range(min(traj.shape[0], n_traj)):
        for a in range(n_agent):
            px = 4 * a
            py = 4 * a + 1
            plt.plot(traj[i, px, :], traj[i, py, :], marker='o', markersize=4, color=colors[a])

    # Plot obstacles as circles
    for center in obstacle_centers:
        circle = plt.Circle((center[0, 0].item(), center[0, 1].item()), obstacle_radius,
                            color='red', alpha=0.5)
        plt.gca().add_patch(circle)
        safety_circle = plt.Circle((center[0, 0].item(), center[0, 1].item()), obstacle_radius + eps_safe,
                           edgecolor='black', fill=False, linewidth=2)
        plt.gca().add_patch(safety_circle)

    # Plot target position
    plt.scatter(p_target[:, 0].detach().cpu().numpy(), p_target[:, 1].detach().cpu().numpy(), color='green', label='Target', s=100, marker='X')

    plt.xlabel('Position X')
    plt.ylabel('Position Y')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.savefig("trajs.png", bbox_inches="tight", dpi=400)
    plt.close()
    
