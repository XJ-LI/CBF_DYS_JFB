import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
from utils import DYSProjector, euler_step, rk4_step, ResBlock, ControlNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'


#### Settings for double integrator dynamics
A = torch.tensor([[0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0]]).to(device)
B = torch.tensor([[0.0, 0.0],
                  [0.0, 0.0],
                  [1.0, 0.0],
                  [0.0, 1.0]]).to(device)
                  
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
    output = 0.5 * torch.norm(z[:, :2] - p_target, dim=1)**2  # terminal cost on position only
    output = output.mean()
    assert output.shape == ()
    return output

#### HOCBF Settings
position_dimension = 2

def barrier_function(z, center, r_obs, eps):
    """ψ₀ = h(z) = ‖p − c‖² − r² − ε²   (position-only, relative degree 2)."""
    pos = z[:, :position_dimension]
    return torch.sum((pos - center)**2, dim=-1) - r_obs**2 - eps**2  # (batch,)
    
def evaluate_barriers(z, centers, r_obs, eps):
    """Returns (batch, num_obstacles) values of ψ₀ = h."""
    return torch.stack([barrier_function(z, c, r_obs, eps) for c in centers], dim=-1)
    
def gamma(x):
    """Class-K function used in both HOCBF lifts (α₁ and α₂)."""
    return 1.0 * x
    
def psi1_function(z, center, r_obs, eps):
    """ψ₁(z) = L_f h + γ(ψ₀)  =  2(p−c)ᵀv + γ(h(z))."""
    pos = z[:, :2]
    vel = z[:, 2:]
    dp  = pos - center
    h   = dp.pow(2).sum(-1) - r_obs**2 - eps**2
    return 2 * (dp * vel).sum(-1) + gamma(h)           # (batch,)

def evaluate_psi1(z, centers, r_obs, eps):
    """Returns (batch, num_obstacles) values of ψ₁."""
    return torch.stack([psi1_function(z, c, r_obs, eps) for c in centers], dim=-1)
    
def construct_cbf_constraints(z, centers, r_obs, eps):
    """
    Build the second-order HOCBF constraint  K_i u ≤ d_i  from  ψ₂ ≥ 0.

    System: ṗ = v,  v̇ = u  (double integrator, relative degree 2).

    Barrier hierarchy  (γ is the module-level class-K function)
    -----------------
    ψ₀ = h = ‖p − c‖² − r² − ε²
    ψ₁ = L_f h + γ(ψ₀)  =  2(p−c)ᵀv + γ(h)
    ψ₂ = ψ̇₁ + γ(ψ₁)
       = [2‖v‖² + 2(p−c)ᵀu + γ′(h)·L_f h] + γ(ψ₁)

    For the linear case  γ(x) = c·x,  γ′ = c  is constant, so
        γ′(h)·L_f h  =  γ(L_f h)
    and:
        ψ₂ = 2‖v‖² + 2(p−c)ᵀu + γ(L_f h) + γ(ψ₁) ≥ 0

    Rearranging to  K u ≤ d:
        K_i = −2(p − c_i)ᵀ                          shape (batch, 1, 2)
        d_i =  2‖v‖² + γ(L_f h) + γ(ψ₁)            shape (batch, 1, 1)

    Returns
        K_cbf : (batch, n_obs, 2)
        d_cbf : (batch, n_obs, 1)
    """
    batch = z.shape[0]
    pos = z[:, :2]   # (batch, 2)
    vel = z[:, 2:]   # (batch, 2)

    K_list, d_list = [], []
    for center in centers:
        dp   = pos - center                              # (batch, 2)
        h_i  = dp.pow(2).sum(-1) - r_obs**2 - eps**2     # ψ₀,  (batch,)
        Lf_h = 2 * (dp * vel).sum(-1)                   # L_f h = 2(p−c)ᵀv,  (batch,)
        psi1 = Lf_h + gamma(h_i)                        # ψ₁ = L_f h + γ(h),  (batch,)

        K_i = -2 * dp                                   # (batch, 2)
        d_i = (2 * vel.pow(2).sum(-1)                   # 2‖v‖²  =  L_ff h
               + gamma(Lf_h)                            # γ′(h)·L_f h = γ(L_f h)  [linear γ]
               + gamma(psi1))                           # γ(ψ₁)

        K_list.append(K_i.unsqueeze(1))        # (batch, 1, 2)
        d_list.append(d_i.view(batch, 1, 1))   # (batch, 1, 1)

    K_cbf = torch.cat(K_list, dim=1)   # (batch, n_obs, 2)
    d_cbf = torch.cat(d_list, dim=1)   # (batch, n_obs, 1)
    return K_cbf, d_cbf
        
        
def sample_initial_condition(z0_mean, z0_std, batch_size=64, a=-1, b=0.5, mode="gaussian"):
    """Sample initial conditions"""
    if mode == "gaussian":
        noise = torch.zeros(batch_size, 4, device=device)
        noise[:, :2] = torch.randn(batch_size, 2, device=device) 
        return z0_mean + z0_std * noise
        # return z0_mean + z0_std * torch.randn(batch_size, 4, device=device)
    else:
        noise = torch.zeros(batch_size, 4, device=device)
        noise[:, :2] = (b-a) * torch.rand(batch_size, 2, device=device) +a
        return noise
        
def compute_loss(u, z0, nt, f, p_target, obstacle_centers, r_obstacle, eps_safe,
                 alpha_running, alpha_terminal, proj, verbose=False, return_traj=True, dt=None):
    """
    Rolls out policy u for nt steps under the HOCBF filter, accumulating
    running and terminal costs.

    At each step:
      1. Query policy for u_nom.
      2. Build HOCBF constraints K_cbf u ≤ d_cbf (ψ₂ ≥ 0) at current state.
      3. If u_nom violates any constraint (K_cbf u_nom > d_cbf), solve the QP
         via DYS to obtain u_safe.
      4. Advance the state with the applied control via RK4.

    Args:
        dt: Physical time step. If None, defaults to 1/nt (normalized [0,1] horizon).
    """
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
        assert u_nom.shape == (batch_size, 2)

        # Build HOCBF constraints K_cbf u ≤ d_cbf (ψ₂ ≥ 0) at current state
        K_cbf, d_cbf = construct_cbf_constraints(z.detach(), obstacle_centers, r_obstacle, eps_safe)

        # Check whether u_nom violates ψ₂ ≥ 0 for any obstacle / batch element
        psi2_nom   = d_cbf - K_cbf @ u_nom.unsqueeze(-1)   # (batch, n_obs, 1)
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

        z  = rk4_step(z, current_u, ti, h, f)
        ti = ti + h

        if return_traj:
            traj[:, :, i + 1] = z.detach().clone()

        barrier_value_array[i] = evaluate_barriers(z, obstacle_centers, r_obstacle, eps_safe).min().item()
        running_cost = running_cost + h * lagrangian(current_u)

    running_cost  = running_cost.mean()
    terminal_cost = G(z, p_target)
    total_cost    = alpha_running * running_cost + alpha_terminal * terminal_cost

    assert traj.shape == (batch_size, state_dim, nt + 1) if return_traj else True

    return total_cost, running_cost, terminal_cost, isprojected, n_iters_array, max_res_norm_array, barrier_value_array, traj
    

#### plotter
def plot_trajectory(traj, obstacle_centers, obstacle_radius, p_target, title="Trajectory with Obstacles", eps_safe = 0.1, n_traj=10):
    plt.figure(figsize=(8, 8))
    for i in range(min(traj.shape[0], n_traj)):
        plt.plot(traj[i, 0, :], traj[i, 1, :], label='Trajectory', marker='o')

    # Plot obstacles as circles
    for center in obstacle_centers:
        circle = plt.Circle((center[0, 0].item(), center[0, 1].item()), obstacle_radius,
                            color='red', alpha=0.5)
        plt.gca().add_patch(circle)
        safety_circle = plt.Circle((center[0, 0].item(), center[0, 1].item()), obstacle_radius + eps_safe,
                           edgecolor='black', fill=False, linewidth=2)
        plt.gca().add_patch(safety_circle)

    # Plot target position
    plt.scatter(p_target[:, 0].item(), p_target[:, 1].item(), color='green', label='Target', s=100, marker='X')

    plt.xlabel('Position X')
    plt.ylabel('Position Y')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.savefig("trajs.png", bbox_inches="tight", dpi=400)
    plt.close()
    
