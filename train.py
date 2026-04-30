import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
import os
import psutil
from utils import DYSProjector, euler_step, rk4_step, ResBlock, ControlNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import argparse


# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--problem", choices=['double_integrator_single', "double_integrator_multi", "single_integrator_swarm", "quadcopter_multi", "quadcopter_swarm"])
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--lr_decay", type=int, default=600)
parser.add_argument("--load_path", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--grad_mode", choices=["jfb", "ad"], default="jfb")
parser.add_argument("--plot", action="store_true")

parser.add_argument("--hidden_dim", type=int, default=64)
parser.add_argument("--n_blocks", type=int, default=3)
args = parser.parse_args()


n_epochs = args.epochs
lr = args.lr
lr_decay = args.lr_decay
load_path = args.load_path
seed = args.seed
torch.manual_seed(seed)
problem = args.problem
hidden_dim = args.hidden_dim
n_blocks = args.n_blocks
process = psutil.Process(os.getpid())

# common settings
T = 10.0
dt = 0.2
num_steps = int(T / dt)

if problem == "double_integrator_single":
    weight_decay = 1e-4
else:
    weight_decay = 1e-3
alpha_running = 1
alpha_terminal = 2e1

log_every = 1
batch_size = 32
plot_freq = 50

if problem == "double_integrator_single":
    from double_integrator_single import A, B, f, lagrangian, G, position_dimension, barrier_function, evaluate_barriers, gamma, psi1_function, evaluate_psi1, construct_cbf_constraints, sample_initial_condition, compute_loss, plot_trajectory
    p_target = 3 * torch.tensor([1.0, 1.0]).to(device).view(1, 2)
    z_target = torch.cat([p_target, torch.zeros(1, 2).to(device)], dim=-1)
    p0 = torch.zeros(1, 2).to(device)
    v0 = torch.zeros(1, 2).to(device)
    z0 = torch.cat([p0, v0], dim=-1)
    z0_std = 1e-1
    
    # Obstacles
    obstacle_center_1 = torch.tensor([0.4, 1.0]).view(1, 2).to(device)
    obstacle_center_2 = torch.tensor([2.2, 2.2]).view(1, 2).to(device)
    obstacle_center_3 = torch.tensor([2.4, 0.6]).view(1, 2).to(device)
    obstacle_centers = [obstacle_center_1, obstacle_center_2, obstacle_center_3]
    obstacle_radius = 0.3
    eps_safe = 1e-1
    
    net = ControlNet(input_dim=5, hidden_dim=hidden_dim, output_dim=2, n_blocks=n_blocks).to(device)
    
elif problem == "double_integrator_multi":
    from double_integrator_multi import n_agent, A, B, f, lagrangian, G, barrier_function, evaluate_barriers, gamma, psi1_function, evaluate_psi1, construct_cbf_constraints, sample_initial_condition, compute_loss, plot_trajectory
    center = torch.tensor([1.5, 1.5], device=device)
    radius_target = (2 * 0.2**2) ** 0.5
    angles = -2 * torch.pi * torch.arange(n_agent, device=device) / n_agent
    p_target = center.view(1, 2) + radius_target * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    z0_std = 5e-2

    
    # Obstacles
    obstacle_center_1 = torch.tensor([0.63, 1.0]).view(1, 2).to(device)
    obstacle_center_2 = torch.tensor([1.5, 2.5]).view(1, 2).to(device)
    obstacle_center_3 = torch.tensor([2.37, 1.0]).view(1, 2).to(device)
    obstacle_centers = [obstacle_center_1, obstacle_center_2, obstacle_center_3]

    obstacle_radius = 0.35
    eps_safe = 0.15
    
    net = ControlNet(input_dim=25, hidden_dim=hidden_dim, output_dim=12, n_blocks=n_blocks).to(device)

elif problem == "single_integrator_swarm":
    from single_integrator_swarm import n_agent, A, B, f, lagrangian, G, cylinder_barrier, gamma, evaluate_barriers, construct_cbf_constraints, sample_initial_condition, compute_loss, plot_trajectory
    cyl1_center_xy = torch.tensor([1.5, 3.5], device=device)
    cyl1_radius = 0.5
    cyl2_center_xy = torch.tensor([4.0, 2.0], device=device)
    cyl2_radius = 0.7
    target_center = torch.tensor([2.5, 5.0, 1.5], device=device)  # (3,)
    p_target = target_center.unsqueeze(0).repeat(n_agent, 1)       # (50, 3)
    z0_std = 5e-2  
    eps_safe = 0.1
    
    net = ControlNet(input_dim=151, hidden_dim=hidden_dim, output_dim=150, n_blocks=n_blocks).to(device)
    

elif problem == "quadcopter_multi" or problem == "quadcopter_swarm":
    if problem == "quadcopter_multi":
        import importlib
        import quadcopter_multi
        quadcopter_multi.n_agent = 5
        importlib.reload(quadcopter_multi)

        from quadcopter_multi import n_agent, m, g, T_hover, STATE_DIM, CONTROL_DIM, split_state, thrust_direction, f, lagrangian, G, barrier_function, evaluate_barriers, psi1_function, evaluate_psi1, construct_cbf_constraints, sample_initial_condition, compute_loss, plot_trajectory
        
        obstacle_center_1 = torch.tensor([0.63, 1.0, 1.0]).view(1, 3).to(device)
        obstacle_center_2 = torch.tensor([1.5, 2.5, 0.8]).view(1, 3).to(device)
        obstacle_center_3 = torch.tensor([2.37, 1.0, 1.0]).view(1, 3).to(device)
        obstacle_centers = [obstacle_center_1, obstacle_center_2, obstacle_center_3]
        obstacle_radius = 0.35
        eps_safe = 0.15
        
        p_target     = torch.zeros(n_agent, 3, device=device)
        x_min, x_max = 1.1, 1.9
        x_positions = torch.linspace(x_min, x_max, n_agent, device=device)

        p_target[:, 0] = x_positions   
        p_target[:, 1] = 3.5          
        p_target[:, 2] = 1.0          
        alpha_terminal_final = 2e2
        z0_std = 4e-2
        
    elif problem == "quadcopter_swarm":
        import importlib
        import quadcopter_multi
        quadcopter_multi.n_agent = 30
        importlib.reload(quadcopter_multi)

        from quadcopter_multi import n_agent, m, g, T_hover, STATE_DIM, CONTROL_DIM, split_state, thrust_direction, f, lagrangian, G, barrier_function, evaluate_barriers, psi1_function, evaluate_psi1, construct_cbf_constraints, sample_initial_condition, compute_loss, plot_trajectory
        
        obstacle_center_1 = torch.tensor([0.63, 1.0, 1.0]).view(1, 3).to(device)
        obstacle_center_2 = torch.tensor([1.5, 2.5, 0.8]).view(1, 3).to(device)
        obstacle_center_3 = torch.tensor([2.37, 1.0, 1.0]).view(1, 3).to(device)
        obstacle_centers = [obstacle_center_1, obstacle_center_2, obstacle_center_3]
        obstacle_radius = 0.35
        eps_safe = 0.15

        p_target     = torch.zeros(n_agent, 3, device=device)
        x_min, x_max = 0.2, 2.8
        x_positions = torch.linspace(x_min, x_max, n_agent, device=device)
        p_target[:, 0] = x_positions   
        p_target[:, 1] = 3.5          
        p_target[:, 2] = 1.5 
        alpha_terminal_final = 5e2
        z0_std = 2e-2      
        
    net = ControlNet(input_dim=STATE_DIM + 1, hidden_dim=hidden_dim, output_dim=CONTROL_DIM, n_blocks=n_blocks).to(device)



optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
proj = DYSProjector(grad_mode=args.grad_mode).to(device)
print('Number of parameters in control net:', sum(p.numel() for p in net.parameters()))
    
if problem == "double_integrator_single" or problem == "double_integrator_multi" or problem == "single_integrator_swarm":
    def u_fn(z, ti):
        return net(z, ti)
elif problem == "quadcopter_multi" or problem == "quadcopter_swarm":   
    T_dev_scale = 0.5    # max thrust deviation from hover  (u ∈ [T_hover - scale, T_hover + scale])
    tau_scale   = 0.1    # max angular acceleration
    def u_fn(z, t):
        raw = net(z, t).reshape(z.shape[0], n_agent, 4)
        scaled = torch.cat([
            T_dev_scale * torch.tanh(raw[:, :, 0:1]),
            tau_scale   * torch.tanh(raw[:, :, 1:4]),
        ], dim=-1)
        return scaled.reshape(z.shape[0], CONTROL_DIM)

# Training loop
loss_history = []
run_history = []
term_history = []
proj_history = []
n_iters_history = []
max_res_norm_history = []
barrier_function_history = []
grad_norm_history = []    
    
    
for epoch in range(1, n_epochs+1):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    optimizer.zero_grad()
    
    if problem == "double_integrator_single":
        z0_sample = sample_initial_condition(z0, z0_std, batch_size=batch_size)
    else:
        z0_sample = sample_initial_condition(batch_size, z0_std)
    
    if problem == "single_integrator_swarm":
        total_cost, running_cost, terminal_cost, isprojected, n_iters_array, max_res_norm_array, barrier_value_array, traj = compute_loss(
            u_fn, z0_sample, num_steps, f, p_target, cyl1_center_xy, cyl1_radius, cyl2_center_xy, cyl2_radius, eps_safe,
            alpha_running, alpha_terminal, proj, dt = dt)
    else:
        total_cost, running_cost, terminal_cost, isprojected, n_iters_array, max_res_norm_array, barrier_value_array, traj = compute_loss(
                u_fn, z0_sample, num_steps, f, p_target, obstacle_centers, obstacle_radius, eps_safe,
                alpha_running, alpha_terminal, proj, dt = dt)

    total_cost.backward()
    optimizer.step()
    
    total_grad_norm = 0.0
    for param in net.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_grad_norm += param_norm.item() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    
    end_time = time.time()
    memory_MB = process.memory_info().rss / 1024 / 1024
    gpu_max_memory_MB = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0


    loss_history.append(total_cost.item())
    run_history.append(running_cost.item())
    term_history.append(terminal_cost.item())
    proj_history.append(isprojected)
    n_iters_history.append(n_iters_array.max().item())
    max_res_norm_history.append(max_res_norm_array.max().item())
    barrier_function_history.append(barrier_value_array.min().item())
    grad_norm_history.append(total_grad_norm)

    if epoch % log_every == 0:
            print(f"epoch {epoch:4d} total={total_cost.item():.4e}"
                  f"  L={running_cost.item():.4e}"
                  f"  G={terminal_cost.item():.4e}"
                  f"  proj={int(isprojected)}"
                  f"  res={max_res_norm_array.max().item():.2e}"
                  f"  h={barrier_value_array.min().item():.2e}"
                  f"  iters={int(n_iters_array.mean())}"
                  f"  grad={total_grad_norm:.2e}"
                  f"  t={end_time - start_time:.2f}s"
                  f"  mem={memory_MB:.1f}MB"
                  f"  gpu_mem={gpu_max_memory_MB:.1f}MB")

    if problem == "double_integrator_single":
        if args.plot and (epoch % plot_freq == 0):
            print("  Plotting trajectory...")
            plot_trajectory(traj.cpu().numpy(),
                        [obstacle_center_1.cpu().numpy(), obstacle_center_2.cpu().numpy(), obstacle_center_3.cpu().numpy()],
                         obstacle_radius, p_target)
        if epoch % 10 == 0:
            alpha_terminal += 5
            print("new alpha_terminal: ", alpha_terminal)
    elif problem == "double_integrator_multi":
        if args.plot and (epoch % plot_freq == 0):
            print("  Plotting trajectory...")
            plot_trajectory(traj.cpu().numpy(),
                          [obstacle_center_1.cpu().numpy(), obstacle_center_2.cpu().numpy(), obstacle_center_3.cpu().numpy()],
                          obstacle_radius, p_target, eps_safe=eps_safe)

        if epoch % 10 == 0:
            if alpha_terminal <= 5e2:
                alpha_terminal += 5
                print("new alpha_terminal: ", alpha_terminal)
    elif problem == "single_integrator_swarm":
        if args.plot and (epoch % plot_freq == 0):
            print("  Plotting trajectory...")
            plot_trajectory(traj, cyl1_center_xy, cyl1_radius, cyl2_center_xy, cyl2_radius, p_target, eps_safe=eps_safe)

        if epoch % 20 == 0:
            if alpha_terminal <= 5e2:
                alpha_terminal += 5
                print("new alpha_terminal: ", alpha_terminal)
    elif problem == "quadcopter_multi" or problem == "quadcopter_swarm":
        if args.plot and (epoch % plot_freq == 0):
            print("  Plotting trajectory...")
            plot_trajectory(traj.cpu().numpy(),
                          [obstacle_center_1.cpu().numpy(), obstacle_center_2.cpu().numpy(), obstacle_center_3.cpu().numpy()],
                          obstacle_radius, p_target, eps_safe=eps_safe)
        if epoch % 20 == 0:
            if alpha_terminal <= alpha_terminal_final:
                alpha_terminal += 5
                print("new alpha_terminal: ", alpha_terminal)
                
    if epoch % lr_decay == 0:  
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
                print("new lr: ", param_group['lr'])
                
                
                
                
print("Training complete.")
torch.save(net.state_dict(), f"control_net_{problem}.pth")
print(f"Model saved to control_net_{problem}.pth")   

fig, axes = plt.subplots(1, 2, figsize=(10, 3))

axes[0].semilogy(loss_history, label="total")
axes[0].semilogy(run_history,  label="running",  linestyle="--")
axes[0].semilogy(term_history, label="terminal", linestyle=":")
axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()

axes[1].plot(proj_history)
axes[1].set_title("Projection triggered"); axes[1].set_xlabel("epoch"); axes[1].set_ylabel("0/1")

plt.tight_layout()
plt.savefig(f"training_curve_{problem}.png", bbox_inches="tight", dpi=400)

