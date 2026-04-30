"""
Unified training loop for CBF-based control.

Works for any combination of dynamics and obstacles via TrainingConfig hooks.
"""

import json
import os
import time
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm

try:
    import psutil as _psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

from dynamics import SingleIntegrator, DoubleIntegrator
from barriers import CircularObstacle
from controllers import CBFQPController, PolicyNetwork


class CBFTrainer:
    """
    Trainer for CBF-based safe control policies.

    Supports simple systems (integrators) via type strings in TrainingConfig,
    and complex systems (quadrotors, multi-agent) via pre-built objects and
    callable hooks (output_transform, cost_fn, sample_fn, plot_fn).
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = config.dtype

        # --- Dynamics ---
        if config.dynamics is not None:
            self.dynamics = config.dynamics
        elif config.dynamics_type == 'single_integrator':
            self.dynamics = SingleIntegrator(dim=config.position_dim)
        elif config.dynamics_type == 'double_integrator':
            self.dynamics = DoubleIntegrator(dim=config.position_dim)
        else:
            raise ValueError(f"Unknown dynamics_type: {config.dynamics_type}")

        # --- Obstacles ---
        if config.obstacles_list is not None:
            self.obstacles = config.obstacles_list
        else:
            self.obstacles = [
                CircularObstacle(
                    center=obs['center'],
                    radius=obs['radius'],
                    epsilon=obs.get('epsilon', 0.1),
                    dynamics=self.dynamics
                )
                for obs in config.obstacles
            ]

        # --- CBF-QP Controller ---
        self.cbf_controller = CBFQPController(
            dynamics=self.dynamics,
            obstacles=self.obstacles,
            alpha=config.cbf_alpha,
            verbose=False,
            use_slack=config.use_slack,
            slack_penalty=config.slack_penalty,
        )

        # --- Policy Network ---
        self.policy = PolicyNetwork(
            state_dim=config.state_dim,
            control_dim=config.control_dim,
            hidden_dim=config.hidden_dim,
            num_hidden_layers=config.num_hidden_layers,
            activation=config.activation,
            use_time=config.use_time,
        ).to(self.device).to(self.dtype)

        # --- Optimizer ---
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Training state
        self.best_loss = float('inf')
        self.epoch = 0
        self._alpha_terminal = config.terminal_cost_weight

        # History (accessible after training)
        self.history = {
            'loss': [], 'running': [], 'terminal': [],
            'grad_norm': [], 'min_barrier': [], 'alpha_terminal': [],
            'epoch_time': [], 'elapsed_time': [], 'cpu_mem_mb': [], 'gpu_mem_mb': [],
        }

        self._proc = _psutil.Process(os.getpid()) if _PSUTIL_AVAILABLE else None
        self._train_start_time = None
        self._log_file = None

    def sample_initial_states(self, batch_size):
        """
        Sample initial states. Uses config.sample_fn if provided,
        otherwise adds Gaussian noise to config.initial_state_tensor.
        """
        if self.config.sample_fn is not None:
            return self.config.sample_fn(batch_size)

        noise = torch.randn(
            batch_size, self.config.state_dim,
            dtype=self.dtype, device=self.device
        ) * self.config.initial_state_std

        return self.config.initial_state_tensor.unsqueeze(0) + noise

    def rollout(self, z0):
        """
        Simulate trajectory with CBF safety filtering.

        Args:
            z0: (batch_size, state_dim)

        Returns:
            states:   (batch_size, num_steps+1, state_dim)
            controls: (batch_size, num_steps,   control_dim)
        """
        states = [z0]
        controls = []
        z = z0
        t = 0.0

        for _ in range(self.config.num_steps):
            u_raw = self.policy(z, t / self.config.T)

            if self.config.output_transform is not None:
                u_desired = self.config.output_transform(u_raw)
            else:
                u_desired = u_raw

            try:
                u_safe = self.cbf_controller.filter_control(z, u_desired)
            except Exception as e:
                print(f"\n  Warning: CBF-QP failed: {e}. Using desired control.")
                u_safe = u_desired

            z = self.dynamics.step(z, u_safe, self.config.dt, t)
            t += self.config.dt
            states.append(z)
            controls.append(u_safe)

        states = torch.stack(states, dim=1)    # (batch, T+1, state_dim)
        controls = torch.stack(controls, dim=1)  # (batch, T,   control_dim)
        return states, controls

    def compute_loss(self, states, controls):
        """
        Compute training loss.

        Uses config.cost_fn if provided (signature: fn(states, controls, alpha_terminal)
        -> (total, running, terminal)), otherwise uses the default quadratic cost.

        Returns:
            total_loss, running_cost, terminal_cost
        """
        if self.config.cost_fn is not None:
            return self.config.cost_fn(states, controls, self._alpha_terminal)

        # Default: control effort + terminal distance to target
        running_cost = torch.mean(
            torch.sum(controls ** 2, dim=-1).sum(dim=-1)
        ) * self.config.dt

        # Optional velocity penalty (for dynamics that expose a velocity() method,
        # e.g. DoubleIntegrator). Controlled by config.velocity_penalty.
        if self.config.velocity_penalty and hasattr(self.dynamics, 'velocity'):
            mid = states[:, 1:, :]                                 # (batch, T, state_dim)
            batch, T, sdim = mid.shape
            vel = self.dynamics.velocity(mid.reshape(batch * T, sdim))  # (batch*T, vel_dim)
            vel = vel.reshape(batch, T, -1)
            running_cost = running_cost + torch.mean(
                vel.pow(2).sum(dim=-1).sum(dim=-1)
            ) * self.config.dt

        final_state = states[:, -1, :]
        target = self.config.target_state_tensor.unsqueeze(0)
        terminal_cost = torch.mean(torch.sum((final_state - target) ** 2, dim=-1))

        total_loss = (
            self.config.control_penalty * running_cost
            + self._alpha_terminal * terminal_cost
        )
        return total_loss, running_cost, terminal_cost

    def train_epoch(self):
        """
        One training epoch.

        Returns:
            loss, running_cost, terminal_cost, grad_norm, last_states
            where last_states is (batch, state_dim) — final step, detached.
        """
        self.policy.train()
        self.optimizer.zero_grad()

        z0 = self.sample_initial_states(self.config.batch_size)
        states, controls = self.rollout(z0)
        loss, running, terminal = self.compute_loss(states, controls)

        loss.backward()

        # Gradient norm (before clipping)
        grad_norm = sum(
            p.grad.data.norm(2).item() ** 2
            for p in self.policy.parameters()
            if p.grad is not None
        ) ** 0.5

        if self.config.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.grad_clip_norm)

        self.optimizer.step()

        return loss.item(), running.item(), terminal.item(), grad_norm, states[:, -1, :].detach()

    def _tlog(self, msg):
        """Write to tqdm output and the log file."""
        tqdm.write(msg)
        if self._log_file is not None:
            self._log_file.write(msg + '\n')
            self._log_file.flush()

    def _write_history(self, csv_path):
        """Rewrite the full training history CSV (called each epoch)."""
        n = len(self.history['loss'])
        pd.DataFrame({
            'epoch':          range(1, n + 1),
            'total_loss':     self.history['loss'],
            'running_cost':   self.history['running'],
            'terminal_cost':  self.history['terminal'],
            'alpha_terminal': self.history['alpha_terminal'],
            'min_barrier':    self.history['min_barrier'],
            'gradient_norm':  self.history['grad_norm'],
            'epoch_time_s':   self.history['epoch_time'],
            'elapsed_time_s': self.history['elapsed_time'],
            'cpu_mem_mb':     self.history['cpu_mem_mb'],
            'gpu_mem_mb':     self.history['gpu_mem_mb'],
        }).to_csv(csv_path, index=False)

    def _save_config(self):
        """Save training config as JSON in the same directory as the checkpoint."""
        save_dir = os.path.dirname(self.config.save_path) or '.'
        os.makedirs(save_dir, exist_ok=True)

        cfg = {
            'dynamics_type': self.config.dynamics_type,
            'state_dim': self.config.state_dim,
            'control_dim': self.config.control_dim,
            'position_dim': self.config.position_dim,
            'T': self.config.T,
            'dt': self.config.dt,
            'num_steps': self.config.num_steps,
            'cbf_alpha': self.config.cbf_alpha if not isinstance(self.config.cbf_alpha, tuple)
                         else list(self.config.cbf_alpha),
            'control_penalty': self.config.control_penalty,
            'terminal_cost_weight': self.config.terminal_cost_weight,
            'alpha_terminal_final': self.config.alpha_terminal_final,
            'alpha_terminal_step': self.config.alpha_terminal_step,
            'alpha_terminal_every': self.config.alpha_terminal_every,
            'velocity_penalty': self.config.velocity_penalty,
            'hidden_dim': self.config.hidden_dim,
            'num_hidden_layers': self.config.num_hidden_layers,
            'activation': self.config.activation,
            'use_time': self.config.use_time,
            'num_epochs': self.config.num_epochs,
            'batch_size': self.config.batch_size,
            'learning_rate': self.config.learning_rate,
            'weight_decay': self.config.weight_decay,
            'initial_state_std': self.config.initial_state_std,
            'grad_clip_norm': self.config.grad_clip_norm,
            'lr_decay_every': self.config.lr_decay_every,
            'lr_decay_factor': self.config.lr_decay_factor,
            'initial_state': self.config.initial_state,
            'target_state': self.config.target_state,
            'obstacles': [
                {'center': obs.center.tolist(), 'radius': obs.radius,
                 'epsilon': obs.epsilon,
                 **({'n_agent': obs.n_agent} if hasattr(obs, 'n_agent') else {})}
                for obs in self.obstacles
            ] if self.config.obstacles_list is not None else self.config.obstacles,
            'num_obstacles': self.config.get_num_obstacles(),
            'device': self.config.device,
            'use_double_precision': self.config.use_double_precision,
            'save_path': self.config.save_path,
        }

        config_path = os.path.join(save_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(cfg, f, indent=2)

        print(f"Config saved: {config_path}")

    def train(self, verbose=True):
        """
        Full training loop.

        Returns:
            policy: Trained policy network.
        """
        save_dir = os.path.dirname(self.config.save_path) or '.'
        os.makedirs(save_dir, exist_ok=True)

        csv_path = self.config.csv_path or os.path.join(save_dir, 'training_history.csv')
        log_path = self.config.log_path or os.path.join(save_dir, 'training_log.txt')

        self._log_file = open(log_path, 'w', buffering=1)  # line-buffered

        if verbose:
            header = f"\n{self.config}\n\nStarting training..."
            print(header)
            self._log_file.write(header + '\n')

        self._save_config()
        self._alpha_terminal = self.config.terminal_cost_weight
        self._train_start_time = time.perf_counter()

        pbar = tqdm(range(1, self.config.num_epochs + 1), disable=not verbose)

        for epoch in pbar:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            self.epoch = epoch
            t0 = time.perf_counter()
            loss, running, terminal, grad_norm, last_states = self.train_epoch()
            epoch_time = time.perf_counter() - t0
            elapsed_time = time.perf_counter() - self._train_start_time

            # Alpha-terminal scheduling
            if (
                self.config.alpha_terminal_final is not None
                and self.config.alpha_terminal_every > 0
                and epoch % self.config.alpha_terminal_every == 0
                and self._alpha_terminal < self.config.alpha_terminal_final
            ):
                self._alpha_terminal += self.config.alpha_terminal_step
                if verbose:
                    self._tlog(f"  → alpha_terminal: {self._alpha_terminal:.1f}")

            # Learning rate decay
            if self.config.lr_decay_every > 0 and epoch % self.config.lr_decay_every == 0:
                for pg in self.optimizer.param_groups:
                    pg['lr'] *= self.config.lr_decay_factor
                if verbose:
                    self._tlog(f"  → lr: {self.optimizer.param_groups[0]['lr']:.2e}")

            # Barrier monitoring
            min_barrier = float('nan')
            if self.config.log_barrier and self.obstacles:
                min_barrier = min(
                    obs.h(last_states, self.dynamics).min().item() for obs in self.obstacles
                )

            # Memory usage
            cpu_mem_mb = (
                self._proc.memory_info().rss / 1024 ** 2 if self._proc is not None else float('nan')
            )
            if self.device.type == 'cuda':
                gpu_mem_mb = torch.cuda.max_memory_allocated(self.device) / 1024 / 1024
            else:
                gpu_mem_mb = 0.0

            # Record history
            self.history['loss'].append(loss)
            self.history['running'].append(running)
            self.history['terminal'].append(terminal)
            self.history['grad_norm'].append(grad_norm)
            self.history['min_barrier'].append(min_barrier)
            self.history['alpha_terminal'].append(self._alpha_terminal)
            self.history['epoch_time'].append(epoch_time)
            self.history['elapsed_time'].append(elapsed_time)
            self.history['cpu_mem_mb'].append(cpu_mem_mb)
            self.history['gpu_mem_mb'].append(gpu_mem_mb)

            # Save best model
            if loss < self.best_loss:
                self.best_loss = loss
                self.save_checkpoint()

            # Per-epoch CSV update
            self._write_history(csv_path)

            # Logging
            if verbose and epoch % self.config.log_every == 0:
                msg = (
                    f"epoch {epoch:4d}  loss={loss:.4e}  "
                    f"run={running:.4e}  term={terminal:.4e}  "
                    f"alpha_t={self._alpha_terminal:.1f}"
                )
                if self.config.log_grad_norm:
                    msg += f"  grad={grad_norm:.2e}"
                if self.config.log_barrier and not torch.isnan(torch.tensor(min_barrier)):
                    msg += f"  h_min={min_barrier:.2e}"
                msg += f"  lr={self.optimizer.param_groups[0]['lr']:.2e}"
                self._tlog(msg)

            # Plotting
            if (
                self.config.plot_fn is not None
                and self.config.plot_every > 0
                and epoch % self.config.plot_every == 0
            ):
                with torch.no_grad():
                    z0_plot = self.sample_initial_states(1)
                    plot_states, _ = self.rollout(z0_plot)
                self.config.plot_fn(self, epoch, plot_states)

        done_msg = f"\nTraining complete! Best loss: {self.best_loss:.4e}"
        done_msg += f"\nCSV:  {csv_path}\nLog:  {log_path}"
        if verbose:
            print(done_msg)
        if self._log_file is not None:
            self._log_file.write(done_msg + '\n')
            self._log_file.close()
            self._log_file = None

        return self.policy

    def save_checkpoint(self):
        """Save best model checkpoint."""
        metadata = {
            'epoch': self.epoch,
            'loss': self.best_loss,
            'config': {
                'dynamics_type': self.config.dynamics_type,
                'num_obstacles': self.config.get_num_obstacles(),
                'cbf_alpha': self.config.cbf_alpha,
                'dt': self.config.dt,
                'num_steps': self.config.num_steps,
            }
        }
        self.policy.save(self.config.save_path, metadata)

    def load_checkpoint(self, filepath=None):
        """Load model checkpoint."""
        if filepath is None:
            filepath = self.config.save_path
        self.policy, metadata = PolicyNetwork.load(
            filepath, device=self.device, dtype=self.dtype
        )
        return metadata
