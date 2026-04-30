# CVXPyLayers CBF Training

Reference implementation to compare against: `../quadcopter_multi.py` + `../train.py` (DYS/JFB).  
Bug-fix log and derivations: `CHANGES.md`.  
Full architecture and math: `ARCHITECTURE.md`.  
Slack variable deep-dive: `SLACK_VARIABLE.md`.

---

## Quick Start

```bash
# Multi-agent double integrator (matches reference double_integrator_multi)
python examples/train_double_integrator.py

# Multi-quadrotor (matches reference quadcopter_multi)
python examples/train_quadrotor_multi_cvxpy.py
```

Both scripts accept CLI flags — see [Training Settings](#training-settings) below.

---

## What CVXPyLayers Is Doing

At every rollout step, the policy outputs a **desired control** `u_desired`. A
safety filter solves a small QP to find the closest safe control:

**Soft (default):**
```
minimize   ||u − u_desired||² + p·||s||²
s.t.       A_i · u + s_i ≥ b_i      (one row per agent per obstacle)
           s_i ≥ 0
```

**Hard (`use_slack=False`):**
```
minimize   ||u − u_desired||²
s.t.       A_i · u ≥ b_i
```

Slack variables (`p = 1e4` default) make the QP always feasible — see `SLACK_VARIABLE.md`.

**CVXPyLayers turns this QP into a differentiable layer.** In the forward pass it
calls SCS (via `diffcp`) to get `u_safe`. In the backward pass it uses the implicit
function theorem on the KKT conditions to compute `∂u_safe / ∂u_desired`, so
gradients flow from the trajectory loss back through the safety filter into the
policy network.

```
policy(x)  →  u_desired
                  ↓
   A_cbf(x), b_cbf(x)   ← CBF constraint matrices
                  ↓
     CVXPyLayers QP (SCS solver, soft constraints)
                  ↓
              u_safe
                  ↓
       dynamics.step(x, u_safe, dt)   ← RK4 / Euler integration
                  ↓
              x_next
```

---

## File Map

| File | Purpose |
|------|---------|
| `examples/train_double_integrator.py` | Multi-agent double integrator training |
| `examples/train_quadrotor_multi_cvxpy.py` | Multi-quadrotor training |
| `controllers/cbf_qp_layer.py` | `create_cbf_qp_layer()` + `CBFQPController` |
| `controllers/policy_network.py` | `PolicyNetwork` (MLP or ResNet) |
| `barriers/circular_obstacle.py` | 2D circular CBF (single + multi-agent) |
| `barriers/spherical_obstacle.py` | 3D spherical CBF for quadrotor |
| `dynamics/double_integrator.py` | Double integrator dynamics |
| `dynamics/quadrotor.py` | 6-DOF quadrotor dynamics |
| `training/config.py` | `TrainingConfig` dataclass |
| `training/trainer.py` | `CBFTrainer` training loop |

---

## Training Settings

Settings are aligned with the reference (`../readme.md`).

### Double Integrator Multi-Agent

```bash
python examples/train_double_integrator.py \
    --epochs 5000 \
    --lr_decay 800 \
    --hidden_dim 128
```

| Flag | Default | Reference |
|------|---------|-----------|
| `--epochs` | 5000 | 5000 |
| `--lr` | 1e-3 | 1e-3 |
| `--lr_decay` | 800 | 800 |
| `--hidden_dim` | 128 | 128 |
| `--n_blocks` | 3 | 3 |
| `--n_agent` | 6 | 6 |

### Quadrotor Multi

```bash
python examples/train_quadrotor_multi_cvxpy.py \
    --epochs 2800 \
    --lr_decay 600 \
    --hidden_dim 128
```

| Flag | Default | Reference |
|------|---------|-----------|
| `--epochs` | 2800 | 2800 |
| `--lr` | 1e-3 | 1e-3 |
| `--lr_decay` | 600 | 600 |
| `--hidden_dim` | 128 | 128 |
| `--n_blocks` | 3 | 3 |
| `--grad_clip` | off | off |

---

## Key Components

### `CBFQPController` — slack and hard-constraint mode

```python
# Default: soft constraints (always feasible)
controller = CBFQPController(dynamics, obstacles, alpha)

# Hard constraints (use only if QP is always feasible)
controller = CBFQPController(dynamics, obstacles, alpha, use_slack=False)

# Tune slack penalty (default 1e4)
controller = CBFQPController(dynamics, obstacles, alpha, slack_penalty=1e6)
```

### `PolicyNetwork` — save / load

`activation` is now saved in the checkpoint and restored on load. This fixes a
bug where reloading a `resnet` model would reconstruct the wrong architecture,
causing `load_state_dict` to fail.

```python
# Save
policy.save("model.pth", metadata={"epoch": 100, "loss": 0.5})

# Load — restores correct architecture including activation
policy, metadata = PolicyNetwork.load("model.pth", device="cuda")
```

### `TrainingConfig` — velocity penalty

`velocity_penalty` now works in the **default cost function** (not just custom
`cost_fn`). It applies only when the dynamics expose a `velocity()` method
(i.e. `DoubleIntegrator`, not `SingleIntegrator` or `Quadrotor`).

```python
config = TrainingConfig(
    dynamics_type='double_integrator',
    velocity_penalty=True,   # adds ||v||² running cost (default True)
    ...
)
```

For `Quadrotor`, velocity penalty is handled inside the custom `cost_fn` in
`train_quadrotor_multi_cvxpy.py`.

---

## Knobs and What They Do

### Safety / HOCBF

| Parameter | Default | Effect |
|-----------|---------|--------|
| `cbf_alpha` (scalar or tuple) | `(1,1)` | Higher → more conservative, CBF activates earlier, fewer gradient directions available |
| `obstacle_radius` | 0.35 (quadrotor) | Physical radius (m) |
| `epsilon` | 0.15 (quadrotor) | Safety margin; safe boundary at `sqrt(r² + ε²)` |
| `use_slack` | `True` | Soft constraints (always feasible) vs hard constraints |
| `slack_penalty` | `1e4` | Cost on constraint violation; higher recovers hard constraint |

### Cost and Schedule

| Parameter | Default | Effect |
|-----------|---------|--------|
| `terminal_cost_weight` | 20.0 | Initial weight on terminal error |
| `alpha_terminal_final` | varies | Cap for annealing schedule |
| `alpha_terminal_step` | 5.0 | Increment per schedule event |
| `alpha_terminal_every` | 20 | Epochs between increments |
| `control_penalty` | 1.0 | Weight on `||u||²` running cost |
| `velocity_penalty` | `True` | Add `||v||²` running cost (double integrator only) |

### Network Architecture

| Parameter | Default | Effect |
|-----------|---------|--------|
| `hidden_dim` | 128 | Hidden layer width |
| `num_hidden_layers` | 3 | Depth |
| `activation` | `'tanh'` (DI) / `'resnet'` (quadrotor) | Architecture |
| `use_time` | `True` | Append `t/T` to policy input |

### Optimiser

| Parameter | Default | Effect |
|-----------|---------|--------|
| `learning_rate` | 1e-3 | Adam LR |
| `lr_decay_every` | 800 (DI) / 600 (quad) | Halve LR every N epochs |
| `lr_decay_factor` | 0.5 | Multiplier at each decay |
| `weight_decay` | 1e-4 | L2 regularisation |
| `grad_clip_norm` | 0 (DI) / 100 (quad) | Max gradient norm; 0 = off |

### SCS Solver

| Parameter | Value | Effect |
|-----------|-------|--------|
| `eps` | 1e-3 | Primal/dual tolerance. Tighter → "Solved/Inaccurate" with near-zero `A_cbf` |
| `max_iters` | 1000 | Sufficient for typical QP sizes at eps=1e-3 |
| `acceleration_lookback` | 0 | Anderson acceleration disabled (more stable) |

---

## Diagnostic Guide

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `Solved/Inaccurate` | Near-zero `A_cbf` coefficient (quadrotor thrust ⊥ obstacle) | Slack absorbs — expected; check `verbose=True` warnings |
| Loss flat / not decreasing | `grad_clip_norm` too small or `cbf_alpha` too large (zeroing gradients) | Try `--grad_clip` or reduce `cbf_alpha` |
| Loss increasing as `alpha_terminal` rises | Corrupted gradients from SCS | Check `eps`, reduce `alpha_terminal_step` |
| `h_min < 0` despite CBF | HOCBF formula error | Check `b_cbf` in barrier — see `CHANGES.md` |
| Training diverges | Unclamped policy → infeasible QP | Add `tanh` output scaling via `output_transform` |
| Load fails with key mismatch | Old checkpoint saved before `activation` was stored | Retrain; `activation` is now saved correctly |
