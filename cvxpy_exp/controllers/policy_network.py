"""
Neural network policy for control.

Simple MLP that maps state to control action.
"""

import torch
import torch.nn as nn
import os


class _ResBlock(nn.Module):
    """Residual block with SiLU activations — matches reference ControlNet/ResBlock."""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class PolicyNetwork(nn.Module):
    """
    Neural network policy for control.

    Two architectures selectable via ``activation``:

    * ``'resnet'`` — input_proj → ResBlock × n → output_proj (matches reference
      ControlNet with residual connections + SiLU).  Best for hard problems like
      quadrotor.
    * any other string (``'relu'``, ``'silu'``, ``'tanh'``) — plain MLP.
    """

    def __init__(self, state_dim, control_dim, hidden_dim=64, num_hidden_layers=3, activation='relu', use_time=False):
        """
        Args:
            state_dim: Input dimension (state space)
            control_dim: Output dimension (control space)
            hidden_dim: Hidden layer size
            num_hidden_layers: Number of hidden layers / residual blocks
            activation: 'resnet' | 'relu' | 'silu' | 'tanh'
            use_time: If True, append normalized time t/T as extra input
        """
        super().__init__()

        self.state_dim = state_dim
        self.control_dim = control_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.activation = activation
        self.use_time = use_time

        input_dim = state_dim + 1 if use_time else state_dim

        if activation.lower() == 'resnet':
            # ResNet architecture matching reference ControlNet
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.blocks = nn.Sequential(
                *[_ResBlock(hidden_dim) for _ in range(num_hidden_layers)]
            )
            output_layer = nn.Linear(hidden_dim, control_dim)
            self.output_proj = output_layer
            self.network = None  # use resnet path
        else:
            # Plain MLP
            if activation.lower() == 'relu':
                act_fn = nn.ReLU
            elif activation.lower() == 'silu':
                act_fn = nn.SiLU
            elif activation.lower() == 'tanh':
                act_fn = nn.Tanh
            else:
                raise ValueError(f"Unknown activation: {activation}")

            layers = [nn.Linear(input_dim, hidden_dim), act_fn()]
            for _ in range(num_hidden_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(act_fn())
            output_layer = nn.Linear(hidden_dim, control_dim)
            layers.append(output_layer)
            self.network = nn.Sequential(*layers)

        # Small output init — policy starts near-zero so u ≈ hover_bias
        nn.init.uniform_(output_layer.weight, -0.01, 0.01)
        nn.init.zeros_(output_layer.bias)

    def forward(self, state, t=None):
        """
        Forward pass.

        Args:
            state: State tensor (batch_size, state_dim)
            t: Normalized time scalar in [0, 1] (ignored if use_time=False)

        Returns:
            control: Control tensor (batch_size, control_dim)
        """
        if self.use_time:
            t_val = t if t is not None else 0.0
            t_tensor = torch.full(
                (state.shape[0], 1), t_val,
                dtype=state.dtype, device=state.device
            )
            inp = torch.cat([state, t_tensor], dim=-1)
        else:
            inp = state

        if self.network is not None:
            return self.network(inp)
        # ResNet path
        h = self.input_proj(inp)
        h = self.blocks(h)
        return self.output_proj(h)

    def save(self, filepath, metadata=None):
        """
        Save model checkpoint with metadata.

        Args:
            filepath: Path to save checkpoint
            metadata: Optional dictionary with training info
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'architecture': {
                'state_dim': self.state_dim,
                'control_dim': self.control_dim,
                'hidden_dim': self.hidden_dim,
                'num_hidden_layers': self.num_hidden_layers,
                'activation': self.activation,
                'use_time': self.use_time,
            }
        }

        if metadata is not None:
            checkpoint['metadata'] = metadata

        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        torch.save(checkpoint, filepath)
        print(f"Model saved: {filepath}")

    @staticmethod
    def load(filepath, device=None, dtype=torch.float32):
        """
        Load model from checkpoint.

        Args:
            filepath: Path to checkpoint
            device: Target device (auto-detected if None)
            dtype: Target dtype

        Returns:
            model: Loaded policy network
            metadata: Training metadata (if available)
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load(filepath, map_location=device)
        arch = checkpoint['architecture']

        model = PolicyNetwork(
            state_dim=arch['state_dim'],
            control_dim=arch['control_dim'],
            hidden_dim=arch['hidden_dim'],
            num_hidden_layers=arch.get('num_hidden_layers', 3),
            activation=arch.get('activation', 'relu'),
            use_time=arch.get('use_time', False),
        ).to(device).to(dtype)

        model.load_state_dict(checkpoint['model_state_dict'])
        metadata = checkpoint.get('metadata', None)

        print(f"Model loaded: {filepath}")
        if metadata:
            print(f"  Epoch {metadata.get('epoch', 'N/A')}, Loss {metadata.get('loss', 'N/A'):.4f}")

        return model, metadata

    def __repr__(self):
        return (f"PolicyNetwork(state_dim={self.state_dim}, "
                f"control_dim={self.control_dim}, "
                f"hidden_dim={self.hidden_dim}, "
                f"layers={self.num_hidden_layers})")
