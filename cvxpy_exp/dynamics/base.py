"""
Base class for control-affine dynamical systems.

All dynamics must implement the control-affine form:
    ẋ = f(x) + g(x)u

Key concept: RELATIVE DEGREE
    - The relative degree determines how many derivatives are needed before
      control appears in the barrier function derivative
    - Single integrator (ẋ = u): relative degree = 1
    - Double integrator (ẍ = a): relative degree = 2
"""

import torch
from abc import ABC, abstractmethod


class ControlAffineDynamics(ABC):
    """
    Abstract base class for control-affine systems: ẋ = f(x) + g(x)u

    Subclasses must implement:
        - state_dim: dimension of state space
        - control_dim: dimension of control space
        - relative_degree: for CBF constraint formulation
        - f(x): drift dynamics
        - g(x): control matrix
        - step(x, u, dt): discrete-time integration
    """

    def __init__(self, state_dim, control_dim, relative_degree):
        """
        Initialize dynamics.

        Args:
            state_dim: Dimension of state space
            control_dim: Dimension of control input
            relative_degree: Relative degree for CBF (1 for standard, 2 for HOCBF)
        """
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.relative_degree = relative_degree

    @abstractmethod
    def f(self, x, t=0.0):
        """
        Drift dynamics f(x, t).

        Args:
            x: State tensor (batch_size, state_dim)
            t: Current time (default 0.0). These dynamics are time-invariant;
               t is accepted to match the reference interface f(z, u, t).

        Returns:
            f: Drift vector (batch_size, state_dim)
        """
        pass

    @abstractmethod
    def g(self, x, t=0.0):
        """
        Control matrix g(x, t).

        Args:
            x: State tensor (batch_size, state_dim)
            t: Current time (default 0.0). Accepted to match reference interface.

        Returns:
            g: Control matrix (batch_size, state_dim, control_dim)
        """
        pass

    @abstractmethod
    def step(self, x, u, dt, t=0.0):
        """
        Simulate one time step: x_{t+1} = integrate(f(x,t) + g(x,t)u, dt)

        Args:
            x: Current state (batch_size, state_dim)
            u: Control input (batch_size, control_dim)
            dt: Time step
            t: Current time (default 0.0). Passed through to f and g at each
               RK4 sub-step, matching reference: rk4_step(z, u, ti, h, f).

        Returns:
            x_next: Next state (batch_size, state_dim)
        """
        pass

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"state_dim={self.state_dim}, "
                f"control_dim={self.control_dim}, "
                f"relative_degree={self.relative_degree})")
