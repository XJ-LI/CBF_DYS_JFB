"""
Base class for barrier functions in CBF framework.

Barrier functions define safe sets and enable safety constraint formulation.

Standard CBF (relative degree 1):
    h(x) ≥ 0 defines safe set
    ḣ + α(h) ≥ 0 guarantees safety

Higher-Order CBF (relative degree r):
    h₀(x) ≥ 0 defines safe set
    h₁ = ḣ₀ + α₁(h₀) ≥ 0
    h₂ = ḣ₁ + α₂(h₁) ≥ 0
    ...
    h_r^(r) + α_r(h_{r-1}) ≥ 0  (control appears here)
"""

import torch
from abc import ABC, abstractmethod


class BarrierFunction(ABC):
    """
    Abstract base class for barrier functions.

    A barrier function h(x) defines:
        - Safe set: S = {x : h(x) ≥ 0}
        - Unsafe set: U = {x : h(x) < 0}

    For use with CBF, must compute Lie derivatives compatible with
    the relative degree of the dynamics.
    """

    def __init__(self, relative_degree=1):
        """
        Initialize barrier function.

        Args:
            relative_degree: Number of derivatives before control appears
                           1 = standard CBF
                           2 = HOCBF (for double integrator)
        """
        self.relative_degree = relative_degree

    @abstractmethod
    def h(self, x, dynamics):
        """
        Evaluate barrier function.

        Args:
            x: State (batch_size, state_dim)
            dynamics: Dynamics object (used to extract position if needed)

        Returns:
            h: Barrier value (batch_size,)
               h > 0: safe
               h < 0: unsafe
        """
        pass

    @abstractmethod
    def compute_cbf_constraint(self, x, dynamics, alpha):
        """
        Compute CBF constraint parameters for QP:
            A_cbf @ u ≥ b_cbf

        The form depends on relative degree:
            - Relative degree 1: A = Lg h, b = -Lf h - α(h)
            - Relative degree 2: A = Lg Lf h, b = -Lf² h - α₂(h₁)

        Args:
            x: State (batch_size, state_dim)
            dynamics: Dynamics object (provides f, g)
            alpha: CBF class-K parameter(s)
                  Can be scalar or tuple (alpha1, alpha2) for HOCBF

        Returns:
            A_cbf: Constraint gradient (batch_size, control_dim)
            b_cbf: Constraint offset (batch_size,)
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(relative_degree={self.relative_degree})"


class RelativeDegree1Barrier(BarrierFunction):
    """
    Base class for standard CBF (relative degree 1).

    Constraint: Lf h + Lg h · u ≥ -α(h)
    QP form: (Lg h) @ u ≥ -Lf h - α(h)
    """

    def __init__(self):
        super().__init__(relative_degree=1)

    def compute_lie_derivatives(self, x, dynamics):
        """
        Compute Lie derivatives for relative degree 1.

        Args:
            x: State (batch_size, state_dim)
            dynamics: Dynamics object

        Returns:
            Lf_h: Drift Lie derivative (batch_size,)
            Lg_h: Control Lie derivative (batch_size, control_dim)
            h_val: Barrier value (batch_size,)
        """
        # This will be implemented by subclasses
        # Can use autograd or manual differentiation
        raise NotImplementedError

    def compute_cbf_constraint(self, x, dynamics, alpha):
        """
        Standard CBF constraint: Lg h · u ≥ -Lf h - α(h)

        Args:
            x: State (batch_size, state_dim)
            dynamics: Dynamics object
            alpha: Class-K parameter (scalar)

        Returns:
            A_cbf: Lg h (batch_size, control_dim)
            b_cbf: -Lf h - α(h) (batch_size,)
        """
        Lf_h, Lg_h, h_val = self.compute_lie_derivatives(x, dynamics)

        A_cbf = Lg_h
        b_cbf = -Lf_h - alpha * h_val

        return A_cbf, b_cbf


class RelativeDegree2Barrier(BarrierFunction):
    """
    Base class for HOCBF (relative degree 2).

    For double integrator with position barrier h₀(p):

    Auxiliary barrier:
        h₁ = ḣ₀ + α₁(h₀)

    HOCBF constraint (from ψ̇₁ + α₂ψ₁ ≥ 0):
        Lf²h + LgLf h·u + α₁·Lf h + α₂·ψ₁ ≥ 0

    QP form:
        (Lg Lf h₀) @ u ≥ -Lf² h₀ - α₁·Lf h₀ - α₂(h₁)

    Computation:
        ḣ₀ = ∇h₀ · ẋ = ∇h₀ · (f + gu) = ∇h₀ · f + ∇h₀ · g · u
        ḧ₀ = d/dt(∇h₀ · f) + d/dt(∇h₀ · g) · u
           = Lf(Lf h₀) + Lg(Lf h₀) · u  (assuming g constant for double int)

    For circular barrier on double integrator:
        h₀(p) = ||p - c||² - r²
        ∇h₀ = 2(p - c)
        ḣ₀ = 2(p - c) · v
        ḧ₀ = 2||v||² + 2(p - c) · a
    """

    def __init__(self):
        super().__init__(relative_degree=2)

    def compute_hocbf_terms(self, x, dynamics):
        """
        Compute HOCBF terms for relative degree 2.

        Args:
            x: State (batch_size, state_dim)
            dynamics: Dynamics object (must be DoubleIntegrator)

        Returns:
            h0: Original barrier h₀(p) (batch_size,)
            h0_dot: First derivative ḣ₀ (batch_size,)
            Lf2_h0: Second Lie derivative Lf²h₀ (batch_size,)
            Lg_Lf_h0: Control Lie derivative Lg Lf h₀ (batch_size, control_dim)
        """
        # This will be implemented by subclasses
        raise NotImplementedError

    def compute_cbf_constraint(self, x, dynamics, alpha):
        """
        HOCBF constraint: Lg Lf h₀ · u ≥ -Lf² h₀ - α₂(h₁)

        Args:
            x: State (batch_size, state_dim)
            dynamics: Dynamics object
            alpha: Tuple (alpha1, alpha2) for HOCBF

        Returns:
            A_cbf: Lg Lf h₀ (batch_size, control_dim)
            b_cbf: -Lf² h₀ - α₂(h₁) (batch_size,)
        """
        if isinstance(alpha, (int, float)):
            # If single value given, use same for both
            alpha1 = alpha2 = alpha
        else:
            alpha1, alpha2 = alpha

        h0, h0_dot, Lf2_h0, Lg_Lf_h0 = self.compute_hocbf_terms(x, dynamics)

        # Auxiliary barrier: h₁ = ḣ₀ + α₁ h₀
        h1 = h0_dot + alpha1 * h0

        # QP constraint
        A_cbf = Lg_Lf_h0
        b_cbf = -Lf2_h0 - alpha1 * h0_dot - alpha2 * h1

        return A_cbf, b_cbf
