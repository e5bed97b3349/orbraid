"""
orbraid — Three-dimensional choreographic N-body orbit search.

This package provides the core computational infrastructure:
  - Fourier synthesis and analysis of periodic curves
  - Action functional evaluation and gradient computation
  - Yoshida 6th-order symplectic integration
  - Reference orbit data (Chenciner-Montgomery figure-eight)
"""

import jax

# Enable float64 on GPU — required for the precision targets in this project.
jax.config.update("jax_enable_x64", True)

from orbraid.fourier import synthesize_curve, synthesize_velocity, extract_fourier_coeffs
from orbraid.action import build_action
from orbraid.integrator import yoshida_step, integrate_yoshida

__all__ = [
    "synthesize_curve",
    "synthesize_velocity",
    "extract_fourier_coeffs",
    "build_action",
    "yoshida_step",
    "integrate_yoshida",
]
