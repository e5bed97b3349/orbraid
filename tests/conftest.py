"""
Shared test fixtures for orbraid test suite.

Provides the Chenciner-Montgomery figure-eight orbit as Fourier
coefficients at K=64, M=512 — the standard test case for validating
the action functional and integrator.
"""

import pytest
import jax
import jax.numpy as jnp

# Ensure float64 is enabled before any test runs
jax.config.update("jax_enable_x64", True)

from orbraid.figure_eight import (
    compute_figure_eight_fourier,
    chenciner_montgomery_ics,
    integrate_figure_eight,
    PERIOD,
    MASS,
    G_CONST,
    compute_figure_eight_energy,
)


@pytest.fixture(scope="session")
def figure_eight_K64_M512():
    """
    Fourier coefficients for body 1 of the figure-eight, K=64, M=512.

    Returns dict with:
        coeff: jnp.ndarray (64, 2, 3) — Fourier coefficients
        theta: jnp.ndarray (384,) — flattened for action(theta)
        K: 64
        M: 512
        N: 3
    """
    K, M = 64, 512
    coeff = compute_figure_eight_fourier(K=K, M=M)
    theta = coeff.ravel()
    return {
        "coeff": coeff,
        "theta": theta,
        "K": K,
        "M": M,
        "N": 3,
    }


@pytest.fixture(scope="session")
def figure_eight_ics():
    """Initial conditions for the figure-eight orbit."""
    q0, v0 = chenciner_montgomery_ics()
    return jnp.array(q0), jnp.array(v0)


@pytest.fixture(scope="session")
def figure_eight_trajectory():
    """
    High-precision reference trajectory (1 period, 1024 points).
    """
    t_eval, positions, velocities = integrate_figure_eight(M=1024, n_periods=1)
    return {
        "t_eval": t_eval,
        "positions": jnp.array(positions),
        "velocities": jnp.array(velocities),
    }


@pytest.fixture(scope="session")
def figure_eight_energy():
    """Total energy of the figure-eight at t=0."""
    return compute_figure_eight_energy()
