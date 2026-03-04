"""
Tests for the action functional (issue #6).

Validates build_action against the Chenciner-Montgomery figure-eight orbit:
1. Fourier round-trip: synthesize -> extract -> compare
2. Action gradient at figure-eight: ||grad(A)|| < 1e-4
3. Action value cross-check: Fourier-space vs numerical quadrature
4. JIT and differentiability: jax.jit and jax.grad work without error
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp

from orbraid.fourier import synthesize_curve, synthesize_velocity, extract_fourier_coeffs
from orbraid.action import build_action
from orbraid.figure_eight import (
    chenciner_montgomery_ics,
    integrate_figure_eight,
    PERIOD,
    MASS,
    G_CONST,
)


class TestFourierRoundtrip:
    """Test that Fourier synthesis/analysis are consistent inverses."""

    def test_roundtrip_figure_eight(self, figure_eight_K64_M512):
        """synthesize(coeff, M) -> extract(trajectory, K) recovers coeff."""
        coeff = figure_eight_K64_M512["coeff"]
        K = figure_eight_K64_M512["K"]
        M = figure_eight_K64_M512["M"]

        trajectory = synthesize_curve(coeff, M)
        assert trajectory.shape == (M, 3)

        coeff_recovered = extract_fourier_coeffs(trajectory, K)
        assert coeff_recovered.shape == (K, 2, 3)

        err = jnp.max(jnp.abs(coeff - coeff_recovered))
        assert err < 1e-10, f"Fourier round-trip error: {err}"

    def test_roundtrip_random(self):
        """Round-trip with random coefficients."""
        key = jax.random.PRNGKey(42)
        K, M = 32, 256
        coeff = jax.random.normal(key, shape=(K, 2, 3)) * 0.1

        trajectory = synthesize_curve(coeff, M)
        coeff_recovered = extract_fourier_coeffs(trajectory, K)

        err = jnp.max(jnp.abs(coeff - coeff_recovered))
        assert err < 1e-10, f"Random round-trip error: {err}"

    def test_velocity_consistency(self, figure_eight_K64_M512):
        """Velocity from synthesize_velocity matches finite differences."""
        coeff = figure_eight_K64_M512["coeff"]
        M = 2048

        gamma = synthesize_curve(coeff, M)
        gamma_dot = synthesize_velocity(coeff, M)

        dt = 1.0 / M
        gamma_dot_fd = (jnp.roll(gamma, -1, axis=0) - jnp.roll(gamma, 1, axis=0)) / (2 * dt)

        err = jnp.max(jnp.abs(gamma_dot - gamma_dot_fd))
        assert err < 1e-4, f"Velocity FD error: {err}"


class TestActionFunctional:
    """Tests for build_action (issue #6 core)."""

    def test_action_gradient_at_figure_eight(self, figure_eight_K64_M512):
        """
        The figure-eight is a critical point of the action.
        ||grad(A)(theta)|| should be very small.
        """
        theta = figure_eight_K64_M512["theta"]
        K = figure_eight_K64_M512["K"]
        M = figure_eight_K64_M512["M"]
        N = figure_eight_K64_M512["N"]

        action = build_action(N=N, K=K, M=M, T=PERIOD, G=G_CONST, m=MASS)
        grad_action = jax.grad(action)

        g = grad_action(theta)
        grad_norm = jnp.sqrt(jnp.sum(g ** 2))

        assert grad_norm < 1e-4, (
            f"Gradient norm at figure-eight: {grad_norm} "
            f"(expected < 1e-4 for K={K}, M={M})"
        )

    def test_action_value_cross_check(self, figure_eight_K64_M512):
        """
        Cross-check: compute the action in Fourier space and independently
        via numerical quadrature on the Cartesian trajectory.  They should
        agree to within the quadrature error.

        The action in normalised time is:
          A = m/(2T) <|gamma'|^2> + (T/2) G m^2 sum_s <1/r_s>
        where gamma' = d gamma/dt_norm = T * v_phys.
        """
        theta = figure_eight_K64_M512["theta"]
        K = figure_eight_K64_M512["K"]
        M = figure_eight_K64_M512["M"]
        N = figure_eight_K64_M512["N"]

        # Fourier-space action
        action_fn = build_action(N=N, K=K, M=M, T=PERIOD, G=G_CONST, m=MASS)
        A_fourier = float(action_fn(theta))

        # Independent Cartesian quadrature using the scipy-integrated trajectory
        _, positions, velocities = integrate_figure_eight(M=M, n_periods=1)

        # Body 1 positions and velocities (physical)
        q1 = positions[:, 0, :]   # (M, 3)
        v1 = velocities[:, 0, :]  # (M, 3), dq/dt_phys

        # gamma_dot (normalised) = T * v_phys
        v1_norm = v1 * PERIOD

        # Kinetic: m/(2T) * <|gamma_dot|^2>
        kinetic_quad = (MASS / (2.0 * PERIOD)) * np.mean(np.sum(v1_norm ** 2, axis=-1))

        # Force function: G m^2 sum_s <1/|gamma(t) - gamma(t-s/N)|>
        U_quad = 0.0
        for s_idx in range(1, N):
            shift = int(round(s_idx * M / N))
            q1_shifted = np.roll(q1, -shift, axis=0)
            diff = q1 - q1_shifted
            r = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-14)
            U_quad += G_CONST * MASS ** 2 * np.mean(1.0 / r)

        # Action: kinetic + (T/2) * U
        A_quad = kinetic_quad + (PERIOD / 2.0) * U_quad

        rel_err = abs(A_fourier - A_quad) / (abs(A_quad) + 1e-30)
        assert rel_err < 5e-4, (
            f"Action cross-check failed: Fourier={A_fourier}, "
            f"Quadrature={A_quad}, rel_err={rel_err}"
        )

    def test_action_jit_differentiable(self, figure_eight_K64_M512):
        """jax.jit and jax.grad work without error on the action."""
        K = figure_eight_K64_M512["K"]
        M = figure_eight_K64_M512["M"]
        N = figure_eight_K64_M512["N"]

        action = build_action(N=N, K=K, M=M, T=PERIOD)
        grad_action = jax.grad(action)
        jit_grad = jax.jit(grad_action)

        theta = figure_eight_K64_M512["theta"]
        val = float(action(theta))
        g = jit_grad(theta)

        assert jnp.isfinite(val), f"Action value not finite: {val}"
        assert g.shape == theta.shape
        assert jnp.all(jnp.isfinite(g)), "Gradient contains non-finite values"

    def test_action_zero_for_zero_coefficients(self):
        """Action of zero coefficients should be finite (with softening)."""
        K, M, N = 16, 128, 3
        action = build_action(N=N, K=K, M=M)
        theta = jnp.zeros(6 * K)
        val = float(action(theta))
        assert jnp.isfinite(val), f"Action of zero theta is not finite: {val}"
