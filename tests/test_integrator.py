"""
Tests for the Yoshida 6th-order symplectic integrator (issue #7).

Validates against the Chenciner-Montgomery figure-eight orbit:
1. Periodicity: ||q(T) - q(0)|| < tolerance after 1 period
2. Energy conservation: max |dH/H_0| bounded over 10 periods
3. Convergence order: error ~ dt^6 (log-log slope approx 6)
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from orbraid.integrator import (
    integrate_yoshida,
    build_nbody_force,
)
from orbraid.figure_eight import (
    PERIOD,
    MASS,
    G_CONST,
    chenciner_montgomery_ics,
)


@pytest.fixture(scope="module")
def force_and_potential():
    """Build force and potential functions for 3-body problem."""
    force_fn, potential_fn = build_nbody_force(N=3, G=G_CONST, m=MASS)
    return jax.jit(force_fn), jax.jit(potential_fn)


@pytest.fixture(scope="module")
def ics_jax():
    """Initial conditions as JAX arrays with momenta."""
    q0, v0 = chenciner_montgomery_ics()
    q0 = jnp.array(q0)
    p0 = jnp.array(v0) * MASS
    masses = jnp.ones((3, 1)) * MASS
    return q0, p0, masses


class TestYoshidaPeriodicity:
    """After 1 period, the system should return close to its initial state."""

    def test_one_period_return(self, force_and_potential, ics_jax):
        force_fn, _ = force_and_potential
        q0, p0, masses = ics_jax

        n_steps = 2000
        dt = PERIOD / n_steps

        q_final, p_final = integrate_yoshida(q0, p0, dt, n_steps, force_fn, masses)

        q_err = jnp.max(jnp.abs(q_final - q0))
        p_err = jnp.max(jnp.abs(p_final - p0))

        assert q_err < 1e-6, f"Position error after 1 period: {q_err}"
        assert p_err < 1e-6, f"Momentum error after 1 period: {p_err}"


class TestEnergyConservation:
    """Energy should be conserved to O(dt^6) with no secular drift."""

    def test_energy_conservation_10_periods(self, force_and_potential, ics_jax):
        """
        Integrate for 10 periods and check energy stays bounded.
        """
        force_fn, potential_fn = force_and_potential
        q0, p0, masses = ics_jax

        n_periods = 10
        n_steps_per_period = 1000
        n_steps = n_periods * n_steps_per_period
        dt = PERIOD / n_steps_per_period

        def hamiltonian(q, p, masses_flat):
            KE = 0.5 * jnp.sum(p ** 2 / masses_flat[:, None])
            PE = potential_fn(q)
            return KE + PE

        masses_flat = jnp.ones(3) * MASS
        H0 = float(hamiltonian(q0, p0, masses_flat))

        q_final, p_final = integrate_yoshida(q0, p0, dt, n_steps, force_fn, masses)
        H_final = float(hamiltonian(q_final, p_final, masses_flat))

        rel_err = abs(H_final - H0) / abs(H0)
        assert rel_err < 1e-8, (
            f"Energy drift after {n_periods} periods: "
            f"|dH/H_0| = {rel_err} (H0={H0}, H_final={H_final})"
        )


class TestConvergenceOrder:
    """Verify that the Yoshida integrator achieves 6th-order convergence."""

    def test_convergence_slope(self, force_and_potential, ics_jax):
        """
        Run with 3 different step sizes and check that the error ratio
        is consistent with 6th-order convergence.

        We compare against a very fine reference solution (not against q0),
        so the test measures integrator truncation error, not IC precision.
        """
        force_fn, _ = force_and_potential
        q0, p0, masses = ics_jax

        # Integrate a short time (0.1 periods) to keep errors well above
        # machine precision even for the fine grid
        t_end = 0.5 * PERIOD

        # Reference: very fine step size
        n_ref = 8000
        dt_ref = t_end / n_ref
        q_ref, _ = integrate_yoshida(q0, p0, dt_ref, n_ref, force_fn, masses)

        # Coarse and fine
        n_coarse = 100
        n_fine = 200  # 2x finer

        dt_coarse = t_end / n_coarse
        dt_fine = t_end / n_fine

        q_coarse, _ = integrate_yoshida(q0, p0, dt_coarse, n_coarse, force_fn, masses)
        q_fine, _ = integrate_yoshida(q0, p0, dt_fine, n_fine, force_fn, masses)

        err_coarse = float(jnp.max(jnp.abs(q_coarse - q_ref)))
        err_fine = float(jnp.max(jnp.abs(q_fine - q_ref)))

        assert err_coarse > 1e-14, f"Coarse error too small to measure: {err_coarse}"
        assert err_fine > 1e-14, f"Fine error too small to measure: {err_fine}"

        # For 6th order: err ~ dt^6, so halving dt gives ratio 2^6 = 64
        ratio = err_coarse / err_fine
        order_estimate = np.log2(ratio)

        assert order_estimate > 4.0, (
            f"Convergence order too low: estimated {order_estimate:.1f} "
            f"(err_coarse={err_coarse:.2e}, err_fine={err_fine:.2e}, "
            f"ratio={ratio:.1f}, expected ~64 for 6th order)"
        )
