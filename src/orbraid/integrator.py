"""
Yoshida 6th-order symplectic integrator for N-body problems.

Implements the Yoshida (1990) Set-1 composition of Störmer-Verlet (leapfrog)
steps.  A 6th-order method composes 8 leapfrog substeps with carefully chosen
coefficients so that all error terms up to O(dt^5) cancel.

The integrator preserves the symplectic structure of the Hamiltonian flow,
giving long-term energy conservation bounded by O(dt^6) with no secular drift.

Force computation uses jax.grad of the potential energy for exact autodiff
derivatives — no finite-differencing needed.

Issue: https://github.com/e5bed97b3349/orbraid/issues/7
CompSpec: §4.1
"""

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Yoshida (1990) 6th-order, Set 1 coefficients
# ---------------------------------------------------------------------------
# These are the 8 position-step coefficients (d_i) and 8 momentum-step
# coefficients (c_i) for the 6th-order method.
#
# Reference: H. Yoshida, "Construction of higher order symplectic integrators",
#            Phys. Lett. A 150 (1990), 262-268.
#
# The method is built from the 2nd-order Störmer-Verlet (leapfrog) step S(h)
# via triple composition:
#   S6(h) = S2(w3 h) S2(w2 h) S2(w1 h) S2(w0 h) S2(w1 h) S2(w2 h) S2(w3 h)
#
# where the weights satisfy sum(w_i) = 1 and cancel error terms through O(h^5).

# Fourth-order Yoshida (intermediate building block)
_W1_4 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
_W0_4 = 1.0 - 2.0 * _W1_4

# Sixth-order Yoshida Set 1
_W1_6 = 1.0 / (2.0 - 2.0 ** (1.0 / 5.0))
_W0_6 = 1.0 - 2.0 * _W1_6

# The full 6th-order integrator is a triple composition of the 4th-order:
# S6(h) = S4(z1 h) S4(z0 h) S4(z1 h)
# where z1 = 1/(2 - 2^(1/5)), z0 = 1 - 2*z1
#
# Expanding, the 4th-order is itself S4(h) = S2(w1 h) S2(w0 h) S2(w1 h)
# So S6 has 3*3 = 9 S2 substeps, but adjacent ones combine, giving 7 distinct
# S2 steps with weights:

def _yoshida6_weights():
    """Compute the 7 Störmer-Verlet substep weights for Yoshida 6th order."""
    # 4th-order weights
    w = jnp.array([_W1_4, _W0_4, _W1_4])
    # 6th-order composition weights
    z = jnp.array([_W1_6, _W0_6, _W1_6])
    # Expand: each z[i] scales the entire 4th-order sequence
    weights = []
    for zi in z:
        for wj in w:
            weights.append(float(zi * wj))
    # Adjacent substeps between blocks merge
    # weights = [z0*w0, z0*w1, z0*w2+z1*w0, z1*w1, z1*w2+z2*w0, z2*w1, z2*w2]
    # But we keep them separate for the leapfrog decomposition.
    return jnp.array(weights)


# Precompute the 9 substep weights (some adjacent pairs will be summed
# in the leapfrog decomposition)
_SUBSTEP_WEIGHTS = _yoshida6_weights()


def _build_nbody_potential(N, G, masses):
    """
    Build a potential energy function for N point masses.

    Parameters
    ----------
    N : int
        Number of bodies.
    G : float
        Gravitational constant.
    masses : jnp.ndarray, shape (N,)
        Masses of the bodies.

    Returns
    -------
    potential : callable
        potential(q) -> scalar, where q has shape (N, 3).
    """
    def potential(q):
        """Gravitational potential energy V(q) = -G sum_{i<j} m_i m_j / |q_i - q_j|"""
        V = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                rij = q[j] - q[i]
                r = jnp.sqrt(jnp.sum(rij ** 2))
                V -= G * masses[i] * masses[j] / r
        return V
    return potential


def _build_nbody_force(N, G, masses):
    """
    Build a force function using jax.grad of the potential.

    Returns force(q) -> shape (N, 3), where force = -grad(V).
    """
    potential = _build_nbody_potential(N, G, masses)

    def force(q):
        return -jax.grad(potential)(q)

    return force


def yoshida_step(q, p, dt, force_fn, masses):
    """
    One Yoshida 6th-order symplectic step.

    Uses the Störmer-Verlet (position-Verlet) decomposition:
    For each substep weight w_i:
        q <- q + (w_i * dt / 2) * p / m     (half position kick)
        p <- p + (w_i * dt) * F(q)           (full momentum kick)
        q <- q + (w_i * dt / 2) * p / m     (half position kick)

    But adjacent half-kicks merge, so the actual implementation uses the
    standard DKD (drift-kick-drift) formulation for efficiency.

    Parameters
    ----------
    q : jnp.ndarray, shape (N, 3)
        Positions.
    p : jnp.ndarray, shape (N, 3)
        Momenta (= masses * velocities).
    dt : float
        Base timestep.
    force_fn : callable
        force_fn(q) -> shape (N, 3).
    masses : jnp.ndarray, shape (N, 1)
        Masses (broadcast-ready, shape (N, 1) for division).

    Returns
    -------
    q_new, p_new : jnp.ndarray
        Updated positions and momenta.
    """
    weights = _SUBSTEP_WEIGHTS

    for w in weights:
        h = w * dt
        # Position-Verlet (DKD) for this substep:
        # Half drift
        q = q + (h / 2.0) * p / masses
        # Full kick
        p = p + h * force_fn(q)
        # Half drift
        q = q + (h / 2.0) * p / masses

    return q, p


def integrate_yoshida(q0, p0, dt, n_steps, force_fn, masses):
    """
    Integrate an N-body system using the Yoshida 6th-order method.

    Uses jax.lax.scan for JIT-compatible time stepping.

    Parameters
    ----------
    q0 : jnp.ndarray, shape (N, 3)
        Initial positions.
    p0 : jnp.ndarray, shape (N, 3)
        Initial momenta.
    dt : float
        Timestep.
    n_steps : int
        Number of integration steps.
    force_fn : callable
        force_fn(q) -> (N, 3) force array.
    masses : jnp.ndarray, shape (N, 1)
        Masses, broadcast-ready.

    Returns
    -------
    q_final : jnp.ndarray, shape (N, 3)
        Final positions.
    p_final : jnp.ndarray, shape (N, 3)
        Final momenta.
    """
    weights = _SUBSTEP_WEIGHTS

    def scan_step(carry, _):
        q, p = carry
        for w in weights:
            h = w * dt
            q = q + (h / 2.0) * p / masses
            p = p + h * force_fn(q)
            q = q + (h / 2.0) * p / masses
        return (q, p), None

    (q_final, p_final), _ = jax.lax.scan(scan_step, (q0, p0), None, length=n_steps)
    return q_final, p_final


def integrate_yoshida_trajectory(q0, p0, dt, n_steps, force_fn, masses):
    """
    Like integrate_yoshida but returns the full trajectory.

    Returns
    -------
    q_traj : jnp.ndarray, shape (n_steps + 1, N, 3)
        Positions at each step (including initial).
    p_traj : jnp.ndarray, shape (n_steps + 1, N, 3)
        Momenta at each step (including initial).
    """
    weights = _SUBSTEP_WEIGHTS

    def scan_step(carry, _):
        q, p = carry
        for w in weights:
            h = w * dt
            q = q + (h / 2.0) * p / masses
            p = p + h * force_fn(q)
            q = q + (h / 2.0) * p / masses
        return (q, p), (q, p)

    (q_final, p_final), (q_traj, p_traj) = jax.lax.scan(
        scan_step, (q0, p0), None, length=n_steps
    )

    # Prepend initial state
    q_traj = jnp.concatenate([q0[None, ...], q_traj], axis=0)
    p_traj = jnp.concatenate([p0[None, ...], p_traj], axis=0)

    return q_traj, p_traj


def build_nbody_force(N, G=1.0, m=1.0):
    """
    Convenience function: build an N-body gravitational force function.

    Parameters
    ----------
    N : int
        Number of bodies.
    G : float
        Gravitational constant.
    m : float
        Mass of each body (equal masses assumed).

    Returns
    -------
    force_fn : callable
        JIT-compatible force_fn(q) -> (N, 3).
    potential_fn : callable
        JIT-compatible potential_fn(q) -> scalar.
    """
    masses = jnp.ones(N) * m
    potential_fn = _build_nbody_potential(N, G, masses)
    force_fn = _build_nbody_force(N, G, masses)
    return force_fn, potential_fn
