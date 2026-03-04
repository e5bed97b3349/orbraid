"""
Chenciner-Montgomery figure-eight orbit — reference data for validation.

The figure-eight is a periodic three-body choreography discovered by
Moore (1993) and proven to exist by Chenciner & Montgomery (2000).
All three equal-mass bodies chase each other along a single figure-eight
shaped curve in the plane.

Initial conditions and period are taken from Simo (2002) high-precision
values.  The reference trajectory is integrated with scipy DOP853
(non-JAX) to provide an independent validation target.
"""

import numpy as np
from scipy.integrate import solve_ivp

import jax.numpy as jnp

from orbraid.fourier import extract_fourier_coeffs

# ---------------------------------------------------------------------------
# High-precision initial conditions from Simo (2002) / Chenciner-Montgomery
# ---------------------------------------------------------------------------
# Period (full precision from Simo's numerical continuation)
PERIOD = 6.3259135014546330

# Equal masses
MASS = 1.0

# Gravitational constant
G_CONST = 1.0

# Positions at t = 0
_Q1_0 = np.array([-0.97000436, 0.24308753, 0.0])
_Q2_0 = np.array([0.97000436, -0.24308753, 0.0])
_Q3_0 = np.array([0.0, 0.0, 0.0])

# Velocities at t = 0
# Body 3 velocity from Simo's high-precision data
_V3 = np.array([-0.93240737, -0.86473146, 0.0])
_V1_0 = -0.5 * _V3
_V2_0 = -0.5 * _V3
_V3_0 = _V3.copy()


def chenciner_montgomery_ics():
    """
    Return the initial conditions for the Chenciner-Montgomery figure-eight.

    Returns
    -------
    positions : np.ndarray, shape (3, 3)
        Initial positions of the 3 bodies.
    velocities : np.ndarray, shape (3, 3)
        Initial velocities of the 3 bodies.
    """
    positions = np.stack([_Q1_0, _Q2_0, _Q3_0], axis=0)
    velocities = np.stack([_V1_0, _V2_0, _V3_0], axis=0)
    return positions, velocities


def _three_body_rhs(t, y):
    """Right-hand side for the 3-body ODE: dy/dt = f(t, y)."""
    q = y[:9].reshape(3, 3)    # positions
    v = y[9:].reshape(3, 3)    # velocities

    acc = np.zeros_like(q)
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            rij = q[j] - q[i]
            r = np.sqrt(np.dot(rij, rij))
            acc[i] += G_CONST * MASS * rij / (r ** 3)

    return np.concatenate([v.ravel(), acc.ravel()])


def integrate_figure_eight(M=512, n_periods=1, rtol=1e-13, atol=1e-15):
    """
    Integrate the figure-eight orbit using scipy DOP853.

    Parameters
    ----------
    M : int
        Number of output sample points per period.
    n_periods : int
        Number of periods to integrate.
    rtol, atol : float
        Tolerances for the adaptive integrator.

    Returns
    -------
    t_eval : np.ndarray, shape (M * n_periods,)
        Evaluation times.
    positions : np.ndarray, shape (M * n_periods, 3, 3)
        Positions of the 3 bodies at each time.
    velocities : np.ndarray, shape (M * n_periods, 3, 3)
        Velocities of the 3 bodies at each time.
    """
    q0, v0 = chenciner_montgomery_ics()
    y0 = np.concatenate([q0.ravel(), v0.ravel()])
    T = PERIOD * n_periods
    t_eval = np.linspace(0, T, M * n_periods, endpoint=False)

    sol = solve_ivp(
        _three_body_rhs,
        [0, T + 1e-10],  # slight overshoot to ensure last point is covered
        y0,
        method='DOP853',
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        dense_output=True,
    )

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    positions = sol.y[:9].T.reshape(-1, 3, 3)
    velocities = sol.y[9:].T.reshape(-1, 3, 3)

    return t_eval, positions, velocities


def compute_figure_eight_fourier(K=64, M=512):
    """
    Compute Fourier coefficients of body 1's trajectory in the figure-eight.

    The curve is normalised to unit period: gamma(t) where t in [0, 1).

    Parameters
    ----------
    K : int
        Number of Fourier modes.
    M : int
        Number of sample points for FFT (should be >= 2K, ideally 8K).

    Returns
    -------
    coeff : jnp.ndarray, shape (K, 2, 3)
        Fourier coefficients of body 1's trajectory.
    """
    _, positions, _ = integrate_figure_eight(M=M, n_periods=1)
    # Body 1 trajectory: shape (M, 3)
    body1 = jnp.array(positions[:, 0, :])
    coeff = extract_fourier_coeffs(body1, K)
    return coeff


def compute_figure_eight_energy():
    """
    Compute the total energy (Hamiltonian) of the figure-eight at t=0.

    Returns
    -------
    H : float
        Total energy = kinetic + potential.
    """
    q0, v0 = chenciner_montgomery_ics()

    # Kinetic energy: sum_i (1/2) m |v_i|^2
    kinetic = 0.5 * MASS * np.sum(v0 ** 2)

    # Potential energy: -G sum_{i<j} m^2 / |q_i - q_j|
    potential = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            rij = q0[j] - q0[i]
            r = np.sqrt(np.dot(rij, rij))
            potential -= G_CONST * MASS ** 2 / r

    return kinetic + potential
