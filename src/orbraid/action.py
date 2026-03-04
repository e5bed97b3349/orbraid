"""
Action functional for N-body choreographies in Fourier space.

Implements build_action(N, K, M, T) per CompSpec §2.3, with two bug fixes
in the potential_shift function (lines 339, 341 of CompSpec) and a
correction for the period-dependent kinetic/potential balance.

Bug fixes in CompSpec potential_shift:
  CompSpec line 339:  einsum(sin_s, cos_t, a)  ->  einsum(sin_s, sin_t, a)
  CompSpec line 341:  einsum(sin_s, sin_t, b)  ->  -einsum(sin_s, cos_t, b)

These corrections follow from the cos/sin addition formula for gamma(t - s):
    cos(k(t-s)) = cos(ks) cos(kt) + sin(ks) sin(kt)
    sin(k(t-s)) = cos(ks) sin(kt) - sin(ks) cos(kt)

Period correction:
  The CompSpec omits the period T from the action, implicitly assuming T=1.
  For a choreography with physical period T, the correct action in normalised
  time t in [0,1) is:

    A[gamma] = m/(2T) * <|gamma'|^2> + (T/2) * G m^2 * sum_s <1/r_s>

  where gamma' = d gamma / dt_norm = T * v_physical.  The physical
  Lagrangian action S = integral_0^T [KE + U] dt_phys transforms under
  t_phys = T * t_norm to give the T-dependent prefactors above.

Issue: https://github.com/e5bed97b3349/orbraid/issues/6
"""

import jax
import jax.numpy as jnp


def build_action(N, K, M, T=1.0, G=1.0, m=1.0):
    """
    Build a JAX-jittable action functional for an N-body choreography.

    Parameters
    ----------
    N : int
        Number of equal-mass bodies.
    K : int
        Fourier truncation order (modes k = 1..K).
    M : int
        Number of quadrature points in [0, 1).
    T : float
        Period of the orbit.  The Fourier coefficients encode gamma(t)
        for t in [0, 1), which maps to physical time [0, T).
    G : float
        Gravitational constant.
    m : float
        Mass of each body.

    Returns
    -------
    action : callable
        A function action(theta) -> scalar, where theta has shape (6*K,).
        theta encodes Fourier coefficients reshaped internally to (K, 2, 3):
          coeff[:, 0, :] = a_k (cosine coefficients)
          coeff[:, 1, :] = b_k (sine coefficients)
        Centre-of-mass is assumed zeroed (a_0 = 0).
    """
    t = jnp.linspace(0.0, 1.0, M, endpoint=False)       # normalised time
    tau = jnp.arange(1, N) / N                           # phase offsets s = 1/N, ..., (N-1)/N

    @jax.jit
    def action(theta):
        # Unpack: (6K,) -> (K, 2, 3)
        coeff = theta.reshape(K, 2, 3)
        ks = jnp.arange(1, K + 1)                       # (K,)

        # Precompute trig basis: shape (K, M)
        phase = 2.0 * jnp.pi * ks[:, None] * t[None, :]
        cos_t = jnp.cos(phase)                           # (K, M)
        sin_t = jnp.sin(phase)                           # (K, M)

        # gamma(t): (M, 3)
        gamma = (jnp.einsum('ki,kc->ic', cos_t, coeff[:, 0, :])
                 + jnp.einsum('ki,kc->ic', sin_t, coeff[:, 1, :]))

        # gamma_dot(t) = d gamma / dt_norm: (M, 3)
        omega_k = 2.0 * jnp.pi * ks                     # (K,)
        gamma_dot = (jnp.einsum('k,ki,kc->ic', omega_k, -sin_t, coeff[:, 0, :])
                     + jnp.einsum('k,ki,kc->ic', omega_k, cos_t, coeff[:, 1, :]))

        # Kinetic term: m/(2T) * <|gamma_dot|^2>
        # Physical velocity v = gamma_dot / T, kinetic per body = (1/2)m|v|^2
        # Integrated over dt_phys = T dt_norm, times N bodies, divided by T:
        #   N * (1/2) m |gamma_dot|^2 / T^2 * T / T = N m |gamma_dot|^2 / (2T)
        # Dividing by N (overall constant): m/(2T) * <|gamma_dot|^2>
        kinetic = (m / (2.0 * T)) * jnp.mean(jnp.sum(gamma_dot ** 2, axis=-1))

        # Force function (positive): U = sum_s G m^2 <1/|gamma(t) - gamma(t-s)|>
        def force_function_shift(s):
            """Compute G m^2 <1/|gamma(t) - gamma(t - s)|> for one shift s."""
            phase_s = 2.0 * jnp.pi * ks * s              # (K,)
            cos_s = jnp.cos(phase_s)                      # (K,)
            sin_s = jnp.sin(phase_s)                      # (K,)

            # gamma(t - s) via addition formulas
            gamma_s = (
                jnp.einsum('k,ki,kc->ic', cos_s, cos_t, coeff[:, 0, :])   # a_k cos_s cos_t
                + jnp.einsum('k,ki,kc->ic', sin_s, sin_t, coeff[:, 0, :]) # a_k sin_s sin_t  [FIXED]
                + jnp.einsum('k,ki,kc->ic', cos_s, sin_t, coeff[:, 1, :]) # b_k cos_s sin_t
                - jnp.einsum('k,ki,kc->ic', sin_s, cos_t, coeff[:, 1, :]) # -b_k sin_s cos_t [FIXED]
            )

            diff = gamma - gamma_s                        # (M, 3)
            r = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-14)  # softened distance
            return G * m ** 2 * jnp.mean(1.0 / r)

        # Total force function summed over all phase shifts
        U_total = jax.vmap(force_function_shift)(tau).sum()

        # Action: A = kinetic + (T/2) * U
        # From the Lagrangian L = KE + U (force function convention),
        # integrated over normalised time with period factors.
        return kinetic + (T / 2.0) * U_total

    return action
