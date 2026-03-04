"""
Fourier synthesis and analysis for periodic curves in R^3.

The choreographic curve gamma(t) is represented by a truncated real Fourier
series with K modes (centre-of-mass zeroed, so a_0 = 0):

    gamma(t) = sum_{k=1}^{K} [ a_k cos(2 pi k t) + b_k sin(2 pi k t) ]

where t is in [0, 1] (normalised period) and a_k, b_k in R^3.

The coefficient tensor has shape (K, 2, 3):
    coeff[k, 0, :] = a_{k+1}   (cosine coefficients)
    coeff[k, 1, :] = b_{k+1}   (sine coefficients)

All functions are jax.jit-compatible and differentiable.
"""

import jax
import jax.numpy as jnp


def synthesize_curve(coeff, M):
    """
    Fourier coefficients -> trajectory in R^3.

    Parameters
    ----------
    coeff : jnp.ndarray, shape (K, 2, 3)
        Fourier coefficients.  coeff[:, 0, :] = a_k (cosine),
        coeff[:, 1, :] = b_k (sine), for k = 1..K.
    M : int
        Number of uniformly spaced sample points in [0, 1).

    Returns
    -------
    gamma : jnp.ndarray, shape (M, 3)
        Sampled trajectory.
    """
    K = coeff.shape[0]
    t = jnp.linspace(0.0, 1.0, M, endpoint=False)           # (M,)
    ks = jnp.arange(1, K + 1)[:, None]                       # (K, 1)
    phase = 2.0 * jnp.pi * ks * t[None, :]                   # (K, M)
    cos_t = jnp.cos(phase)                                    # (K, M)
    sin_t = jnp.sin(phase)                                    # (K, M)
    gamma = (jnp.einsum('ki,kc->ic', cos_t, coeff[:, 0, :])
             + jnp.einsum('ki,kc->ic', sin_t, coeff[:, 1, :]))
    return gamma


def synthesize_velocity(coeff, M):
    """
    Fourier coefficients -> velocity (time derivative) in R^3.

    gamma_dot(t) = sum_{k=1}^K 2 pi k [ -a_k sin(2 pi k t) + b_k cos(2 pi k t) ]

    Parameters
    ----------
    coeff : jnp.ndarray, shape (K, 2, 3)
    M : int

    Returns
    -------
    gamma_dot : jnp.ndarray, shape (M, 3)
    """
    K = coeff.shape[0]
    t = jnp.linspace(0.0, 1.0, M, endpoint=False)
    ks = jnp.arange(1, K + 1)[:, None]                       # (K, 1)
    omega_k = 2.0 * jnp.pi * ks[:, 0]                        # (K,)
    phase = 2.0 * jnp.pi * ks * t[None, :]                   # (K, M)
    cos_t = jnp.cos(phase)
    sin_t = jnp.sin(phase)
    gamma_dot = (jnp.einsum('k,ki,kc->ic', omega_k, -sin_t, coeff[:, 0, :])
                 + jnp.einsum('k,ki,kc->ic', omega_k, cos_t, coeff[:, 1, :]))
    return gamma_dot


def extract_fourier_coeffs(trajectory, K):
    """
    Trajectory in R^3 -> Fourier coefficients via FFT.

    Parameters
    ----------
    trajectory : jnp.ndarray, shape (M, 3)
        Uniformly sampled trajectory (M points in one period).
    K : int
        Number of Fourier modes to extract (k = 1..K).

    Returns
    -------
    coeff : jnp.ndarray, shape (K, 2, 3)
        coeff[:, 0, :] = a_k (cosine), coeff[:, 1, :] = b_k (sine).
    """
    M = trajectory.shape[0]
    # FFT along the time axis for each spatial component
    # fft_vals[k, c] = sum_{m=0}^{M-1} trajectory[m, c] * exp(-2 pi i k m / M)
    fft_vals = jnp.fft.fft(trajectory, axis=0)               # (M, 3)

    # For a real signal x(t) = a_0/2 + sum_k [a_k cos(2 pi k t) + b_k sin(2 pi k t)]
    # the DFT gives: X[k] = (M/2)(a_k - i b_k) for k = 1..K
    # So a_k = 2 Re(X[k]) / M,  b_k = -2 Im(X[k]) / M

    a_k = 2.0 * jnp.real(fft_vals[1:K + 1, :]) / M           # (K, 3)
    b_k = -2.0 * jnp.imag(fft_vals[1:K + 1, :]) / M          # (K, 3)

    coeff = jnp.stack([a_k, b_k], axis=1)                     # (K, 2, 3)
    return coeff
