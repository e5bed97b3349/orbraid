# Changelog

All notable changes to this project will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-03-04

Milestone M0 complete: environment validated, action functional and Yoshida
integrator implemented, figure-eight orbit reproduced to 1e-12.

### Added

- `src/orbraid/fourier.py` — Fourier synthesis and extraction (`synthesize_curve`, `synthesize_velocity`, `extract_fourier_coeffs`)
- `src/orbraid/action.py` — `build_action(N, K, M, T, G, m)` per CompSpec §2.3, with period `T` parameter
- `src/orbraid/integrator.py` — Yoshida 6th-order symplectic integrator (`yoshida_step`, `integrate_yoshida`, `build_nbody_force`)
- `src/orbraid/figure_eight.py` — Chenciner-Montgomery figure-eight reference orbit (Simó 2002 ICs, DOP853 integration, FFT extraction)
- `src/orbraid/__init__.py` — package init, enables `jax_enable_x64`, exports public API
- `pyproject.toml` — setuptools config for `pip install -e .`
- `tests/` — 10 tests covering Fourier round-trip, action gradient/cross-check/JIT, periodicity, energy conservation, and 6th-order convergence
- `environment/environment.yml` — miniforge spec (Python 3.11, JAX[cuda12], conda-forge only)
- `environment/SETUP.md` — WSL2/CUDA12 setup guide
- `environment/pip-lock.txt` — pip dependency snapshot (JAX 0.9.1, CUDA 13.2)

### Fixed

- CompSpec §2.3 `potential_shift`: corrected two trig addition formula errors in the `γ(t−s)` expansion (line 339: `sin_s, cos_t` → `sin_s, sin_t` for a_k term; line 341: `+sin_s, sin_t` → `−sin_s, cos_t` for b_k term). CompSpec bumped to v0.3.
- CompSpec §2.3: added explicit period `T` to action formula. The spec implicitly assumed T=1; the implementation now accepts `T` as a parameter for correct kinetic/potential balance.

### Changed

- RDD bumped from v0.1 to v0.2 (non-planarity reframed, dimensional miracle argument, video render spec)
- CompSpec bumped from v0.1 to v0.2 (C₃ᵥ worked example fix, Reynolds operator replaced with O(K) projector), then to v0.3 (trig bug fixes above)
- `environment/requirements.txt` — JAX floor raised from `>=0.4.25` to `>=0.4.35` for Blackwell SM_100 support

## [0.1.0] — 2026-03-04

Initial repository structure and documentation.

### Added

- `docs/rdd/rdd.tex` — Research Design Document v0.1
- `docs/compspec/compspec.tex` — Computational Specification v0.1
- `docs/mathframe/mathframe.tex` — Mathematical Framework v0.1
- `README.md` — project overview, milestone table, citation
- `LICENSE` — MIT licence
- `environment/requirements.txt` — initial pip requirements
- `environment/smoke_test.py` — 5-check GPU validation script
