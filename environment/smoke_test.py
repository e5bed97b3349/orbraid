"""
orbraid M0 smoke test — issue #5 acceptance criteria
=====================================================
Run on the workstation after installing requirements.txt:

    python environment/smoke_test.py

All five checks must pass before proceeding to M1.
"""

import sys
import time

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"

results = []


def check(label, fn):
    try:
        info = fn()
        print(f"{PASS}  {label}" + (f"  [{info}]" if info else ""))
        results.append(True)
    except Exception as e:
        print(f"{FAIL}  {label}  [{e}]")
        results.append(False)


print("\n── orbraid M0 smoke test ──────────────────────────────────────\n")


# 1. JAX imports cleanly
def _jax_import():
    import jax          # noqa: F401
    import jax.numpy    # noqa: F401
    return f"jax {__import__('jax').__version__}"

check("JAX imports", _jax_import)


# 2. GPU device visible and is CUDA
def _gpu_device():
    import jax
    devices = jax.devices()
    cuda = [d for d in devices if d.platform == "gpu"]
    if not cuda:
        raise RuntimeError(f"No GPU found. Devices: {devices}")
    names = ", ".join(str(d) for d in cuda)
    return names

check("CUDA GPU visible (RTX 5090)", _gpu_device)


# 3. jit compilation and execution
def _jit():
    import jax
    import jax.numpy as jnp

    @jax.jit
    def f(x):
        return jnp.sum(x ** 2)

    x = jnp.ones(1024)
    val = float(f(x))
    assert abs(val - 1024.0) < 1e-4, f"Expected 1024.0, got {val}"
    return f"result={val:.1f}"

check("jax.jit compiles and executes", _jit)


# 4. vmap over batch B=8192 (key throughput requirement)
def _vmap():
    import jax
    import jax.numpy as jnp

    B, K = 8192, 64  # batch size and truncation order from CompSpec

    @jax.jit
    def batch_norm(theta_batch):
        # Simulate the shape of a batched action evaluation
        return jax.vmap(lambda theta: jnp.sum(theta ** 2))(theta_batch)

    key = jax.random.PRNGKey(0)
    theta_batch = jax.random.normal(key, shape=(B, 6 * K))
    t0 = time.perf_counter()
    out = batch_norm(theta_batch).block_until_ready()
    dt = time.perf_counter() - t0
    assert out.shape == (B,), f"Expected shape ({B},), got {out.shape}"
    return f"B={B}, K={K}, t={dt*1000:.1f}ms"

check("jax.vmap over B=8192 (no OOM)", _vmap)


# 5. Supporting libraries importable
def _support_libs():
    import optax    # noqa: F401
    import scipy    # noqa: F401
    import h5py     # noqa: F401
    import pyvista  # noqa: F401
    return (
        f"optax={optax.__version__}, "
        f"scipy={scipy.__version__}, "
        f"h5py={h5py.__version__}, "
        f"pyvista={pyvista.__version__}"
    )

check("Supporting libraries (optax, scipy, h5py, pyvista)", _support_libs)


# ── Summary ──────────────────────────────────────────────────────────────
n_pass = sum(results)
n_total = len(results)
print(f"\n── {n_pass}/{n_total} checks passed ", end="")
if n_pass == n_total:
    print("— M0 complete, ready for M1. ✓\n")
    sys.exit(0)
else:
    print("— resolve failures before proceeding.\n")
    sys.exit(1)
