# orbraid — Environment Setup Guide

**Target platform:** Windows 11, NVIDIA RTX 5090 (Blackwell GB202, SM_100),
WSL2 Ubuntu 24.04 LTS, miniforge, JAX/CUDA 12.

**Acceptance criterion (M0 / issue #5):** `python environment/smoke_test.py`
exits 0 with all 5 checks green.

---

## Architecture overview

JAX has no native Windows support. The correct substrate is WSL2, which gives
a full Linux kernel while the NVIDIA Windows driver exposes the GPU to it.

```
Windows NVIDIA driver (≥ 572.16)
        │
        │  libcuda.so stub at /usr/lib/wsl/lib/  (injected by driver)
        ▼
WSL2 Ubuntu 24.04 kernel
        │
        │  miniforge (conda-forge only)
        ▼
conda env: orbraid
        │
        │  jax[cuda12] pip wheel  ← bundles its own CUDA 12.8+ runtime
        ▼
python environment/smoke_test.py
```

**Critical:** do **not** install the NVIDIA CUDA toolkit inside WSL2.
The Windows driver already provides `libcuda.so` via the WSL2 stub mechanism.
Installing a second CUDA stack breaks the stub path and causes JAX to fall
back to CPU. The `jax[cuda12]` pip wheel bundles its own `nvidia-cuda-*`
runtime packages — no system CUDA is needed.

---

## Step 0 — Windows prerequisites

### 0.1  NVIDIA driver

The RTX 5090 (SM_100, Blackwell) requires driver **≥ 572.16** for WSL2 GPU
passthrough. Download from https://www.nvidia.com/drivers and install.

Confirm inside WSL2 after Step 1:

```bash
nvidia-smi
```

Expected: your RTX 5090 appears with CUDA Version ≥ 12.8.

### 0.2  WSL2 feature flag (PowerShell as Administrator)

```powershell
wsl --install
wsl --set-default-version 2
```

If WSL2 is already installed, just confirm the version:

```powershell
wsl --status        # should show "Default Version: 2"
```

---

## Step 1 — Install Ubuntu 24.04 LTS inside WSL2

```powershell
wsl --install -d Ubuntu-24.04
```

Once the Ubuntu shell opens, complete the initial user setup (username +
password). Then update the base system:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git curl wget
```

Clone the repo into your WSL2 home (or work directly from the Windows
filesystem path, e.g. `/mnt/c/Users/bryn/...`):

```bash
git clone https://github.com/e5bed97b3349/orbraid.git
cd orbraid
```

---

## Step 2 — Install miniforge

Miniforge ships with conda and mamba pre-configured to use **conda-forge
only** — which is exactly what the environment requires.

```bash
curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
     -o /tmp/miniforge.sh
bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
rm /tmp/miniforge.sh
"$HOME/miniforge3/bin/conda" init bash
source ~/.bashrc
```

Verify:

```bash
conda --version        # ≥ 24.x
mamba --version        # ≥ 1.x
```

### 2.1  Lock down to conda-forge only (repo-wide)

```bash
conda config --system --add channels conda-forge
conda config --system --set channel_priority strict
conda config --system --remove channels defaults 2>/dev/null || true
```

Confirm:

```bash
conda config --show channels
# channels:
#   - conda-forge
```

---

## Step 3 — Create the environment

**Option A — from `environment.yml` (flexible, resolves latest-matching
versions):**

```bash
conda env create -f environment/environment.yml
```

**Option B — from `conda-lock.yml` (bit-for-bit reproducible, one command):**

```bash
conda-lock install --name orbraid environment/conda-lock.yml
```

Use Option B whenever the lock file exists and you want an exact replica of
a known-good build. Use Option A after updating `environment.yml`, then
regenerate the lock file (Step 4).

Activate:

```bash
conda activate orbraid
```

---

## Step 4 — Verify JAX sees the GPU

Before running the smoke test, do a quick sanity check:

```bash
python - <<'EOF'
import jax
print("JAX version :", jax.__version__)
print("Devices     :", jax.devices())
EOF
```

Expected output:

```
JAX version : 0.4.3x
Devices     : [CudaDevice(id=0)]
```

If you see `[CpuDevice(id=0)]` instead, see **Troubleshooting** below.

---

## Step 5 — Run the acceptance test

```bash
cd /path/to/orbraid
python environment/smoke_test.py
```

All 5 checks must pass:

```
── orbraid M0 smoke test ──────────────────────────────────────

  PASS  JAX imports  [jax 0.4.3x]
  PASS  CUDA GPU visible (RTX 5090)  [CudaDevice(id=0)]
  PASS  jax.jit compiles and executes  [result=1024.0]
  PASS  jax.vmap over B=8192 (no OOM)  [B=8192, K=64, t=xx.xms]
  PASS  Supporting libraries (optax, scipy, h5py, pyvista)  [...]

── 5/5 checks passed — M0 complete, ready for M1. ✓
```

---

## Step 6 — Generate the conda lock file

After any change to `environment.yml`, regenerate the lock file so
collaborators can reproduce the exact build:

```bash
conda activate orbraid    # conda-lock must be on PATH
conda-lock lock \
    --mamba \
    --platform linux-64 \
    -f environment/environment.yml \
    -k unified \
    --lockfile environment/conda-lock.yml
```

Commit both files:

```bash
git add environment/environment.yml environment/conda-lock.yml
git commit -m "env: update conda-lock for <reason>"
```

---

## Troubleshooting

### JAX falls back to CPU (`[CpuDevice(id=0)]`)

1. Check `nvidia-smi` runs inside WSL2 and shows the RTX 5090 with
   CUDA ≥ 12.8. If it fails, the Windows driver is too old (need ≥ 572.16)
   or WSL2 integration is disabled in the NVIDIA control panel.

2. Confirm no system CUDA toolkit is installed in WSL2:

   ```bash
   which nvcc     # should print nothing
   ls /usr/local/cuda* 2>/dev/null   # should be empty
   ```

   If a CUDA toolkit is present, it may be shadowing the WSL2 stub.
   Remove it: `sudo apt remove --purge cuda* nvidia-cuda-toolkit`.

3. Confirm the stub is visible:

   ```bash
   ls /usr/lib/wsl/lib/libcuda.so*
   ```

   If absent, the Windows driver install did not set up WSL2 integration.
   Reinstall the driver with "WSL2 integration" checked.

4. Confirm `jax[cuda12]` is the pip wheel, not a CPU-only jax:

   ```bash
   pip show jax | grep -i location
   pip list | grep nvidia-cuda    # should list several nvidia-cuda-* packages
   ```

### `snappy` import error

SnapPy may require `libgomp` on some Ubuntu installs:

```bash
sudo apt install -y libgomp1
```

### `pyvista` headless rendering error in WSL2

If any later code calls `pyvista.start_xvfb()` and fails:

```bash
sudo apt install -y xvfb libgl1-mesa-glx
```

The `xvfbwrapper` conda package (already in `environment.yml`) provides the
Python side; the system packages above provide the X/GL binaries.

### RTX 5090 SM_100 not recognised by JAX

If JAX reports a PTX compilation error or falls back to CPU despite the GPU
being visible, the bundled CUDA runtime in the wheel may pre-date Blackwell.
Force the latest wheel:

```bash
pip install -U "jax[cuda12]"
```

JAX ≥ 0.4.35 ships with CUDA 12.8 which has SM_100 support.
