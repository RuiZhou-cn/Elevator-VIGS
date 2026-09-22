#!/usr/bin/env bash
# Native build of README "Getting Started" step 4 -- everything that has to be compiled against
# this conda environment's PyTorch, in order:
#   1. sophuspy and torch-scatter from source (the PyPI sophuspy wheel segfaults during IMU
#      initialisation; torch-scatter must match the installed torch)
#   2. the vigs_backends and lietorch CUDA kernels (setup.py)
#   3. the IMU pre-integrator, CMake/pybind11 -- required, it has no Python fallback
#   4. the two Gaussian-splatting kernels vendored under thirdparty/
# About 10 minutes. Stops at the first failing step; rerun after fixing it.
#
#   conda activate elevator-vigs
#   bash scripts/install.sh
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

# -- preflight ---------------------------------------------------------------------------------
python -c "import torch, pybind11" 2>/dev/null \
    || { echo "torch / pybind11 not importable: activate the environment first (README step 3)"; exit 1; }
command -v cmake >/dev/null || { echo "cmake not found: it ships with the conda environment (README step 3)"; exit 1; }
command -v g++   >/dev/null || { echo "g++ not found: sudo apt install build-essential (README step 2)"; exit 1; }
CUDA_HOME_TORCH=$(python -c "from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME or '')")
[ -n "$CUDA_HOME_TORCH" ] && [ -x "$CUDA_HOME_TORCH/bin/nvcc" ] \
    || { echo "no CUDA toolkit found: install CUDA 12.8 and export CUDA_HOME (README step 2)."; \
         echo "Without it torch-scatter silently builds CPU-only and the CUDA kernels cannot build."; exit 1; }
for f in thirdparty/eigen/Eigen/Core thirdparty/Sophus/sophus/so3.hpp thirdparty/lietorch_5090/setup.py \
         thirdparty/simple-knn/setup.py thirdparty/diff-gaussian-rasterization/third_party/glm/glm/glm.hpp; do
    [ -e "$f" ] || { echo "missing $f: fetch the submodules with 'git submodule update --init --recursive'"; exit 1; }
done
echo "[install] python $(command -v python) | nvcc $CUDA_HOME_TORCH/bin/nvcc | $(g++ --version | head -1)"

# -- 1. source builds against this environment's PyTorch ---------------------------------------
echo "[install] 1/4 sophuspy and torch-scatter from source"
pip install --no-binary=:all: --no-deps --force-reinstall --no-cache-dir sophuspy
pip install --no-binary=:all: --no-build-isolation --no-cache-dir torch-scatter

# -- 2. vigs_backends and lietorch (two torch CUDAExtensions in setup.py) -----------------------
echo "[install] 2/4 vigs_backends and lietorch CUDA kernels"
python setup.py install

# -- 3. IMU pre-integrator; the .so lands in vigs/imu_cpp/build/, which the code adds to sys.path
echo "[install] 3/4 IMU pre-integrator"
cmake -S vigs/imu_cpp -B vigs/imu_cpp/build -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE="$(command -v python)"
cmake --build vigs/imu_cpp/build -j "$(nproc)"

# -- 4. Gaussian-splatting kernels; follow TORCH_CUDA_ARCH_LIST, else the GPU present ----------
echo "[install] 4/4 simple-knn and diff-gaussian-rasterization"
pip install --no-build-isolation thirdparty/simple-knn thirdparty/diff-gaussian-rasterization

# -- check -------------------------------------------------------------------------------------
python - <<'PY'
import sys; sys.path.append("vigs/imu_cpp/build")
import torch  # noqa: F401  -- loads libc10 before the kernels that link it
import vigs_backends, lietorch, sophuspy, simple_knn, diff_gaussian_rasterization, imu_integrator_cpp  # noqa: F401
import torch_scatter, os
assert any(f.endswith("_cuda.so") for f in os.listdir(os.path.dirname(torch_scatter.__file__))), "torch_scatter built CPU-only"
print("[install] done: every native module imports")
PY
