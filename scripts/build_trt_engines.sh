#!/usr/bin/env bash
# Build the TensorRT engines of README "Getting Started" step 6 without trtexec: the same five
# FP16 engines, built with polygraphy from the conda environment. Runs the tree's own
# scripts/export_*.py, simplifies the Omnidata graphs, and writes the engines into
# <root>/pretrained_models/, where the runtime picks them up automatically (it falls back to
# PyTorch while they are absent). Every step keeps an output that already exists, so an
# interrupted build resumes; delete the file to redo a step. Move the .engine files out of
# pretrained_models/ to get the PyTorch path back.
#
#   bash scripts/build_trt_engines.sh            # this repository
#   bash scripts/build_trt_engines.sh <root>     # another checkout with the same layout
#
# Engines are device-specific: build on the machine that runs them.
set -euo pipefail

ROOT=$(cd "${1:-$(dirname "${BASH_SOURCE[0]}")/..}" && pwd)
PM="$ROOT/pretrained_models"
WORKSPACE_MB=24576     # TensorRT scratch pool for build and inference; lower it on a smaller GPU

# -- preflight ---------------------------------------------------------------------------------
for f in droid.pth omnidata_dpt_depth_v2.ckpt omnidata_dpt_normal_v2.ckpt; do
    [ -e "$PM/$f" ] || { echo "missing $PM/$f (README step 5)"; exit 1; }
done
python -c "import tensorrt, polygraphy, onnxsim, torch" 2>/dev/null \
    || { echo "tensorrt / polygraphy / onnxsim not importable: activate the environment of README step 3"; exit 1; }
FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
NEED_MB=$((WORKSPACE_MB + 3072))
if [ "$FREE_MB" -lt "$NEED_MB" ]; then
    echo "only ${FREE_MB} MiB of GPU memory free; the build wants ~${NEED_MB} MiB (workspace ${WORKSPACE_MB} + models)."
    echo "Another GPU job is probably running: wait for it, or lower WORKSPACE_MB in this script."
    exit 1
fi
echo "[trt] tree $ROOT | ${FREE_MB} MiB free | tensorrt $(python -c 'import tensorrt; print(tensorrt.__version__)')"

keep() { [ -e "$1" ] && echo "[trt] keep $1"; }

# -- 1. export ONNX; from the tree root, the export scripts use paths relative to it -----------
cd "$ROOT"
keep "$PM/omnidata_normal_512.onnx"   || python scripts/export_omnidata.py
keep "$PM/update_module_partial.onnx" || python scripts/export_droidnet.py

# -- 2. simplify the Omnidata graphs; in pretrained_models/, the .onnx.data files are relative --
cd "$PM"
for m in depth normal; do
    keep "omnidata_${m}_512_simplified.onnx" \
        || python -m onnxsim "omnidata_${m}_512.onnx" "omnidata_${m}_512_simplified.onnx" --skip-fuse-bn
done

# -- 3. build the FP16 engines -----------------------------------------------------------------
build() {   # build <onnx> <engine> [optimisation-profile flags]
    local onnx=$1 engine=$2; shift 2
    keep "$engine" && return 0
    echo "[trt] build $engine"
    polygraphy convert "$onnx" --convert-to trt --fp16 --pool-limit "workspace:${WORKSPACE_MB}M" "$@" -o "$engine"
}
build omnidata_depth_512_simplified.onnx  omnidata_depth_512_simplified_fp16.engine
build omnidata_normal_512_simplified.onnx omnidata_normal_512_simplified_fp16.engine
# DroidNet feature encoder, dynamic resolution
build droidnet_fnet.onnx droidnet_fnet_fp16.engine \
    --trt-min-shapes input:[1,1,3,328,328] \
    --trt-opt-shapes input:[1,1,3,368,584] \
    --trt-max-shapes input:[1,1,3,656,656]
# DroidNet update module, frontend window
build update_module_partial.onnx update_module_partial_fp16.engine \
    --trt-min-shapes net:[1,1,128,41,41]  inp:[1,1,128,41,41]  corr:[1,1,196,41,41]  flow:[1,1,4,41,41] \
    --trt-opt-shapes net:[1,24,128,43,77] inp:[1,24,128,43,77] corr:[1,24,196,43,77] flow:[1,24,4,43,77] \
    --trt-max-shapes net:[1,60,128,82,82] inp:[1,60,128,82,82] corr:[1,60,196,82,82] flow:[1,60,4,82,82]
# DroidNet update module, backend / PGBA / final BA window
build update_module_partial.onnx update_module_partial_pgba_fp16.engine \
    --trt-min-shapes net:[1,1,128,41,41]   inp:[1,1,128,41,41]   corr:[1,1,196,41,41]   flow:[1,1,4,41,41] \
    --trt-opt-shapes net:[1,85,128,43,77]  inp:[1,85,128,43,77]  corr:[1,85,196,43,77]  flow:[1,85,4,43,77] \
    --trt-max-shapes net:[1,120,128,82,82] inp:[1,120,128,82,82] corr:[1,120,196,82,82] flow:[1,120,4,82,82]

echo "[trt] done:"
ls -la "$PM"/*.engine
echo "[trt] runs started from $ROOT now use these engines; move them out of pretrained_models/ for the PyTorch path."
