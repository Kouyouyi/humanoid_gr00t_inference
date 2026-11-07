#!/usr/bin/env bash
# Source only system/vendor SDK libraries, never another application's environment.
_GROOT_NATIVE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TensorRT_ROOT="${TensorRT_ROOT:-/home/kinetra-5090-01/TensorRT}"
export onnxruntime_ROOT="${onnxruntime_ROOT:-/opt/onnxruntime}"
export CUDAToolkit_ROOT="${CUDAToolkit_ROOT:-/usr/local/cuda-12.8}"
export LD_LIBRARY_PATH="$_GROOT_NATIVE_DIR/lib:$TensorRT_ROOT/lib:$onnxruntime_ROOT/lib:$CUDAToolkit_ROOT/lib64:${LD_LIBRARY_PATH:-}"
export HAS_ROS2=0
unset SONIC_AUTO_START
unset _GROOT_NATIVE_DIR
