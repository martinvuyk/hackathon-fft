#!/usr/bin/env bash
# Pixi activation hook (NVIDIA-only): load machine-local
# MODULAR_NVPTX_COMPILER_PATH if present.
# Created by: pixi run enable-nvidia-gpu
#
# Harmless on AMD / Apple / CPU-only: the env file simply does not exist unless
# enable-nvidia-gpu wrote it after detecting an NVIDIA GPU.

_nvidia_gpu_env="${PIXI_PROJECT_ROOT:-.}/.nvidia-gpu-env"
if [[ -f "${_nvidia_gpu_env}" ]]; then
  # shellcheck disable=SC1090
  source "${_nvidia_gpu_env}"
fi
unset _nvidia_gpu_env
