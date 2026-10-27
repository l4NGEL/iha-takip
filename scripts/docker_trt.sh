#!/usr/bin/env bash
# TensorRT gelistirme konteynerinde komut calistirir; proje /work'e baglanir.
#   scripts/docker_trt.sh "cmake --build cpp/build-trt"
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
if command -v cygpath >/dev/null; then ROOT="$(cygpath -w "$ROOT")"; fi
MSYS_NO_PATHCONV=1 docker run --rm --gpus all --network host -v "$ROOT:/work" -v iha-data:/data -w /work iha-takip:trt bash -c "$*"
