#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMP_DIR="${ROOT_DIR}/temp"

mkdir -p "${TEMP_DIR}"

MODE="${1:-ALL}"
if [[ "${MODE}" != "CPU" && "${MODE}" != "GPU" && "${MODE}" != "ALL" ]]; then
    echo "Usage: $0 [CPU|GPU|ALL]" >&2
    exit 1
fi

echo "Writing GDS outputs to ${TEMP_DIR}"
echo "Mode: ${MODE}"

# ========================================
# CPU Version
# ========================================

if [[ "${MODE}" == "CPU" || "${MODE}" == "ALL" ]]; then
    python -u "${ROOT_DIR}/eval.py" --benchmark CPU_metal
    python -u "${ROOT_DIR}/eval.py" --benchmark CPU_via
    python -u "${ROOT_DIR}/eval.py" --benchmark CPU_activate
    python -u "${ROOT_DIR}/eval.py" --benchmark CPU_poly
    python -u "${ROOT_DIR}/eval.py" --benchmark gcd_activate
    python -u "${ROOT_DIR}/eval.py" --benchmark gcd_metal
    python -u "${ROOT_DIR}/eval.py" --benchmark gcd_polygon
    python -u "${ROOT_DIR}/eval.py" --benchmark gcd_via
    python -u "${ROOT_DIR}/eval.py" --benchmark ibex_active
    python -u "${ROOT_DIR}/eval.py" --benchmark ibex_metal
    python -u "${ROOT_DIR}/eval.py" --benchmark ibex_polygon
    python -u "${ROOT_DIR}/eval.py" --benchmark ibex_via
fi

# ========================================
# GPU Version
# ========================================

if [[ "${MODE}" == "GPU" || "${MODE}" == "ALL" ]]; then
    python -u "${ROOT_DIR}/eval.py" --benchmark CPU_metal --GPU
    python -u "${ROOT_DIR}/eval.py" --benchmark CPU_via --GPU
    python -u "${ROOT_DIR}/eval.py" --benchmark CPU_activate --GPU
    python -u "${ROOT_DIR}/eval.py" --benchmark CPU_poly --GPU
    python -u "${ROOT_DIR}/eval.py" --benchmark gcd_activate --GPU
    python -u "${ROOT_DIR}/eval.py" --benchmark gcd_metal --GPU
    python -u "${ROOT_DIR}/eval.py" --benchmark gcd_polygon --GPU
    python -u "${ROOT_DIR}/eval.py" --benchmark gcd_via --GPU
    python -u "${ROOT_DIR}/eval.py" --benchmark ibex_active --GPU
    python -u "${ROOT_DIR}/eval.py" --benchmark ibex_metal --GPU
    python -u "${ROOT_DIR}/eval.py" --benchmark ibex_polygon --GPU
    python -u "${ROOT_DIR}/eval.py" --benchmark ibex_via --GPU
fi

echo "Done."
