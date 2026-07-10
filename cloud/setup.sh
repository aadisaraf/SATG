#!/usr/bin/env bash
# =============================================================================
# setup.sh — SATG Cloud GPU Environment Setup
# =============================================================================
# Checks CUDA/GPU availability, installs Python dependencies, and verifies
# PyTorch + CUDA are working.
#
# Usage:
#   ./cloud/setup.sh
#
# Exit codes:
#   0 — GPU READY
#   1 — CUDA/GPU check failed or PyTorch CUDA not available
# =============================================================================

set -euo pipefail

trap 'echo "=== SETUP FAILED at line $LINENO ===" >&2; exit 1' ERR

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "  SATG Cloud GPU Environment Setup"
echo "=========================================="
echo ""

# ------------------------------------------------------------------
# 1. Check CUDA / GPU availability
# ------------------------------------------------------------------
echo "--- [1/4] Checking GPU ---"
if command -v nvidia-smi &> /dev/null; then
    echo "  nvidia-smi found:"
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
    echo ""
elif command -v lspci &> /dev/null; then
    echo "  nvidia-smi not found, trying lspci ..."
    if lspci | grep -qi nvidia; then
        echo "  NVIDIA GPU detected via lspci:"
        lspci | grep -i nvidia
        echo "  WARNING: nvidia-smi not found; NVIDIA driver may not be loaded."
    else
        echo "  ERROR: No NVIDIA GPU detected."
        echo "  Ensure you are running on a GPU-enabled instance."
        exit 1
    fi
else
    echo "  ERROR: No GPU detection tools available (nvidia-smi, lspci)."
    exit 1
fi

# ------------------------------------------------------------------
# 2. Install Python dependencies
# ------------------------------------------------------------------
echo ""
echo "--- [2/4] Installing Python dependencies ---"
if command -v uv &> /dev/null; then
    echo "  Using uv (fast)"
    uv pip install -r requirements.txt
elif command -v pip3 &> /dev/null; then
    echo "  Using pip3"
    pip3 install -r requirements.txt
elif command -v pip &> /dev/null; then
    echo "  Using pip"
    pip install -r requirements.txt
else
    echo "  ERROR: No Python package installer found (pip/pip3/uv)."
    exit 1
fi
echo "  Dependencies installed."

# ------------------------------------------------------------------
# 3. Verify PyTorch CUDA
# ------------------------------------------------------------------
echo ""
echo "--- [3/4] Verifying PyTorch CUDA ---"
python -c "
import torch, sys
print(f'  PyTorch version:  {torch.__version__}')
print(f'  CUDA available:   {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA toolk:  {torch.version.cuda}')
    props = torch.cuda.get_device_properties(0)
    print(f'  GPU device:       {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:             {props.total_mem / 1024**3:.1f} GiB')
    print(f'  Compute cap:      {props.major}.{props.minor}')
    print()
    print('  >>> GPU READY <<<')
    sys.exit(0)
else:
    print('  ERROR: PyTorch CUDA is NOT available.')
    print('  Check your PyTorch installation matches your CUDA driver.')
    sys.exit(1)
"
PYTHON_EXIT=$?
if [ "$PYTHON_EXIT" -ne 0 ]; then
    exit 1
fi

# ------------------------------------------------------------------
# 4. Summary
# ------------------------------------------------------------------
echo ""
echo "--- [4/4] Environment Summary ---"
echo "  Work dir: $SCRIPT_DIR"
echo "  Python:   $(python --version 2>&1)"
echo "  Pip:      $(pip --version 2>&1 | head -1)"
echo ""
echo "=========================================="
echo "  Environment ready — proceed to data prep"
echo "=========================================="
