#!/usr/bin/env bash
# setup_stack.sh — Bootstrap the dual Python + Julia stack.
#
# Usage:
#   chmod +x setup_stack.sh
#   ./setup_stack.sh
#
# Prerequisites:
#   - Python 3.9+ (with pip)
#   - Julia 1.9+ (https://julialang.org/downloads/)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JULIA_PROJECT="$SCRIPT_DIR/julia"

echo "=== FINCHES-FUS-Physics: Dual-Stack Setup ==="
echo ""

# ── Step 1: Python dependencies ──────────────────────────────────────────
echo "[1/4] Installing Python dependencies..."
if command -v pip &> /dev/null; then
    pip install -r "$SCRIPT_DIR/requirements.txt"
elif command -v pip3 &> /dev/null; then
    pip3 install -r "$SCRIPT_DIR/requirements.txt"
else
    echo "ERROR: pip not found. Please install Python 3.9+ first."
    exit 1
fi
echo "  ✓ Python packages installed"
echo ""

# ── Step 2: Check Julia ──────────────────────────────────────────────────
echo "[2/4] Checking Julia installation..."
if ! command -v julia &> /dev/null; then
    echo "  Julia not found. Install from: https://julialang.org/downloads/"
    echo "  (Recommended: Julia 1.9+)"
    echo ""
    echo "  On macOS with Homebrew:  brew install julia"
    echo "  On Linux:                curl -fsSL https://install.julialang.org | sh"
    echo ""
    echo "  Skipping Julia setup. Python-only mode will work."
    echo "  Re-run this script after installing Julia."
    echo ""
    SKIP_JULIA=1
else
    JULIA_VERSION=$(julia --version | grep -oE '[0-9]+\.[0-9]+')
    echo "  ✓ Julia $JULIA_VERSION found"
    SKIP_JULIA=0
fi

# ── Step 3: Julia project instantiation ──────────────────────────────────
if [ "$SKIP_JULIA" -eq 0 ]; then
    echo "[3/4] Installing Julia dependencies (this may take a few minutes on first run)..."
    julia --project="$JULIA_PROJECT" -e '
        import Pkg
        Pkg.instantiate()
        Pkg.precompile()
        println("  ✓ Julia packages installed and precompiled")
    '
    echo ""
else
    echo "[3/4] Skipped (Julia not available)"
    echo ""
fi

# ── Step 4: Verify ───────────────────────────────────────────────────────
echo "[4/4] Verifying installation..."

# Python verification
python3 -c "
import numpy, scipy, matplotlib
print('  ✓ Python core: numpy, scipy, matplotlib')
try:
    import networkx
    print('  ✓ Python graph: networkx')
except ImportError:
    print('  ⚠ networkx not available')
try:
    import gtda
    print('  ✓ Python TDA: giotto-tda')
except ImportError:
    print('  ⚠ giotto-tda not available (optional)')
try:
    import juliacall
    print('  ✓ Python-Julia bridge: juliacall')
except ImportError:
    print('  ⚠ juliacall not available (Julia features disabled)')
"

# Julia verification
if [ "$SKIP_JULIA" -eq 0 ]; then
    julia --project="$JULIA_PROJECT" -e '
        using FinchesPhysics
        println("  ✓ Julia FinchesPhysics module loads successfully")
        # Quick smoke test
        seq = "MASNDYTQQATQSYGAYPTQPGQGY"
        H = hamiltonian_energy(seq)
        println("  ✓ Hamiltonian energy test: H = $(round(H, digits=2)) kT")
    ' 2>/dev/null || echo "  ⚠ Julia module failed to load (run tests for details)"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Quick start:"
echo "  Python:  python run_pipeline.py"
echo "  Julia:   julia --project=julia -e 'using FinchesPhysics'"
echo "  Tests:   julia --project=julia test/runtests.jl"
echo "  Notebooks: cd notebooks && jupyter lab"
