"""
FINCHES-FUS-Physics: Physics-based analysis pipeline for FUS low-complexity domain.

This package provides tools for:
- Sequence management and mutation utilities
- FINCHES/MPIPI-GG style residue-residue interaction models
- Interaction map generation and smoothing
- Sticker-linker segmentation
- Biophysical metrics computation
- Publication-quality visualization
"""

from . import sequences
from . import forcefield
from . import intermaps
from . import segmentation
from . import metrics
from . import topology
from . import homology
from . import entropy
from . import hamiltonian
from . import variants
from . import calibration
from . import plotting

__version__ = "0.2.0"
__author__ = "Arjun Mathur"
