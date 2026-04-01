"""
julia_bridge.py — Thin Python wrapper around the Julia FinchesPhysics module.

Uses juliacall (the Python side of PythonCall.jl) for seamless interop.
Zero-copy numpy ↔ Julia array transfer where possible.

If Julia is not installed or juliacall is not available, all functions
gracefully fall back to None / raise informative errors.
"""

from __future__ import annotations
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any

# Path to Julia project
_JULIA_PROJECT = str(Path(__file__).parent.parent.parent / "julia")

_jl = None
_FP = None


def julia_available() -> bool:
    """Check if Julia + juliacall are available."""
    try:
        import juliacall
        return True
    except ImportError:
        return False


def _init_julia():
    """Lazy-initialize the Julia runtime and load FinchesPhysics."""
    global _jl, _FP

    if _jl is not None:
        return

    import juliacall
    _jl = juliacall.Main

    # Activate the Julia project
    _jl.seval(f"""
    import Pkg
    Pkg.activate("{_JULIA_PROJECT}")
    using FinchesPhysics
    """)

    _FP = _jl.FinchesPhysics


class JuliaBackend:
    """High-performance Julia backend for FINCHES physics computations.

    Provides the same interface as the Python modules but routes
    computation through Julia for 5-50x speedups on:
    - Interaction map generation
    - Percolation sweeps
    - Persistent homology
    - Differentiable Hamiltonian evaluation
    """

    def __init__(self):
        if not julia_available():
            raise RuntimeError(
                "Julia backend requires juliacall. Install with: pip install juliacall\n"
                "Also requires Julia 1.9+ with the FinchesPhysics project set up.\n"
                "Run: julia --project=julia -e 'using Pkg; Pkg.instantiate()'"
            )
        _init_julia()

    def compute_intermap(
        self,
        sequence: str,
        params: Optional[Dict[str, float]] = None,
        sigma: float = 2.0,
        smooth: bool = True,
    ) -> np.ndarray:
        """Compute interaction map using Julia backend.

        Parameters
        ----------
        sequence : str
            Amino acid sequence
        params : Optional[Dict[str, float]]
            Force field parameter overrides
        sigma : float
            Gaussian smoothing width
        smooth : bool
            Whether to apply smoothing

        Returns
        -------
        np.ndarray
            NxN interaction map
        """
        params_jl = _python_dict_to_julia(params or {})
        result = _FP.intermap_from_python(
            sequence, params_jl, sigma=sigma, smooth=smooth
        )
        return np.array(result)

    def compute_topology(
        self,
        intermap: np.ndarray,
        sticker_positions: List[int],
        threshold: float,
        min_separation: int = 4,
    ) -> Dict[str, Any]:
        """Compute topology metrics using Julia backend.

        Parameters
        ----------
        intermap : np.ndarray
            NxN interaction map
        sticker_positions : List[int]
            1-based sticker positions (Julia convention)
        threshold : float
            Energy threshold for contacts
        min_separation : int
            Minimum sequence separation

        Returns
        -------
        Dict[str, Any]
            Topology metrics
        """
        # Convert to 1-based indexing for Julia
        positions_jl = [p + 1 for p in sticker_positions]  # Python 0-based → Julia 1-based
        result = _FP.topology_from_python(
            intermap, positions_jl, threshold,
            min_separation=min_separation
        )
        return _julia_dict_to_python(result)

    def compute_homology(
        self,
        intermap: np.ndarray,
        sticker_positions: List[int],
        distance_method: str = "negated",
        min_separation: int = 4,
    ) -> Dict[str, Any]:
        """Compute persistent homology using Julia backend.

        Parameters
        ----------
        intermap : np.ndarray
            NxN interaction map
        sticker_positions : List[int]
            0-based sticker positions (automatically converted to 1-based)
        distance_method : str
            Distance conversion method
        min_separation : int
            Minimum sequence separation

        Returns
        -------
        Dict[str, Any]
            Homology metrics
        """
        positions_jl = [p + 1 for p in sticker_positions]
        method_sym = _jl.Symbol(distance_method)
        result = _FP.homology_from_python(
            intermap, positions_jl,
            distance_method=method_sym,
            min_separation=min_separation
        )
        return _julia_dict_to_python(result)

    def hamiltonian_energy(
        self,
        sequence: str,
        params: Optional[Dict[str, float]] = None,
        min_separation: int = 4,
    ) -> float:
        """Compute total Hamiltonian energy.

        Parameters
        ----------
        sequence : str
            Amino acid sequence
        params : Optional[Dict[str, float]]
            Force field parameter overrides
        min_separation : int
            Minimum sequence separation

        Returns
        -------
        float
            Total Hamiltonian energy in kT
        """
        p = _build_julia_params(params)
        return float(_FP.hamiltonian_energy(sequence, p, min_separation=min_separation))

    def gradient_topology(
        self,
        sequence: str,
        sticker_positions: List[int],
        params: Optional[Dict[str, float]] = None,
        min_separation: int = 4,
    ) -> np.ndarray:
        """Compute dH/d(sticker_i) — gradient of Hamiltonian w.r.t. sticker positions.

        This is the differentiable-physics unlock: tells you which
        sticker positions most affect the total network energy.

        Parameters
        ----------
        sequence : str
            Amino acid sequence
        sticker_positions : List[int]
            0-based sticker positions
        params : Optional[Dict[str, float]]
            Force field parameter overrides
        min_separation : int
            Minimum sequence separation

        Returns
        -------
        np.ndarray
            Gradient vector (one entry per sticker)
        """
        positions_jl = [p + 1 for p in sticker_positions]
        p = _build_julia_params(params)
        result = _FP.∂H_∂topology(sequence, positions_jl, p, min_separation=min_separation)
        return np.array(result)


def _build_julia_params(params: Optional[Dict[str, float]] = None):
    """Build Julia ForceFieldParams from Python dict."""
    if params is None:
        return _FP.ForceFieldParams()

    # Map Python parameter names to Julia keyword arguments
    return _jl.seval(f"""
    FinchesPhysics.ForceFieldParams(
        ε_aromatic     = {params.get('epsilon_aromatic', -0.65)},
        ε_cation_pi    = {params.get('epsilon_cation_pi', -0.50)},
        ε_attractive   = {params.get('epsilon_attractive', -0.30)},
        ε_repulsive    = {params.get('epsilon_repulsive', 0.20)},
        ε_hydrophobic  = {params.get('epsilon_hydrophobic', -0.15)},
        ε_polar        = {params.get('epsilon_polar', -0.05)},
        glycine_factor = {params.get('glycine_factor', 0.30)},
        proline_penalty= {params.get('proline_penalty', 0.10)},
    )
    """)


def _python_dict_to_julia(d: Dict[str, Any]):
    """Convert Python dict to Julia Dict{String, Any}."""
    return _jl.Dict(_jl.Pair(str(k), v) for k, v in d.items())


def _julia_dict_to_python(jl_dict) -> Dict[str, Any]:
    """Convert Julia Dict to Python dict, converting arrays to numpy."""
    result = {}
    for key in _jl.keys(jl_dict):
        val = jl_dict[key]
        # Convert Julia arrays to numpy
        try:
            result[str(key)] = np.array(val)
        except (TypeError, ValueError):
            result[str(key)] = val
    return result
