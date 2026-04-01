"""
bridge — Python ↔ Julia interop layer.

Provides transparent access to the Julia FinchesPhysics module
from Python. Falls back to pure-Python implementations when Julia
is not available.

Usage:
    from src.bridge import julia_available, JuliaBackend

    if julia_available():
        backend = JuliaBackend()
        imap = backend.compute_intermap(sequence)
    else:
        # Fall back to pure-Python
        from src.intermaps import compute_interaction_map_fast
        imap = compute_interaction_map_fast(sequence)
"""

from .julia_bridge import julia_available, JuliaBackend

__all__ = ["julia_available", "JuliaBackend"]
