"""
entropy.py - Entropy measures for sticker arrangement and network structure.

Quantifies the information content and disorder of sticker patterning:

- Spacing entropy: Shannon entropy of inter-sticker distance distribution
- Block entropy: entropy of sticker/linker block length distribution
- Interaction entropy: entropy of the interaction energy distribution
- Network entropy: graph entropy from degree distribution

Key hypothesis: higher spacing entropy -> less regular sticker arrangement
-> weaker/less robust phase separation networks.

This connects sequence patterning to material-level behavior through
information-theoretic measures.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

from .segmentation import StickerMask, compute_linker_lengths


# =============================================================================
# CORE ENTROPY FUNCTIONS
# =============================================================================

def shannon_entropy(counts: np.ndarray) -> float:
    """
    Compute Shannon entropy from a count or probability distribution.

    H = -sum(p * log2(p)) for p > 0

    Parameters
    ----------
    counts : np.ndarray
        Count or probability distribution (non-negative)

    Returns
    -------
    float
        Shannon entropy in bits
    """
    counts = np.asarray(counts, dtype=float)
    total = np.sum(counts)

    if total <= 0:
        return 0.0

    p = counts / total
    p = p[p > 0]  # remove zeros

    return float(-np.sum(p * np.log2(p)))


def normalized_entropy(counts: np.ndarray) -> float:
    """
    Compute normalized Shannon entropy (0 = ordered, 1 = maximum disorder).

    Parameters
    ----------
    counts : np.ndarray
        Count or probability distribution

    Returns
    -------
    float
        Normalized entropy (0 to 1)
    """
    n_nonzero = np.sum(np.asarray(counts) > 0)
    if n_nonzero <= 1:
        return 0.0

    h = shannon_entropy(counts)
    h_max = np.log2(n_nonzero)

    return h / h_max if h_max > 0 else 0.0


# =============================================================================
# STICKER SPACING ENTROPY
# =============================================================================

def compute_spacing_entropy(
    sticker_mask: Union[np.ndarray, StickerMask],
    n_bins: int = 10,
) -> float:
    """
    Compute Shannon entropy of inter-sticker spacing distribution.

    High entropy: spacings are diverse (irregular patterning)
    Low entropy: spacings are uniform (regular patterning)

    Parameters
    ----------
    sticker_mask : Union[np.ndarray, StickerMask]
        Sticker mask
    n_bins : int
        Number of bins for spacing histogram

    Returns
    -------
    float
        Spacing entropy in bits
    """
    if isinstance(sticker_mask, StickerMask):
        positions = sticker_mask.positions
    else:
        positions = np.where(sticker_mask)[0]

    if len(positions) < 2:
        return 0.0

    spacings = np.diff(positions)

    # Bin the spacings
    counts, _ = np.histogram(spacings, bins=n_bins, range=(1, max(spacings) + 1))

    return shannon_entropy(counts)


def compute_normalized_spacing_entropy(
    sticker_mask: Union[np.ndarray, StickerMask],
    n_bins: int = 10,
) -> float:
    """
    Compute normalized spacing entropy (0 = perfectly regular, 1 = maximally disordered).

    Parameters
    ----------
    sticker_mask : Union[np.ndarray, StickerMask]
        Sticker mask
    n_bins : int
        Number of bins

    Returns
    -------
    float
        Normalized spacing entropy (0 to 1)
    """
    if isinstance(sticker_mask, StickerMask):
        positions = sticker_mask.positions
    else:
        positions = np.where(sticker_mask)[0]

    if len(positions) < 2:
        return 0.0

    spacings = np.diff(positions)
    counts, _ = np.histogram(spacings, bins=n_bins, range=(1, max(spacings) + 1))

    return normalized_entropy(counts)


# =============================================================================
# BLOCK LENGTH ENTROPY
# =============================================================================

def compute_block_entropy(
    sticker_mask: Union[np.ndarray, StickerMask],
) -> float:
    """
    Compute entropy of sticker and linker block length distribution.

    Captures the diversity of segment sizes in the sticker-linker chain.

    Parameters
    ----------
    sticker_mask : Union[np.ndarray, StickerMask]
        Sticker mask

    Returns
    -------
    float
        Block entropy in bits
    """
    if isinstance(sticker_mask, StickerMask):
        mask = sticker_mask.mask
    else:
        mask = sticker_mask

    # Find all contiguous blocks (both sticker and linker)
    block_lengths = []
    if len(mask) == 0:
        return 0.0

    current = mask[0]
    length = 1

    for i in range(1, len(mask)):
        if mask[i] == current:
            length += 1
        else:
            block_lengths.append(length)
            current = mask[i]
            length = 1
    block_lengths.append(length)

    if len(block_lengths) <= 1:
        return 0.0

    # Histogram of block lengths
    block_lengths = np.array(block_lengths)
    max_len = int(np.max(block_lengths))
    counts = np.bincount(block_lengths, minlength=max_len + 1)[1:]  # exclude 0

    return shannon_entropy(counts)


# =============================================================================
# INTERACTION ENERGY ENTROPY
# =============================================================================

def compute_interaction_entropy(
    intermap: np.ndarray,
    sticker_mask: Union[np.ndarray, StickerMask],
    n_bins: int = 20,
) -> float:
    """
    Compute entropy of sticker-sticker interaction energy distribution.

    High entropy: diverse interaction strengths (heterogeneous network)
    Low entropy: uniform interactions (homogeneous network)

    Parameters
    ----------
    intermap : np.ndarray
        NxN interaction map
    sticker_mask : Union[np.ndarray, StickerMask]
        Sticker mask
    n_bins : int
        Number of bins for energy histogram

    Returns
    -------
    float
        Interaction entropy in bits
    """
    if isinstance(sticker_mask, StickerMask):
        positions = sticker_mask.positions
    else:
        positions = np.where(sticker_mask)[0]

    if len(positions) < 2:
        return 0.0

    # Extract sticker-sticker energies (upper triangle)
    sub = intermap[np.ix_(positions, positions)]
    upper = np.triu_indices(len(positions), k=1)
    energies = sub[upper]

    if len(energies) == 0:
        return 0.0

    counts, _ = np.histogram(energies, bins=n_bins)
    return shannon_entropy(counts)


# =============================================================================
# NETWORK DEGREE ENTROPY
# =============================================================================

def compute_degree_entropy(degree_sequence: np.ndarray) -> float:
    """
    Compute entropy of the degree distribution of a graph.

    High entropy: heterogeneous connectivity (some hubs, some leaves)
    Low entropy: uniform connectivity (regular graph)

    Parameters
    ----------
    degree_sequence : np.ndarray
        Degree of each node

    Returns
    -------
    float
        Degree entropy in bits
    """
    if len(degree_sequence) == 0:
        return 0.0

    max_deg = int(np.max(degree_sequence))
    counts = np.bincount(degree_sequence.astype(int), minlength=max_deg + 1)

    return shannon_entropy(counts)


# =============================================================================
# COMPOSITIONAL ENTROPY
# =============================================================================

def compute_sticker_composition_entropy(
    sticker_mask: Union[np.ndarray, StickerMask],
    sequence: str,
) -> float:
    """
    Compute entropy of amino acid composition among sticker residues.

    Low entropy: stickers are dominated by one residue type
    High entropy: stickers are chemically diverse

    Parameters
    ----------
    sticker_mask : Union[np.ndarray, StickerMask]
        Sticker mask
    sequence : str
        Amino acid sequence

    Returns
    -------
    float
        Compositional entropy in bits
    """
    if isinstance(sticker_mask, StickerMask):
        positions = sticker_mask.positions
    else:
        positions = np.where(sticker_mask)[0]

    if len(positions) == 0:
        return 0.0

    sticker_aas = [sequence[p] for p in positions]

    # Count each amino acid type
    aa_counts = {}
    for aa in sticker_aas:
        aa_counts[aa] = aa_counts.get(aa, 0) + 1

    counts = np.array(list(aa_counts.values()))
    return shannon_entropy(counts)


# =============================================================================
# COMPREHENSIVE ENTROPY METRICS
# =============================================================================

@dataclass
class EntropyMetrics:
    """Complete entropy metrics for a variant."""
    spacing_entropy: float              # Shannon entropy of inter-sticker spacings
    normalized_spacing_entropy: float   # Normalized (0-1) spacing entropy
    block_entropy: float                # Entropy of sticker/linker block lengths
    interaction_entropy: float          # Entropy of sticker-sticker energies
    composition_entropy: float          # Entropy of sticker amino acid types

    def to_dict(self) -> Dict[str, float]:
        return {
            "spacing_entropy": self.spacing_entropy,
            "normalized_spacing_entropy": self.normalized_spacing_entropy,
            "block_entropy": self.block_entropy,
            "interaction_entropy": self.interaction_entropy,
            "composition_entropy": self.composition_entropy,
        }


def compute_entropy_metrics(
    intermap: np.ndarray,
    sticker_mask: Union[np.ndarray, StickerMask],
    sequence: str,
) -> EntropyMetrics:
    """
    Compute all entropy metrics for a variant.

    Parameters
    ----------
    intermap : np.ndarray
        NxN interaction map
    sticker_mask : Union[np.ndarray, StickerMask]
        Sticker mask
    sequence : str
        Amino acid sequence

    Returns
    -------
    EntropyMetrics
        Complete entropy analysis
    """
    return EntropyMetrics(
        spacing_entropy=compute_spacing_entropy(sticker_mask),
        normalized_spacing_entropy=compute_normalized_spacing_entropy(sticker_mask),
        block_entropy=compute_block_entropy(sticker_mask),
        interaction_entropy=compute_interaction_entropy(intermap, sticker_mask),
        composition_entropy=compute_sticker_composition_entropy(sticker_mask, sequence),
    )
