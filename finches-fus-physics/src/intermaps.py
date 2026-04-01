"""
intermaps.py - FINCHES-style interaction map generation.

This module generates NxN residue-residue interaction maps for protein sequences.
Maps can be:
- Raw: Direct lookup from force field
- Smoothed: Gaussian-smoothed to capture local interaction neighborhoods
- Normalized: Scaled for comparison across sequences

The interaction map I[i,j] represents the interaction energy between
residue i and residue j based on their amino acid identities.
"""

from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Dict, Optional, Union, Tuple
from dataclasses import dataclass

from .forcefield import (
    get_default_matrix,
    aa_index,
    AA_ORDER,
    N_AMINO_ACIDS
)
from .sequences import SequenceRecord, get_variant_registry


# =============================================================================
# INTERACTION MAP GENERATION
# =============================================================================

def compute_interaction_map(
    sequence: Union[str, SequenceRecord],
    interaction_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute the raw NxN interaction map for a sequence.

    I[i,j] = E(aa_i, aa_j) from the force field.

    Parameters
    ----------
    sequence : Union[str, SequenceRecord]
        Amino acid sequence or SequenceRecord object
    interaction_matrix : Optional[np.ndarray]
        20x20 amino acid interaction matrix. Uses default if None.

    Returns
    -------
    np.ndarray
        NxN interaction map where N = sequence length
    """
    if isinstance(sequence, SequenceRecord):
        seq = sequence.sequence
    else:
        seq = sequence

    if interaction_matrix is None:
        interaction_matrix = get_default_matrix()

    n = len(seq)
    intermap = np.zeros((n, n), dtype=np.float64)

    # Build index array for vectorized lookup
    indices = np.array([aa_index(aa) for aa in seq])

    # Vectorized construction using advanced indexing
    for i in range(n):
        intermap[i, :] = interaction_matrix[indices[i], indices]

    return intermap


def compute_interaction_map_fast(
    sequence: Union[str, SequenceRecord],
    interaction_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Fast vectorized interaction map computation.

    Uses outer product indexing for better performance on long sequences.
    """
    if isinstance(sequence, SequenceRecord):
        seq = sequence.sequence
    else:
        seq = sequence

    if interaction_matrix is None:
        interaction_matrix = get_default_matrix()

    # Convert sequence to indices
    indices = np.array([aa_index(aa) for aa in seq])
    n = len(indices)

    # Use fancy indexing: matrix[i, j] for all pairs
    i_idx = np.tile(indices, (n, 1)).T  # (n, n) with rows = i
    j_idx = np.tile(indices, (n, 1))    # (n, n) with cols = j

    return interaction_matrix[i_idx, j_idx]


# =============================================================================
# SMOOTHING
# =============================================================================

def smooth_interaction_map(
    intermap: np.ndarray,
    sigma: float = 2.0,
    mode: str = "reflect"
) -> np.ndarray:
    """
    Apply Gaussian smoothing to an interaction map.

    Smoothing captures the local interaction neighborhood and
    reduces noise from individual residue pairs.

    Parameters
    ----------
    intermap : np.ndarray
        NxN interaction map
    sigma : float
        Gaussian kernel standard deviation (in residues)
    mode : str
        Boundary handling mode for scipy.ndimage.gaussian_filter

    Returns
    -------
    np.ndarray
        Smoothed interaction map
    """
    return gaussian_filter(intermap, sigma=sigma, mode=mode)


def smooth_interaction_map_anisotropic(
    intermap: np.ndarray,
    sigma_i: float = 2.0,
    sigma_j: float = 2.0,
    mode: str = "reflect"
) -> np.ndarray:
    """
    Apply anisotropic Gaussian smoothing.

    Different smoothing widths along sequence axes.
    """
    return gaussian_filter(intermap, sigma=(sigma_i, sigma_j), mode=mode)


# =============================================================================
# NORMALIZATION
# =============================================================================

def normalize_interaction_map(
    intermap: np.ndarray,
    method: str = "zscore"
) -> np.ndarray:
    """
    Normalize an interaction map.

    Parameters
    ----------
    intermap : np.ndarray
        NxN interaction map
    method : str
        Normalization method:
        - "zscore": (x - mean) / std
        - "minmax": Scale to [0, 1]
        - "symmetric": Scale to [-1, 1] with 0 at mean
        - "energy": Keep physical units (no normalization)

    Returns
    -------
    np.ndarray
        Normalized interaction map
    """
    if method == "energy":
        return intermap.copy()

    elif method == "zscore":
        mean = np.mean(intermap)
        std = np.std(intermap)
        if std < 1e-10:
            return np.zeros_like(intermap)
        return (intermap - mean) / std

    elif method == "minmax":
        min_val = np.min(intermap)
        max_val = np.max(intermap)
        if max_val - min_val < 1e-10:
            return np.zeros_like(intermap)
        return (intermap - min_val) / (max_val - min_val)

    elif method == "symmetric":
        abs_max = np.max(np.abs(intermap))
        if abs_max < 1e-10:
            return np.zeros_like(intermap)
        return intermap / abs_max

    else:
        raise ValueError(f"Unknown normalization method: {method}")


# =============================================================================
# FINCHES-STYLE PROCESSING PIPELINE
# =============================================================================

@dataclass
class InterMapConfig:
    """Configuration for interaction map generation."""
    smooth: bool = True
    sigma: float = 2.0
    normalize: bool = True
    normalize_method: str = "zscore"
    mask_diagonal: bool = False
    diagonal_width: int = 5


def generate_finches_intermap(
    sequence: Union[str, SequenceRecord],
    config: Optional[InterMapConfig] = None,
    interaction_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Generate a FINCHES-style processed interaction map.

    Full pipeline: raw -> smoothed -> normalized

    Parameters
    ----------
    sequence : Union[str, SequenceRecord]
        Input sequence
    config : Optional[InterMapConfig]
        Processing configuration
    interaction_matrix : Optional[np.ndarray]
        Force field matrix

    Returns
    -------
    np.ndarray
        Processed interaction map
    """
    if config is None:
        config = InterMapConfig()

    # Step 1: Raw interaction map
    intermap = compute_interaction_map_fast(sequence, interaction_matrix)

    # Step 2: Smooth
    if config.smooth:
        intermap = smooth_interaction_map(intermap, sigma=config.sigma)

    # Step 3: Mask diagonal (remove self and near-neighbor interactions)
    if config.mask_diagonal:
        intermap = mask_diagonal_band(intermap, width=config.diagonal_width)

    # Step 4: Normalize
    if config.normalize:
        intermap = normalize_interaction_map(intermap, method=config.normalize_method)

    return intermap


def mask_diagonal_band(
    intermap: np.ndarray,
    width: int = 5,
    fill_value: float = 0.0
) -> np.ndarray:
    """
    Mask the diagonal band of an interaction map.

    Residues close in sequence are always in contact; masking removes
    trivial local interactions to highlight longer-range contacts.

    Parameters
    ----------
    intermap : np.ndarray
        NxN interaction map
    width : int
        Half-width of diagonal band to mask (|i-j| <= width)
    fill_value : float
        Value to fill masked region

    Returns
    -------
    np.ndarray
        Masked interaction map
    """
    result = intermap.copy()
    n = result.shape[0]

    for k in range(-width, width + 1):
        np.fill_diagonal(result[max(0, k):, max(0, -k):], fill_value)

    return result


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def compute_all_intermaps(
    variants: Optional[Dict[str, SequenceRecord]] = None,
    config: Optional[InterMapConfig] = None,
    interaction_matrix: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute interaction maps for all variants.

    Parameters
    ----------
    variants : Optional[Dict[str, SequenceRecord]]
        Variant registry. Uses default if None.
    config : Optional[InterMapConfig]
        Processing configuration
    interaction_matrix : Optional[np.ndarray]
        Force field matrix

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping variant names to interaction maps
    """
    if variants is None:
        variants = get_variant_registry()

    if config is None:
        config = InterMapConfig()

    intermaps = {}
    for name, record in variants.items():
        intermaps[name] = generate_finches_intermap(
            record,
            config=config,
            interaction_matrix=interaction_matrix
        )

    return intermaps


# =============================================================================
# DIFFERENCE MAPS
# =============================================================================

def compute_difference_map(
    intermap_variant: np.ndarray,
    intermap_reference: np.ndarray
) -> np.ndarray:
    """
    Compute difference map (variant - reference).

    Positive values: variant has stronger (more negative) interactions
    Negative values: reference has stronger interactions

    Parameters
    ----------
    intermap_variant : np.ndarray
        Variant interaction map
    intermap_reference : np.ndarray
        Reference (usually WT) interaction map

    Returns
    -------
    np.ndarray
        Difference map
    """
    if intermap_variant.shape != intermap_reference.shape:
        raise ValueError(
            f"Shape mismatch: variant {intermap_variant.shape} vs "
            f"reference {intermap_reference.shape}"
        )
    return intermap_variant - intermap_reference


def compute_all_difference_maps(
    intermaps: Dict[str, np.ndarray],
    reference_name: str = "WT"
) -> Dict[str, np.ndarray]:
    """
    Compute difference maps for all variants relative to reference.

    Parameters
    ----------
    intermaps : Dict[str, np.ndarray]
        Dictionary of interaction maps
    reference_name : str
        Name of reference variant

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary of difference maps (excludes reference)
    """
    if reference_name not in intermaps:
        raise KeyError(f"Reference '{reference_name}' not found in intermaps")

    ref_map = intermaps[reference_name]
    diff_maps = {}

    for name, intermap in intermaps.items():
        if name != reference_name:
            diff_maps[name] = compute_difference_map(intermap, ref_map)

    return diff_maps


# =============================================================================
# INTERACTION MAP ANALYSIS
# =============================================================================

def compute_map_statistics(intermap: np.ndarray) -> Dict[str, float]:
    """
    Compute summary statistics for an interaction map.

    Parameters
    ----------
    intermap : np.ndarray
        NxN interaction map

    Returns
    -------
    Dict[str, float]
        Dictionary of statistics
    """
    # Exclude diagonal for some statistics
    n = intermap.shape[0]
    off_diag_mask = ~np.eye(n, dtype=bool)
    off_diag = intermap[off_diag_mask]

    return {
        "mean": float(np.mean(intermap)),
        "std": float(np.std(intermap)),
        "min": float(np.min(intermap)),
        "max": float(np.max(intermap)),
        "mean_off_diagonal": float(np.mean(off_diag)),
        "total_interaction": float(np.sum(intermap)),
        "n_attractive": int(np.sum(intermap < 0)),
        "n_repulsive": int(np.sum(intermap > 0)),
        "fraction_attractive": float(np.mean(intermap < 0)),
    }


def extract_row_profile(
    intermap: np.ndarray,
    position: int
) -> np.ndarray:
    """
    Extract the interaction profile for a single residue position.

    Parameters
    ----------
    intermap : np.ndarray
        NxN interaction map
    position : int
        0-indexed residue position

    Returns
    -------
    np.ndarray
        1D array of interactions with all other residues
    """
    return intermap[position, :]


def extract_column_profile(
    intermap: np.ndarray,
    position: int
) -> np.ndarray:
    """
    Extract the column profile (should equal row for symmetric maps).
    """
    return intermap[:, position]


def compute_mean_interaction_profile(
    intermap: np.ndarray
) -> np.ndarray:
    """
    Compute mean interaction strength per residue position.

    Parameters
    ----------
    intermap : np.ndarray
        NxN interaction map

    Returns
    -------
    np.ndarray
        1D array of mean interactions per position
    """
    return np.mean(intermap, axis=1)


def compute_local_interaction_density(
    intermap: np.ndarray,
    window: int = 10
) -> np.ndarray:
    """
    Compute local interaction density along the sequence.

    Sums interaction strengths within a sliding window along the diagonal.

    Parameters
    ----------
    intermap : np.ndarray
        NxN interaction map
    window : int
        Window size

    Returns
    -------
    np.ndarray
        1D array of local interaction densities
    """
    n = intermap.shape[0]
    density = np.zeros(n)

    for i in range(n):
        i_start = max(0, i - window)
        i_end = min(n, i + window + 1)
        density[i] = np.mean(intermap[i_start:i_end, i_start:i_end])

    return density


# =============================================================================
# CONTACT MAP EXTRACTION
# =============================================================================

def extract_contact_map(
    intermap: np.ndarray,
    threshold: float = -0.2,
    mode: str = "attractive"
) -> np.ndarray:
    """
    Extract a binary contact map from interaction map.

    Parameters
    ----------
    intermap : np.ndarray
        NxN interaction map
    threshold : float
        Energy threshold for contact
    mode : str
        "attractive": contacts where E < threshold
        "repulsive": contacts where E > threshold
        "absolute": contacts where |E| > |threshold|

    Returns
    -------
    np.ndarray
        Binary contact map (bool)
    """
    if mode == "attractive":
        return intermap < threshold
    elif mode == "repulsive":
        return intermap > threshold
    elif mode == "absolute":
        return np.abs(intermap) > np.abs(threshold)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# =============================================================================
# SERIALIZATION
# =============================================================================

def save_intermaps(
    intermaps: Dict[str, np.ndarray],
    output_path: str
) -> None:
    """Save interaction maps to npz file."""
    np.savez_compressed(output_path, **intermaps)


def load_intermaps(input_path: str) -> Dict[str, np.ndarray]:
    """Load interaction maps from npz file."""
    data = np.load(input_path)
    return {key: data[key] for key in data.files}


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def quick_intermap(
    sequence: Union[str, SequenceRecord],
    sigma: float = 2.0,
    normalize: bool = True
) -> np.ndarray:
    """
    Quick generation of a FINCHES-style interaction map.

    Parameters
    ----------
    sequence : Union[str, SequenceRecord]
        Input sequence
    sigma : float
        Gaussian smoothing width
    normalize : bool
        Whether to z-score normalize

    Returns
    -------
    np.ndarray
        Processed interaction map
    """
    config = InterMapConfig(
        smooth=True,
        sigma=sigma,
        normalize=normalize,
        normalize_method="zscore"
    )
    return generate_finches_intermap(sequence, config=config)


if __name__ == "__main__":
    # Demo
    from .sequences import VARIANTS

    print("Generating interaction maps for all variants...")
    config = InterMapConfig(smooth=True, sigma=2.0, normalize=True)
    intermaps = compute_all_intermaps(config=config)

    for name, imap in intermaps.items():
        stats = compute_map_statistics(imap)
        print(f"\n{name}:")
        print(f"  Shape: {imap.shape}")
        print(f"  Mean: {stats['mean']:.3f}")
        print(f"  Std: {stats['std']:.3f}")
        print(f"  Fraction attractive: {stats['fraction_attractive']:.2%}")
