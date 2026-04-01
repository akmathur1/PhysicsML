"""
segmentation.py - Sticker-linker segmentation for IDP sequences.

This module implements the sticker-linker model for intrinsically disordered
proteins (IDPs). In this framework:

- STICKERS: Residues with strong attractive interactions (aromatics, R/K for
  cation-π) that drive phase separation
- LINKERS: Residues between stickers that modulate chain flexibility and
  effective concentration

The segmentation is based on local interaction energetics from FINCHES-style
analysis, using sliding window smoothing and energetic thresholds.
"""

from __future__ import annotations
import numpy as np
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

from .sequences import SequenceRecord, AROMATIC_RESIDUES, CATIONIC_RESIDUES


# =============================================================================
# SLIDING WINDOW ANALYSIS
# =============================================================================

def sliding_window_mean(
    signal: np.ndarray,
    window_size: int = 5,
    mode: str = "reflect"
) -> np.ndarray:
    """
    Compute sliding window mean of a 1D signal.

    Parameters
    ----------
    signal : np.ndarray
        1D input signal
    window_size : int
        Window size (should be odd for symmetric smoothing)
    mode : str
        Edge handling mode

    Returns
    -------
    np.ndarray
        Smoothed signal
    """
    return uniform_filter1d(signal, size=window_size, mode=mode)


def sliding_window_gaussian(
    signal: np.ndarray,
    sigma: float = 2.0,
    mode: str = "reflect"
) -> np.ndarray:
    """
    Apply Gaussian smoothing to a 1D signal.

    Parameters
    ----------
    signal : np.ndarray
        1D input signal
    sigma : float
        Gaussian standard deviation
    mode : str
        Edge handling mode

    Returns
    -------
    np.ndarray
        Smoothed signal
    """
    return gaussian_filter1d(signal, sigma=sigma, mode=mode)


def compute_interaction_profile(
    intermap: np.ndarray,
    axis: int = 1
) -> np.ndarray:
    """
    Compute the mean interaction profile from an interaction map.

    For each residue i, computes the mean interaction energy with all
    other residues j.

    Parameters
    ----------
    intermap : np.ndarray
        NxN interaction map
    axis : int
        Axis to average over (1 = row means, 0 = column means)

    Returns
    -------
    np.ndarray
        1D interaction profile (length N)
    """
    return np.mean(intermap, axis=axis)


def compute_smoothed_interaction_profile(
    intermap: np.ndarray,
    window_size: int = 5,
    method: str = "gaussian"
) -> np.ndarray:
    """
    Compute smoothed interaction profile.

    Parameters
    ----------
    intermap : np.ndarray
        NxN interaction map
    window_size : int
        Smoothing window (or sigma for gaussian)
    method : str
        "mean" for uniform, "gaussian" for Gaussian smoothing

    Returns
    -------
    np.ndarray
        Smoothed 1D profile
    """
    raw_profile = compute_interaction_profile(intermap)

    if method == "mean":
        return sliding_window_mean(raw_profile, window_size)
    elif method == "gaussian":
        return sliding_window_gaussian(raw_profile, sigma=window_size)
    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# STICKER IDENTIFICATION
# =============================================================================

@dataclass
class StickerThresholds:
    """Thresholds for sticker identification."""
    # Energy-based threshold (more negative = stickier)
    energy_threshold: float = -0.3  # kT units

    # Percentile-based threshold
    percentile_threshold: float = 25.0  # Bottom 25% are stickers

    # Minimum sticker region length
    min_sticker_length: int = 1

    # Maximum linker length before forced sticker
    max_linker_length: int = 20


def identify_stickers_by_energy(
    interaction_profile: np.ndarray,
    threshold: float = -0.3
) -> np.ndarray:
    """
    Identify sticker positions by energy threshold.

    Stickers are residues with interaction energy below threshold.

    Parameters
    ----------
    interaction_profile : np.ndarray
        1D mean interaction profile
    threshold : float
        Energy threshold (negative = attractive)

    Returns
    -------
    np.ndarray
        Boolean mask (True = sticker)
    """
    return interaction_profile < threshold


def identify_stickers_by_percentile(
    interaction_profile: np.ndarray,
    percentile: float = 25.0
) -> np.ndarray:
    """
    Identify stickers by percentile threshold.

    Bottom N% by interaction energy are classified as stickers.

    Parameters
    ----------
    interaction_profile : np.ndarray
        1D mean interaction profile
    percentile : float
        Percentile threshold (0-100)

    Returns
    -------
    np.ndarray
        Boolean mask (True = sticker)
    """
    threshold = np.percentile(interaction_profile, percentile)
    return interaction_profile <= threshold


def identify_stickers_by_sequence(
    sequence: Union[str, SequenceRecord],
    include_aromatics: bool = True,
    include_cations: bool = True
) -> np.ndarray:
    """
    Identify stickers directly from sequence (chemistry-based).

    Parameters
    ----------
    sequence : Union[str, SequenceRecord]
        Input sequence
    include_aromatics : bool
        Include Y, F, W as stickers
    include_cations : bool
        Include R, K as stickers (for cation-π)

    Returns
    -------
    np.ndarray
        Boolean mask (True = sticker)
    """
    if isinstance(sequence, SequenceRecord):
        seq = sequence.sequence
    else:
        seq = sequence

    n = len(seq)
    mask = np.zeros(n, dtype=bool)

    sticker_residues = set()
    if include_aromatics:
        sticker_residues.update(AROMATIC_RESIDUES)
    if include_cations:
        sticker_residues.update(CATIONIC_RESIDUES)

    for i, aa in enumerate(seq):
        if aa in sticker_residues:
            mask[i] = True

    return mask


def identify_stickers_hybrid(
    interaction_profile: np.ndarray,
    sequence: Union[str, SequenceRecord],
    energy_threshold: float = -0.2,
    require_chemistry: bool = True
) -> np.ndarray:
    """
    Hybrid sticker identification combining energy and chemistry.

    A residue is a sticker if:
    1. Its interaction energy is below threshold AND
    2. (optionally) It is a known sticker-type residue (aromatic/cation)

    Parameters
    ----------
    interaction_profile : np.ndarray
        1D mean interaction profile
    sequence : Union[str, SequenceRecord]
        Input sequence
    energy_threshold : float
        Energy threshold
    require_chemistry : bool
        If True, require residue to be aromatic or cationic

    Returns
    -------
    np.ndarray
        Boolean mask (True = sticker)
    """
    energy_mask = identify_stickers_by_energy(interaction_profile, energy_threshold)

    if require_chemistry:
        chemistry_mask = identify_stickers_by_sequence(sequence)
        return energy_mask & chemistry_mask
    else:
        return energy_mask


# =============================================================================
# STICKER MASK EXTRACTION
# =============================================================================

@dataclass
class StickerMask:
    """Container for sticker mask and derived properties."""
    mask: np.ndarray  # Boolean mask
    positions: np.ndarray  # 0-indexed sticker positions
    n_stickers: int
    n_linkers: int
    sticker_fraction: float
    regions: List[Tuple[int, int]]  # (start, end) of each sticker region


def create_sticker_mask(
    sticker_bool: np.ndarray,
    min_length: int = 1
) -> StickerMask:
    """
    Create a StickerMask object from boolean array.

    Parameters
    ----------
    sticker_bool : np.ndarray
        Boolean mask (True = sticker)
    min_length : int
        Minimum contiguous sticker region length

    Returns
    -------
    StickerMask
        Sticker mask with properties
    """
    # Filter short regions if needed
    if min_length > 1:
        sticker_bool = filter_short_regions(sticker_bool, min_length)

    n = len(sticker_bool)
    positions = np.where(sticker_bool)[0]
    n_stickers = int(np.sum(sticker_bool))
    n_linkers = n - n_stickers

    # Find contiguous regions
    regions = find_contiguous_regions(sticker_bool)

    return StickerMask(
        mask=sticker_bool,
        positions=positions,
        n_stickers=n_stickers,
        n_linkers=n_linkers,
        sticker_fraction=n_stickers / n if n > 0 else 0.0,
        regions=regions
    )


def filter_short_regions(
    mask: np.ndarray,
    min_length: int
) -> np.ndarray:
    """
    Remove contiguous True regions shorter than min_length.

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask
    min_length : int
        Minimum region length to keep

    Returns
    -------
    np.ndarray
        Filtered mask
    """
    result = mask.copy()
    regions = find_contiguous_regions(mask)

    for start, end in regions:
        if end - start < min_length:
            result[start:end] = False

    return result


def find_contiguous_regions(mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    Find contiguous True regions in a boolean mask.

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask

    Returns
    -------
    List[Tuple[int, int]]
        List of (start, end) indices for each region
    """
    regions = []
    in_region = False
    start = 0

    for i, val in enumerate(mask):
        if val and not in_region:
            # Start of new region
            start = i
            in_region = True
        elif not val and in_region:
            # End of region
            regions.append((start, i))
            in_region = False

    # Handle region extending to end
    if in_region:
        regions.append((start, len(mask)))

    return regions


# =============================================================================
# LINKER ANALYSIS
# =============================================================================

def compute_linker_lengths(sticker_mask: StickerMask) -> np.ndarray:
    """
    Compute the lengths of linker regions between stickers.

    Parameters
    ----------
    sticker_mask : StickerMask
        Sticker mask object

    Returns
    -------
    np.ndarray
        Array of linker lengths
    """
    positions = sticker_mask.positions
    if len(positions) < 2:
        return np.array([])

    # Gaps between consecutive stickers
    gaps = np.diff(positions) - 1  # -1 because diff gives spacing

    # Only count positive gaps (adjacent stickers have gap 0)
    return gaps[gaps > 0]


def compute_linker_statistics(sticker_mask: StickerMask) -> Dict[str, float]:
    """
    Compute statistics on linker lengths.

    Parameters
    ----------
    sticker_mask : StickerMask
        Sticker mask object

    Returns
    -------
    Dict[str, float]
        Dictionary of linker statistics
    """
    linker_lengths = compute_linker_lengths(sticker_mask)

    if len(linker_lengths) == 0:
        return {
            "n_linkers": 0,
            "mean_length": 0.0,
            "std_length": 0.0,
            "min_length": 0.0,
            "max_length": 0.0,
            "total_linker_residues": int(sticker_mask.n_linkers),
        }

    return {
        "n_linkers": len(linker_lengths),
        "mean_length": float(np.mean(linker_lengths)),
        "std_length": float(np.std(linker_lengths)),
        "min_length": float(np.min(linker_lengths)),
        "max_length": float(np.max(linker_lengths)),
        "total_linker_residues": int(sticker_mask.n_linkers),
    }


# =============================================================================
# 2D STICKER MASK (FOR INTERACTION MAPS)
# =============================================================================

def create_2d_sticker_mask(
    sticker_1d: np.ndarray
) -> np.ndarray:
    """
    Create 2D sticker-sticker interaction mask.

    I[i,j] = True if both i and j are stickers.

    Parameters
    ----------
    sticker_1d : np.ndarray
        1D boolean sticker mask

    Returns
    -------
    np.ndarray
        NxN boolean mask (True = sticker-sticker pair)
    """
    return np.outer(sticker_1d, sticker_1d)


def create_2d_linker_mask(sticker_1d: np.ndarray) -> np.ndarray:
    """Create mask for linker-linker interactions."""
    linker_1d = ~sticker_1d
    return np.outer(linker_1d, linker_1d)


def create_2d_sticker_linker_mask(sticker_1d: np.ndarray) -> np.ndarray:
    """Create mask for sticker-linker (cross) interactions."""
    linker_1d = ~sticker_1d
    # Cross terms: (sticker x linker) OR (linker x sticker)
    return np.outer(sticker_1d, linker_1d) | np.outer(linker_1d, sticker_1d)


def partition_interactions(
    intermap: np.ndarray,
    sticker_1d: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Partition interaction map by sticker/linker identity.

    Parameters
    ----------
    intermap : np.ndarray
        NxN interaction map
    sticker_1d : np.ndarray
        1D sticker mask

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with "sticker_sticker", "linker_linker", "cross" arrays
    """
    ss_mask = create_2d_sticker_mask(sticker_1d)
    ll_mask = create_2d_linker_mask(sticker_1d)
    sl_mask = create_2d_sticker_linker_mask(sticker_1d)

    return {
        "sticker_sticker": intermap[ss_mask],
        "linker_linker": intermap[ll_mask],
        "cross": intermap[sl_mask],
    }


def compute_partitioned_energies(
    intermap: np.ndarray,
    sticker_1d: np.ndarray
) -> Dict[str, float]:
    """
    Compute mean energies for each partition.

    Parameters
    ----------
    intermap : np.ndarray
        NxN interaction map
    sticker_1d : np.ndarray
        1D sticker mask

    Returns
    -------
    Dict[str, float]
        Mean energies for sticker-sticker, linker-linker, and cross
    """
    parts = partition_interactions(intermap, sticker_1d)

    return {
        "E_sticker_sticker": float(np.mean(parts["sticker_sticker"])) if len(parts["sticker_sticker"]) > 0 else 0.0,
        "E_linker_linker": float(np.mean(parts["linker_linker"])) if len(parts["linker_linker"]) > 0 else 0.0,
        "E_cross": float(np.mean(parts["cross"])) if len(parts["cross"]) > 0 else 0.0,
    }


# =============================================================================
# STICKER-LINKER CHAIN MODEL
# =============================================================================

@dataclass
class StickerLinkerSegment:
    """A single segment in the sticker-linker chain."""
    segment_type: str  # "sticker" or "linker"
    start: int  # 0-indexed start position
    end: int    # 0-indexed end position (exclusive)
    sequence: str  # Amino acid sequence of segment

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class StickerLinkerChain:
    """Complete sticker-linker chain representation."""
    segments: List[StickerLinkerSegment]
    full_sequence: str
    n_stickers: int
    n_linkers: int
    sticker_positions: List[int]

    @property
    def total_length(self) -> int:
        return len(self.full_sequence)

    @property
    def n_segments(self) -> int:
        return len(self.segments)

    def get_sticker_sequences(self) -> List[str]:
        """Get sequences of all sticker segments."""
        return [s.sequence for s in self.segments if s.segment_type == "sticker"]

    def get_linker_sequences(self) -> List[str]:
        """Get sequences of all linker segments."""
        return [s.sequence for s in self.segments if s.segment_type == "linker"]

    def to_string_representation(self) -> str:
        """Create string representation: S-L-S-L-S..."""
        parts = []
        for seg in self.segments:
            if seg.segment_type == "sticker":
                parts.append(f"S({seg.length})")
            else:
                parts.append(f"L({seg.length})")
        return "-".join(parts)


def build_sticker_linker_chain(
    sequence: Union[str, SequenceRecord],
    sticker_mask: Union[np.ndarray, StickerMask]
) -> StickerLinkerChain:
    """
    Build a sticker-linker chain from sequence and mask.

    Parameters
    ----------
    sequence : Union[str, SequenceRecord]
        Input sequence
    sticker_mask : Union[np.ndarray, StickerMask]
        Sticker mask (boolean array or StickerMask object)

    Returns
    -------
    StickerLinkerChain
        Chain representation
    """
    if isinstance(sequence, SequenceRecord):
        seq = sequence.sequence
    else:
        seq = sequence

    if isinstance(sticker_mask, StickerMask):
        mask = sticker_mask.mask
    else:
        mask = sticker_mask

    # Find all contiguous regions (both sticker and linker)
    segments = []
    current_type = "sticker" if mask[0] else "linker"
    start = 0

    for i in range(1, len(mask)):
        is_sticker = mask[i]
        new_type = "sticker" if is_sticker else "linker"

        if new_type != current_type:
            # End current segment, start new one
            segments.append(StickerLinkerSegment(
                segment_type=current_type,
                start=start,
                end=i,
                sequence=seq[start:i]
            ))
            start = i
            current_type = new_type

    # Add final segment
    segments.append(StickerLinkerSegment(
        segment_type=current_type,
        start=start,
        end=len(mask),
        sequence=seq[start:]
    ))

    n_stickers = sum(1 for s in segments if s.segment_type == "sticker")
    n_linkers = sum(1 for s in segments if s.segment_type == "linker")
    sticker_positions = list(np.where(mask)[0])

    return StickerLinkerChain(
        segments=segments,
        full_sequence=seq,
        n_stickers=n_stickers,
        n_linkers=n_linkers,
        sticker_positions=sticker_positions
    )


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def segment_all_variants(
    interaction_profiles: Dict[str, np.ndarray],
    sequences: Dict[str, SequenceRecord],
    threshold: float = -0.3
) -> Dict[str, StickerMask]:
    """
    Compute sticker masks for all variants.

    Parameters
    ----------
    interaction_profiles : Dict[str, np.ndarray]
        1D interaction profiles by variant name
    sequences : Dict[str, SequenceRecord]
        Sequence records by variant name
    threshold : float
        Energy threshold for stickers

    Returns
    -------
    Dict[str, StickerMask]
        Sticker masks by variant name
    """
    masks = {}
    for name in interaction_profiles:
        profile = interaction_profiles[name]
        sticker_bool = identify_stickers_by_energy(profile, threshold)
        masks[name] = create_sticker_mask(sticker_bool)
    return masks


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def mask_to_color_array(
    sticker_mask: np.ndarray,
    sticker_color: float = 1.0,
    linker_color: float = 0.0
) -> np.ndarray:
    """
    Convert boolean mask to numeric array for visualization.

    Parameters
    ----------
    sticker_mask : np.ndarray
        Boolean sticker mask
    sticker_color : float
        Value for sticker positions
    linker_color : float
        Value for linker positions

    Returns
    -------
    np.ndarray
        Numeric array
    """
    result = np.full(len(sticker_mask), linker_color)
    result[sticker_mask] = sticker_color
    return result


if __name__ == "__main__":
    # Demo
    from .sequences import VARIANTS
    from .intermaps import compute_all_intermaps, InterMapConfig

    print("Sticker-Linker Segmentation Demo")
    print("=" * 60)

    # Generate interaction maps
    config = InterMapConfig(smooth=True, sigma=2.0, normalize=False)
    intermaps = compute_all_intermaps(config=config)

    for name in ["WT", "AllY_to_S"]:
        seq = VARIANTS[name]
        imap = intermaps[name]
        profile = compute_interaction_profile(imap)

        # Identify stickers
        sticker_bool = identify_stickers_hybrid(profile, seq)
        sticker_mask = create_sticker_mask(sticker_bool)

        print(f"\n{name}:")
        print(f"  Stickers: {sticker_mask.n_stickers} ({sticker_mask.sticker_fraction:.1%})")
        print(f"  Linkers: {sticker_mask.n_linkers}")

        # Build chain
        chain = build_sticker_linker_chain(seq, sticker_mask)
        print(f"  Chain: {chain.to_string_representation()[:80]}...")
