"""
metrics.py - Biophysical metrics for IDP phase separation analysis.

This module computes quantitative metrics that correlate with phase separation
behavior, based on the sticker-linker model and FINCHES-style analysis.

Key metrics:
- Sticker fraction (f_sticker): Fraction of sequence that are stickers
- ΔG proxy: Effective interaction energy from intermaps
- κ (kappa): Linker length distribution metric
- SCD: Sequence charge decoration
- Ω (omega): Mixing parameter from sticker spacing

These metrics are inspired by:
- Pappu lab sequence analysis tools (localCIDER)
- FINCHES/MPIPI coarse-grained models
- Sticker-linker theory (Choi, Holehouse, Pappu)
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass

from .sequences import (
    SequenceRecord,
    AROMATIC_RESIDUES,
    CATIONIC_RESIDUES,
    ANIONIC_RESIDUES,
    compute_sequence_properties
)
from .segmentation import (
    StickerMask,
    compute_linker_lengths,
    compute_partitioned_energies
)
from .topology import TopologyMetrics, compute_topology_metrics
from .homology import HomologyMetrics, compute_homology_metrics
from .entropy import EntropyMetrics, compute_entropy_metrics


# =============================================================================
# STICKER-LINKER METRICS
# =============================================================================

def compute_sticker_fraction(sticker_mask: Union[np.ndarray, StickerMask]) -> float:
    """
    Compute the fraction of residues that are stickers.

    Higher sticker fraction generally correlates with stronger
    phase separation tendency.

    Parameters
    ----------
    sticker_mask : Union[np.ndarray, StickerMask]
        Boolean mask or StickerMask object

    Returns
    -------
    float
        Sticker fraction (0 to 1)
    """
    if isinstance(sticker_mask, StickerMask):
        return sticker_mask.sticker_fraction
    else:
        return float(np.mean(sticker_mask))


def compute_linker_length_kappa(sticker_mask: Union[np.ndarray, StickerMask]) -> float:
    """
    Compute κ (kappa), a metric for linker length distribution.

    κ = <L> / σ_L where <L> is mean linker length and σ_L is std dev.

    Higher κ indicates more uniform linker lengths, which can affect
    the effective valency of the sticker-linker chain.

    Parameters
    ----------
    sticker_mask : Union[np.ndarray, StickerMask]
        Sticker mask

    Returns
    -------
    float
        Kappa value (NaN if insufficient linkers)
    """
    if isinstance(sticker_mask, StickerMask):
        linker_lengths = compute_linker_lengths(sticker_mask)
    else:
        from .segmentation import create_sticker_mask
        mask_obj = create_sticker_mask(sticker_mask)
        linker_lengths = compute_linker_lengths(mask_obj)

    if len(linker_lengths) < 2:
        return np.nan

    mean_L = np.mean(linker_lengths)
    std_L = np.std(linker_lengths)

    if std_L < 1e-10:
        return np.inf  # Perfectly uniform

    return mean_L / std_L


def compute_mean_linker_length(sticker_mask: Union[np.ndarray, StickerMask]) -> float:
    """
    Compute mean linker length.

    Parameters
    ----------
    sticker_mask : Union[np.ndarray, StickerMask]
        Sticker mask

    Returns
    -------
    float
        Mean linker length
    """
    if isinstance(sticker_mask, StickerMask):
        linker_lengths = compute_linker_lengths(sticker_mask)
    else:
        from .segmentation import create_sticker_mask
        mask_obj = create_sticker_mask(sticker_mask)
        linker_lengths = compute_linker_lengths(mask_obj)

    if len(linker_lengths) == 0:
        return 0.0

    return float(np.mean(linker_lengths))


# =============================================================================
# ENERGY-BASED METRICS (ΔG PROXY)
# =============================================================================

def compute_total_interaction_energy(intermap: np.ndarray) -> float:
    """
    Compute total interaction energy from intermap.

    This is a proxy for the enthalpic driving force of phase separation.
    More negative = stronger interactions = more likely to phase separate.

    Parameters
    ----------
    intermap : np.ndarray
        NxN interaction map

    Returns
    -------
    float
        Total interaction energy (sum of all pairwise interactions)
    """
    # Use upper triangle to avoid double counting (excluding diagonal)
    n = intermap.shape[0]
    upper_tri = np.triu_indices(n, k=1)
    return float(np.sum(intermap[upper_tri]))


def compute_mean_interaction_energy(intermap: np.ndarray) -> float:
    """
    Compute mean pairwise interaction energy.

    Parameters
    ----------
    intermap : np.ndarray
        NxN interaction map

    Returns
    -------
    float
        Mean interaction energy per residue pair
    """
    n = intermap.shape[0]
    upper_tri = np.triu_indices(n, k=1)
    n_pairs = len(upper_tri[0])
    if n_pairs == 0:
        return 0.0
    return float(np.sum(intermap[upper_tri]) / n_pairs)


def compute_delta_G_proxy(
    intermap: np.ndarray,
    sticker_mask: Optional[np.ndarray] = None,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Compute ΔG proxy from interaction map.

    This metric weights sticker-sticker interactions more heavily,
    reflecting their importance in driving phase separation.

    Parameters
    ----------
    intermap : np.ndarray
        NxN interaction map
    sticker_mask : Optional[np.ndarray]
        1D sticker mask for weighted calculation
    weights : Optional[Dict[str, float]]
        Weights for "sticker_sticker", "linker_linker", "cross"

    Returns
    -------
    float
        ΔG proxy value
    """
    if sticker_mask is None:
        # Simple mean
        return compute_mean_interaction_energy(intermap)

    if weights is None:
        weights = {
            "sticker_sticker": 2.0,  # Sticker interactions weighted 2x
            "linker_linker": 0.5,
            "cross": 1.0,
        }

    energies = compute_partitioned_energies(intermap, sticker_mask)

    delta_G = (
        weights["sticker_sticker"] * energies["E_sticker_sticker"] +
        weights["linker_linker"] * energies["E_linker_linker"] +
        weights["cross"] * energies["E_cross"]
    )

    return delta_G


def compute_sticker_interaction_strength(
    intermap: np.ndarray,
    sticker_mask: np.ndarray
) -> float:
    """
    Compute mean sticker-sticker interaction strength.

    Parameters
    ----------
    intermap : np.ndarray
        NxN interaction map
    sticker_mask : np.ndarray
        1D sticker mask

    Returns
    -------
    float
        Mean sticker-sticker interaction energy
    """
    ss_mask = np.outer(sticker_mask, sticker_mask)
    ss_values = intermap[ss_mask]

    if len(ss_values) == 0:
        return 0.0

    return float(np.mean(ss_values))


# =============================================================================
# SEQUENCE-BASED METRICS
# =============================================================================

def compute_FCR(sequence: Union[str, SequenceRecord]) -> float:
    """
    Compute Fraction of Charged Residues (FCR).

    FCR = (n_positive + n_negative) / N

    Parameters
    ----------
    sequence : Union[str, SequenceRecord]
        Input sequence

    Returns
    -------
    float
        FCR value (0 to 1)
    """
    if isinstance(sequence, SequenceRecord):
        seq = sequence.sequence
    else:
        seq = sequence

    n = len(seq)
    n_charged = sum(1 for aa in seq if aa in CATIONIC_RESIDUES | ANIONIC_RESIDUES)

    return n_charged / n


def compute_NCPR(sequence: Union[str, SequenceRecord]) -> float:
    """
    Compute Net Charge Per Residue (NCPR).

    NCPR = (n_positive - n_negative) / N

    Parameters
    ----------
    sequence : Union[str, SequenceRecord]
        Input sequence

    Returns
    -------
    float
        NCPR value (-1 to 1)
    """
    if isinstance(sequence, SequenceRecord):
        seq = sequence.sequence
    else:
        seq = sequence

    n = len(seq)
    n_pos = sum(1 for aa in seq if aa in CATIONIC_RESIDUES)
    n_neg = sum(1 for aa in seq if aa in ANIONIC_RESIDUES)

    return (n_pos - n_neg) / n


def compute_hydropathy(sequence: Union[str, SequenceRecord]) -> float:
    """
    Compute mean hydropathy (Kyte-Doolittle scale).

    Parameters
    ----------
    sequence : Union[str, SequenceRecord]
        Input sequence

    Returns
    -------
    float
        Mean hydropathy
    """
    if isinstance(sequence, SequenceRecord):
        seq = sequence.sequence
    else:
        seq = sequence

    # Kyte-Doolittle scale
    KD = {
        "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
        "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
        "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
        "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
    }

    return np.mean([KD.get(aa, 0) for aa in seq])


def compute_aromatic_fraction(sequence: Union[str, SequenceRecord]) -> float:
    """
    Compute fraction of aromatic residues (Y, F, W).

    Parameters
    ----------
    sequence : Union[str, SequenceRecord]
        Input sequence

    Returns
    -------
    float
        Aromatic fraction
    """
    if isinstance(sequence, SequenceRecord):
        seq = sequence.sequence
    else:
        seq = sequence

    n_aromatic = sum(1 for aa in seq if aa in AROMATIC_RESIDUES)
    return n_aromatic / len(seq)


def compute_tyrosine_fraction(sequence: Union[str, SequenceRecord]) -> float:
    """
    Compute fraction of tyrosine residues.

    Parameters
    ----------
    sequence : Union[str, SequenceRecord]
        Input sequence

    Returns
    -------
    float
        Tyrosine fraction
    """
    if isinstance(sequence, SequenceRecord):
        seq = sequence.sequence
    else:
        seq = sequence

    return seq.count("Y") / len(seq)


# =============================================================================
# SEQUENCE CHARGE DECORATION (SCD)
# =============================================================================

def compute_SCD(sequence: Union[str, SequenceRecord]) -> float:
    """
    Compute Sequence Charge Decoration (SCD).

    SCD quantifies the distribution/patterning of charged residues.
    Higher |SCD| indicates more clustered charge distribution.

    SCD = (1/N) * Σ_i Σ_j>i q_i * q_j * sqrt(|j-i|)

    where q_i is the charge of residue i.

    Parameters
    ----------
    sequence : Union[str, SequenceRecord]
        Input sequence

    Returns
    -------
    float
        SCD value
    """
    if isinstance(sequence, SequenceRecord):
        seq = sequence.sequence
    else:
        seq = sequence

    n = len(seq)

    # Assign charges
    charges = []
    for aa in seq:
        if aa in CATIONIC_RESIDUES:
            charges.append(1.0)
        elif aa in ANIONIC_RESIDUES:
            charges.append(-1.0)
        else:
            charges.append(0.0)
    charges = np.array(charges)

    # Compute SCD
    scd = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            scd += charges[i] * charges[j] * np.sqrt(j - i)

    return scd / n


def compute_kappa_charge(sequence: Union[str, SequenceRecord]) -> float:
    """
    Compute charge patterning parameter κ (Das-Pappu).

    κ measures charge segregation:
    - κ ≈ 0: Well-mixed charges
    - κ ≈ 1: Highly segregated (blocky) charges

    Parameters
    ----------
    sequence : Union[str, SequenceRecord]
        Input sequence

    Returns
    -------
    float
        κ value (0 to 1)
    """
    if isinstance(sequence, SequenceRecord):
        seq = sequence.sequence
    else:
        seq = sequence

    n = len(seq)

    # Get charge pattern
    charges = []
    for aa in seq:
        if aa in CATIONIC_RESIDUES:
            charges.append(1)
        elif aa in ANIONIC_RESIDUES:
            charges.append(-1)
        else:
            charges.append(0)

    # Compute δ (local charge asymmetry) in windows
    window_size = 5
    if n < window_size:
        return 0.0

    deltas = []
    for i in range(n - window_size + 1):
        window = charges[i:i + window_size]
        f_plus = sum(1 for c in window if c > 0) / window_size
        f_minus = sum(1 for c in window if c < 0) / window_size
        delta = (f_plus - f_minus) ** 2
        deltas.append(delta)

    # κ = <δ²> / δ_max²
    # For FUS LCD, this is typically small (well-mixed)
    fcr = compute_FCR(seq)
    if fcr < 1e-10:
        return 0.0

    # Simplified: normalize by variance
    delta_mean = np.mean(deltas)
    return min(1.0, delta_mean / (fcr ** 2 + 1e-10))


# =============================================================================
# STICKER SPACING METRICS (OMEGA)
# =============================================================================

def compute_omega(sticker_mask: Union[np.ndarray, StickerMask]) -> float:
    """
    Compute Ω (omega), a sticker spacing uniformity metric.

    Ω quantifies how uniformly spaced stickers are along the sequence.
    - Ω ≈ 0: Highly uniform spacing (ideal for multivalent interactions)
    - Ω ≈ 1: Highly non-uniform (clustered stickers)

    Parameters
    ----------
    sticker_mask : Union[np.ndarray, StickerMask]
        Sticker mask

    Returns
    -------
    float
        Ω value (0 to 1+)
    """
    if isinstance(sticker_mask, StickerMask):
        positions = sticker_mask.positions
    else:
        positions = np.where(sticker_mask)[0]

    if len(positions) < 2:
        return np.nan

    # Compute spacings
    spacings = np.diff(positions)

    if len(spacings) < 2:
        return 0.0

    # Coefficient of variation
    mean_spacing = np.mean(spacings)
    std_spacing = np.std(spacings)

    if mean_spacing < 1e-10:
        return 0.0

    return std_spacing / mean_spacing


def compute_sticker_clustering(
    sticker_mask: Union[np.ndarray, StickerMask],
    distance_threshold: int = 3
) -> float:
    """
    Compute sticker clustering metric.

    Fraction of stickers that have another sticker within distance_threshold.

    Parameters
    ----------
    sticker_mask : Union[np.ndarray, StickerMask]
        Sticker mask
    distance_threshold : int
        Maximum distance for clustering

    Returns
    -------
    float
        Clustering fraction (0 to 1)
    """
    if isinstance(sticker_mask, StickerMask):
        positions = sticker_mask.positions
    else:
        positions = np.where(sticker_mask)[0]

    if len(positions) < 2:
        return 0.0

    n_clustered = 0
    for i, pos in enumerate(positions):
        # Check neighbors
        has_neighbor = False
        for j, other_pos in enumerate(positions):
            if i != j and abs(pos - other_pos) <= distance_threshold:
                has_neighbor = True
                break
        if has_neighbor:
            n_clustered += 1

    return n_clustered / len(positions)


# =============================================================================
# COMPREHENSIVE METRICS DATACLASS
# =============================================================================

@dataclass
class VariantMetrics:
    """Complete metrics for a sequence variant."""
    name: str

    # Sequence properties
    length: int
    n_tyrosine: int
    n_aromatic: int
    tyrosine_fraction: float
    aromatic_fraction: float

    # Charge properties
    FCR: float
    NCPR: float
    SCD: float
    kappa_charge: float

    # Sticker-linker properties
    sticker_fraction: float
    n_stickers: int
    mean_linker_length: float
    kappa_linker: float
    omega: float

    # Energy properties
    total_energy: float
    mean_energy: float
    delta_G_proxy: float
    sticker_energy: float

    # Topology properties (Phase 1)
    topology: Optional[TopologyMetrics] = None

    # Homology properties (Phase 1)
    homology: Optional[HomologyMetrics] = None

    # Entropy properties (Phase 1)
    entropy: Optional[EntropyMetrics] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        d = {
            "name": self.name,
            "length": self.length,
            "n_tyrosine": self.n_tyrosine,
            "n_aromatic": self.n_aromatic,
            "tyrosine_fraction": self.tyrosine_fraction,
            "aromatic_fraction": self.aromatic_fraction,
            "FCR": self.FCR,
            "NCPR": self.NCPR,
            "SCD": self.SCD,
            "kappa_charge": self.kappa_charge,
            "sticker_fraction": self.sticker_fraction,
            "n_stickers": self.n_stickers,
            "mean_linker_length": self.mean_linker_length,
            "kappa_linker": self.kappa_linker,
            "omega": self.omega,
            "total_energy": self.total_energy,
            "mean_energy": self.mean_energy,
            "delta_G_proxy": self.delta_G_proxy,
            "sticker_energy": self.sticker_energy,
        }
        if self.topology is not None:
            d["topology"] = self.topology.to_dict()
        if self.homology is not None:
            d["homology"] = self.homology.to_dict()
        if self.entropy is not None:
            d["entropy"] = self.entropy.to_dict()
        return d


def compute_all_metrics(
    name: str,
    sequence: Union[str, SequenceRecord],
    intermap: np.ndarray,
    sticker_mask: Union[np.ndarray, StickerMask],
    include_topology: bool = True,
) -> VariantMetrics:
    """
    Compute all metrics for a variant.

    Parameters
    ----------
    name : str
        Variant name
    sequence : Union[str, SequenceRecord]
        Sequence
    intermap : np.ndarray
        Interaction map
    sticker_mask : Union[np.ndarray, StickerMask]
        Sticker mask
    include_topology : bool
        If True, compute topology, homology, and entropy metrics (Phase 1)

    Returns
    -------
    VariantMetrics
        Complete metrics
    """
    if isinstance(sequence, SequenceRecord):
        seq = sequence.sequence
    else:
        seq = sequence

    if isinstance(sticker_mask, StickerMask):
        mask = sticker_mask.mask
        mask_obj = sticker_mask
    else:
        mask = sticker_mask
        from .segmentation import create_sticker_mask
        mask_obj = create_sticker_mask(sticker_mask)

    # Phase 1: topology engine metrics
    topo = None
    homo = None
    entr = None
    if include_topology:
        topo = compute_topology_metrics(intermap, mask_obj, sequence=seq)
        homo = compute_homology_metrics(intermap, mask_obj)
        entr = compute_entropy_metrics(intermap, mask_obj, seq)

    return VariantMetrics(
        name=name,
        # Sequence
        length=len(seq),
        n_tyrosine=seq.count("Y"),
        n_aromatic=sum(1 for aa in seq if aa in AROMATIC_RESIDUES),
        tyrosine_fraction=compute_tyrosine_fraction(seq),
        aromatic_fraction=compute_aromatic_fraction(seq),
        # Charge
        FCR=compute_FCR(seq),
        NCPR=compute_NCPR(seq),
        SCD=compute_SCD(seq),
        kappa_charge=compute_kappa_charge(seq),
        # Sticker-linker
        sticker_fraction=compute_sticker_fraction(mask_obj),
        n_stickers=mask_obj.n_stickers,
        mean_linker_length=compute_mean_linker_length(mask_obj),
        kappa_linker=compute_linker_length_kappa(mask_obj),
        omega=compute_omega(mask_obj),
        # Energy
        total_energy=compute_total_interaction_energy(intermap),
        mean_energy=compute_mean_interaction_energy(intermap),
        delta_G_proxy=compute_delta_G_proxy(intermap, mask),
        sticker_energy=compute_sticker_interaction_strength(intermap, mask),
        # Phase 1: Topology engine
        topology=topo,
        homology=homo,
        entropy=entr,
    )


def compute_metrics_table(
    variants: Dict[str, SequenceRecord],
    intermaps: Dict[str, np.ndarray],
    sticker_masks: Dict[str, StickerMask]
) -> Dict[str, VariantMetrics]:
    """
    Compute metrics for all variants.

    Returns
    -------
    Dict[str, VariantMetrics]
        Metrics by variant name
    """
    metrics = {}
    for name in variants:
        metrics[name] = compute_all_metrics(
            name,
            variants[name],
            intermaps[name],
            sticker_masks[name]
        )
    return metrics


def metrics_to_dataframe(metrics: Dict[str, VariantMetrics]):
    """
    Convert metrics dictionary to pandas DataFrame.

    Note: Requires pandas to be installed.
    """
    try:
        import pandas as pd
        rows = [m.to_dict() for m in metrics.values()]
        return pd.DataFrame(rows).set_index("name")
    except ImportError:
        raise ImportError("pandas required for DataFrame conversion")


if __name__ == "__main__":
    # Demo
    from .sequences import VARIANTS
    from .intermaps import compute_all_intermaps, InterMapConfig, compute_interaction_profile
    from .segmentation import identify_stickers_by_energy, create_sticker_mask

    print("Computing metrics for all variants...")
    print("=" * 60)

    config = InterMapConfig(smooth=True, sigma=2.0, normalize=False)
    intermaps = compute_all_intermaps(config=config)

    for name, record in VARIANTS.items():
        imap = intermaps[name]
        profile = compute_interaction_profile(imap)
        sticker_bool = identify_stickers_by_energy(profile, threshold=-0.2)
        sticker_mask = create_sticker_mask(sticker_bool)

        metrics = compute_all_metrics(name, record, imap, sticker_mask)

        print(f"\n{name}:")
        print(f"  Tyrosines: {metrics.n_tyrosine} ({metrics.tyrosine_fraction:.1%})")
        print(f"  Stickers: {metrics.n_stickers} ({metrics.sticker_fraction:.1%})")
        print(f"  Mean linker: {metrics.mean_linker_length:.1f} residues")
        print(f"  ΔG proxy: {metrics.delta_G_proxy:.3f}")
        print(f"  Ω (spacing): {metrics.omega:.3f}")
