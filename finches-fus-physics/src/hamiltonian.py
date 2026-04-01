"""
hamiltonian.py - Hybrid effective Hamiltonian combining chemistry and topology.

The central idea of Phase 2:

    H_eff = H_chemistry + H_topology

Where:
- H_chemistry captures pairwise residue-residue interaction energetics
  (from FINCHES/MPIPI-GG force field)
- H_topology captures emergent network properties of the sticker-sticker
  interaction graph that modulate the *effective* driving force for LLPS

This is the paper-level contribution: showing that sequence chemistry alone
is insufficient — the *topology* of the interaction network contributes
an independent, quantifiable term to the effective Hamiltonian.

Physical motivation:
- Two sequences with identical composition but different sticker arrangement
  have the same H_chemistry but different H_topology
- H_topology captures: network connectivity (valency), clustering (local
  robustness), percolation (global connectivity), and arrangement regularity
- The combined H_eff should better predict phase separation propensity
  than either term alone

Functional form:

    H_eff = E_chem + α·Φ_network + β·Φ_percolation + γ·Φ_arrangement

Where:
- E_chem = weighted partition energy (sticker-sticker, cross, linker-linker)
- Φ_network = f(clustering, degree, connectivity)
- Φ_percolation = f(percolation_threshold, giant_component_onset)
- Φ_arrangement = f(spacing_entropy, omega)
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

from .topology import TopologyMetrics
from .homology import HomologyMetrics
from .entropy import EntropyMetrics
from .segmentation import StickerMask, compute_partitioned_energies


# =============================================================================
# COUPLING CONSTANTS (default calibration)
# =============================================================================

@dataclass
class HamiltonianParams:
    """Coupling constants for the hybrid Hamiltonian.

    These weights control the relative contribution of each topology
    term to H_eff. Default values are calibrated so that each term
    contributes on the same order of magnitude as H_chemistry for
    the WT FUS LCD sequence.

    Physical reasoning for defaults:
    - alpha_clustering: clustering stabilizes multivalent networks
    - alpha_connectivity: fewer components = more connected = more stable
    - alpha_percolation: deeper threshold = stronger network
    - alpha_arrangement: more regular spacing = better multivalency
    - alpha_homology: persistent features indicate robust topology
    """
    # Chemistry partition weights (inherited from existing ΔG proxy)
    w_sticker_sticker: float = 2.0
    w_cross: float = 1.0
    w_linker_linker: float = 0.5

    # Topology coupling constants
    alpha_clustering: float = 0.05      # weight for clustering term
    alpha_connectivity: float = 0.02    # weight for connectivity term
    alpha_percolation: float = 1.0      # weight for percolation threshold
    alpha_arrangement: float = 0.02     # weight for arrangement regularity
    alpha_homology: float = 0.5         # weight for H0 persistence


# =============================================================================
# TOPOLOGY ENERGY TERMS
# =============================================================================

def compute_phi_network(
    topology: TopologyMetrics,
    n_stickers: int,
) -> Dict[str, float]:
    """
    Compute network topology energy contributions.

    Φ_network has two components:

    1. Clustering term: -C_sticker
       Higher clustering coefficient → more tightly interconnected sticker
       neighborhoods → more robust multivalent interactions → stabilizing.

    2. Connectivity term: (n_components - 1) / (n_stickers - 1)
       Normalized fragmentation. 0 = fully connected, 1 = fully fragmented.
       More fragments → weaker effective network → destabilizing.

    Parameters
    ----------
    topology : TopologyMetrics
        Topology analysis results
    n_stickers : int
        Number of sticker residues

    Returns
    -------
    Dict with "clustering" and "connectivity" energy terms
    """
    # Clustering: negative = stabilizing (more clustered = lower energy)
    clustering_term = -topology.sticker_clustering_coefficient

    # Connectivity: positive = destabilizing (more fragmented = higher energy)
    if n_stickers > 1:
        # Normalize: 0 (one component) to 1 (every sticker isolated)
        connectivity_term = (topology.sticker_n_components - 1) / (n_stickers - 1)
    else:
        connectivity_term = 0.0

    return {
        "clustering": clustering_term,
        "connectivity": connectivity_term,
    }


def compute_phi_percolation(
    topology: TopologyMetrics,
) -> float:
    """
    Compute percolation energy contribution.

    Φ_percolation = percolation_threshold

    The percolation threshold is already in energy units (kT).
    More negative threshold → network connects at stronger interaction
    strength → stronger driving force for LLPS → stabilizing.

    This is the most direct topology → LLPS mapping: percolation
    threshold ~ onset of connected network ~ onset of phase separation.

    Parameters
    ----------
    topology : TopologyMetrics
        Topology analysis results

    Returns
    -------
    float
        Percolation energy term (kT)
    """
    return topology.percolation_threshold


def compute_phi_arrangement(
    entropy_metrics: EntropyMetrics,
    omega: float,
) -> float:
    """
    Compute arrangement regularity energy contribution.

    Φ_arrangement = H_norm_spacing + Ω

    - H_norm_spacing: normalized spacing entropy (0 = perfectly regular,
      1 = maximally disordered). Higher → less regular → destabilizing.
    - Ω: coefficient of variation of sticker spacing (0 = uniform, >1 = irregular).

    Regular sticker spacing maximizes effective valency of the chain.
    Irregular spacing creates "dead zones" where multivalent contacts
    cannot form efficiently.

    Parameters
    ----------
    entropy_metrics : EntropyMetrics
        Entropy analysis results
    omega : float
        Sticker spacing omega (from metrics)

    Returns
    -------
    float
        Arrangement energy term (dimensionless, positive = destabilizing)
    """
    h_norm = entropy_metrics.normalized_spacing_entropy
    omega_safe = omega if np.isfinite(omega) else 1.0

    # Both terms: higher = more disordered = destabilizing
    return h_norm + omega_safe


def compute_phi_homology(
    homology: HomologyMetrics,
    n_stickers: int,
) -> float:
    """
    Compute persistent homology energy contribution.

    Φ_homology = -H0_total_persistence / n_stickers

    H0 total persistence measures how "spread out" the merge events are
    in the connectivity filtration. Higher total persistence means the
    sticker clusters are more distinct and take longer to merge →
    indicates a more structured, hierarchical network → stabilizing.

    Normalized by n_stickers so the term scales properly across variants
    with different numbers of stickers.

    Parameters
    ----------
    homology : HomologyMetrics
        Homology analysis results
    n_stickers : int
        Number of sticker residues

    Returns
    -------
    float
        Homology energy term (negative = stabilizing)
    """
    if n_stickers < 2:
        return 0.0

    return -homology.h0_total_persistence / n_stickers


# =============================================================================
# HYBRID HAMILTONIAN
# =============================================================================

@dataclass
class HamiltonianDecomposition:
    """Full decomposition of the effective Hamiltonian.

    Stores each energy term separately for analysis and interpretation.
    """
    # Chemistry terms
    E_sticker_sticker: float    # weighted sticker-sticker energy
    E_cross: float              # weighted cross-term energy
    E_linker_linker: float      # weighted linker-linker energy
    H_chemistry: float          # total chemistry Hamiltonian

    # Topology terms
    phi_clustering: float       # clustering energy contribution
    phi_connectivity: float     # connectivity energy contribution
    phi_percolation: float      # percolation threshold contribution
    phi_arrangement: float      # arrangement regularity contribution
    phi_homology: float         # persistent homology contribution
    H_topology: float           # total topology Hamiltonian

    # Combined
    H_eff: float                # H_chemistry + H_topology

    # Metadata
    variant_name: str = ""
    n_stickers: int = 0

    @property
    def topology_fraction(self) -> float:
        """Fraction of H_eff contributed by topology."""
        if abs(self.H_eff) < 1e-15:
            return 0.0
        return abs(self.H_topology) / (abs(self.H_chemistry) + abs(self.H_topology))

    @property
    def chemistry_fraction(self) -> float:
        """Fraction of H_eff contributed by chemistry."""
        return 1.0 - self.topology_fraction

    def to_dict(self) -> Dict[str, float]:
        return {
            "E_sticker_sticker": self.E_sticker_sticker,
            "E_cross": self.E_cross,
            "E_linker_linker": self.E_linker_linker,
            "H_chemistry": self.H_chemistry,
            "phi_clustering": self.phi_clustering,
            "phi_connectivity": self.phi_connectivity,
            "phi_percolation": self.phi_percolation,
            "phi_arrangement": self.phi_arrangement,
            "phi_homology": self.phi_homology,
            "H_topology": self.H_topology,
            "H_eff": self.H_eff,
            "topology_fraction": self.topology_fraction,
            "chemistry_fraction": self.chemistry_fraction,
            "variant_name": self.variant_name,
            "n_stickers": self.n_stickers,
        }


def compute_H_eff(
    intermap: np.ndarray,
    sticker_mask: Union[np.ndarray, StickerMask],
    topology: TopologyMetrics,
    homology: HomologyMetrics,
    entropy_metrics: EntropyMetrics,
    omega: float,
    params: Optional[HamiltonianParams] = None,
    variant_name: str = "",
) -> HamiltonianDecomposition:
    """
    Compute the full hybrid effective Hamiltonian.

    H_eff = H_chemistry + H_topology

    where:
        H_chemistry = w_ss·E_ss + w_sl·E_sl + w_ll·E_ll
        H_topology  = α_c·Φ_cluster + α_n·Φ_connect + α_p·Φ_perc
                     + α_a·Φ_arrange + α_h·Φ_homology

    Parameters
    ----------
    intermap : np.ndarray
        NxN interaction map
    sticker_mask : Union[np.ndarray, StickerMask]
        Sticker mask
    topology : TopologyMetrics
        Topology metrics
    homology : HomologyMetrics
        Homology metrics
    entropy_metrics : EntropyMetrics
        Entropy metrics
    omega : float
        Sticker spacing omega
    params : Optional[HamiltonianParams]
        Coupling constants (uses defaults if None)
    variant_name : str
        Name for labeling

    Returns
    -------
    HamiltonianDecomposition
        Full energy decomposition
    """
    if params is None:
        params = HamiltonianParams()

    if isinstance(sticker_mask, StickerMask):
        mask = sticker_mask.mask
        n_stickers = sticker_mask.n_stickers
    else:
        mask = sticker_mask
        n_stickers = int(np.sum(sticker_mask))

    # ─── Chemistry ───
    energies = compute_partitioned_energies(intermap, mask)

    E_ss = params.w_sticker_sticker * energies["E_sticker_sticker"]
    E_cross = params.w_cross * energies["E_cross"]
    E_ll = params.w_linker_linker * energies["E_linker_linker"]
    H_chem = E_ss + E_cross + E_ll

    # ─── Topology ───
    phi_net = compute_phi_network(topology, n_stickers)
    phi_clust = params.alpha_clustering * phi_net["clustering"]
    phi_conn = params.alpha_connectivity * phi_net["connectivity"]
    phi_perc = params.alpha_percolation * compute_phi_percolation(topology)
    phi_arr = params.alpha_arrangement * compute_phi_arrangement(entropy_metrics, omega)
    phi_hom = params.alpha_homology * compute_phi_homology(homology, n_stickers)

    H_topo = phi_clust + phi_conn + phi_perc + phi_arr + phi_hom

    return HamiltonianDecomposition(
        E_sticker_sticker=E_ss,
        E_cross=E_cross,
        E_linker_linker=E_ll,
        H_chemistry=H_chem,
        phi_clustering=phi_clust,
        phi_connectivity=phi_conn,
        phi_percolation=phi_perc,
        phi_arrangement=phi_arr,
        phi_homology=phi_hom,
        H_topology=H_topo,
        H_eff=H_chem + H_topo,
        variant_name=variant_name,
        n_stickers=n_stickers,
    )


# =============================================================================
# SATURATION CONCENTRATION PREDICTION
# =============================================================================

def predict_csat(
    H_eff: float,
    reference_csat: float = 1.0,
    reference_H: float = -0.075,
    kT: float = 1.0,
) -> float:
    """
    Predict saturation concentration from H_eff.

    Uses a Boltzmann-like relationship:

        c_sat / c_sat_ref = exp((H_eff - H_ref) / kT)

    More negative H_eff → lower c_sat → stronger phase separation.
    Less negative H_eff → higher c_sat → weaker phase separation.

    Parameters
    ----------
    H_eff : float
        Effective Hamiltonian value
    reference_csat : float
        Reference c_sat (arbitrary units, e.g., WT = 1.0)
    reference_H : float
        Reference H_eff value (typically WT)
    kT : float
        Thermal energy (= 1 if H is already in kT units)

    Returns
    -------
    float
        Predicted c_sat (relative to reference)
    """
    return reference_csat * np.exp((H_eff - reference_H) / kT)


# =============================================================================
# SENSITIVITY ANALYSIS
# =============================================================================

@dataclass
class SensitivityResult:
    """Results from Hamiltonian sensitivity analysis."""
    term_names: List[str]
    term_values: np.ndarray         # (n_variants, n_terms)
    variant_names: List[str]
    H_eff_values: np.ndarray        # (n_variants,)
    sensitivities: Dict[str, float] # normalized |contribution| per term


def compute_sensitivity(
    decompositions: Dict[str, HamiltonianDecomposition],
) -> SensitivityResult:
    """
    Analyze which Hamiltonian terms contribute most to variant discrimination.

    Sensitivity of term k = std(term_k across variants) / std(H_eff across variants)

    Higher sensitivity → that term varies more across variants → more
    discriminating → more important for predicting phase separation differences.

    Parameters
    ----------
    decompositions : Dict[str, HamiltonianDecomposition]
        H_eff decompositions keyed by variant name

    Returns
    -------
    SensitivityResult
        Sensitivity analysis
    """
    names = list(decompositions.keys())
    n = len(names)

    term_keys = [
        "E_sticker_sticker", "E_cross", "E_linker_linker",
        "phi_clustering", "phi_connectivity", "phi_percolation",
        "phi_arrangement", "phi_homology",
    ]

    values = np.zeros((n, len(term_keys)))
    H_eff_arr = np.zeros(n)

    for i, name in enumerate(names):
        d = decompositions[name]
        H_eff_arr[i] = d.H_eff
        values[i, 0] = d.E_sticker_sticker
        values[i, 1] = d.E_cross
        values[i, 2] = d.E_linker_linker
        values[i, 3] = d.phi_clustering
        values[i, 4] = d.phi_connectivity
        values[i, 5] = d.phi_percolation
        values[i, 6] = d.phi_arrangement
        values[i, 7] = d.phi_homology

    # Sensitivity: how much does each term vary relative to total
    H_eff_std = np.std(H_eff_arr)
    sensitivities = {}
    for j, key in enumerate(term_keys):
        term_std = np.std(values[:, j])
        sensitivities[key] = float(term_std / H_eff_std) if H_eff_std > 1e-15 else 0.0

    return SensitivityResult(
        term_names=term_keys,
        term_values=values,
        variant_names=names,
        H_eff_values=H_eff_arr,
        sensitivities=sensitivities,
    )


# =============================================================================
# BATCH COMPUTATION
# =============================================================================

def compute_all_H_eff(
    variants: Dict,
    intermaps: Dict[str, np.ndarray],
    sticker_masks: Dict[str, StickerMask],
    topology_metrics: Dict[str, TopologyMetrics],
    homology_metrics: Dict[str, HomologyMetrics],
    entropy_metrics: Dict[str, EntropyMetrics],
    omega_values: Dict[str, float],
    params: Optional[HamiltonianParams] = None,
) -> Dict[str, HamiltonianDecomposition]:
    """
    Compute H_eff for all variants.

    Parameters
    ----------
    variants : Dict
        Variant registry
    intermaps : Dict[str, np.ndarray]
        Interaction maps
    sticker_masks : Dict[str, StickerMask]
        Sticker masks
    topology_metrics : Dict[str, TopologyMetrics]
        Topology results
    homology_metrics : Dict[str, HomologyMetrics]
        Homology results
    entropy_metrics : Dict[str, EntropyMetrics]
        Entropy results
    omega_values : Dict[str, float]
        Omega values per variant
    params : Optional[HamiltonianParams]
        Coupling constants

    Returns
    -------
    Dict[str, HamiltonianDecomposition]
        H_eff decomposition per variant
    """
    results = {}
    for name in variants:
        results[name] = compute_H_eff(
            intermap=intermaps[name],
            sticker_mask=sticker_masks[name],
            topology=topology_metrics[name],
            homology=homology_metrics[name],
            entropy_metrics=entropy_metrics[name],
            omega=omega_values[name],
            params=params,
            variant_name=name,
        )
    return results
