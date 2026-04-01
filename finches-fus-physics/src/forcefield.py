"""
forcefield.py - FINCHES/MPIPI-GG style residue-residue interaction model.

This module implements a coarse-grained force field for intrinsically
disordered protein (IDP) interactions, based on:
- MPIPI-GG (Mao et al., 2010; Dignon et al., 2018)
- FINCHES (Holehouse et al.)

Key interaction types:
1. Aromatic-aromatic (π-π stacking): Y-Y, F-F, W-W, Y-F, etc.
2. Cation-π: R/K with Y/F/W
3. Electrostatic: R/K with D/E
4. Hydrophobic: I, L, V, M, A interactions
5. Hydrogen bonding proxy: S, T, N, Q interactions

Energy units: kT (thermal energy at 300K)
Negative = attractive, Positive = repulsive
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


# =============================================================================
# AMINO ACID INDEXING
# =============================================================================

# Standard amino acid order for matrix indexing
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: i for i, aa in enumerate(AA_ORDER)}
INDEX_TO_AA = {i: aa for i, aa in enumerate(AA_ORDER)}
N_AMINO_ACIDS = len(AA_ORDER)


def aa_index(aa: str) -> int:
    """Get matrix index for an amino acid."""
    if aa not in AA_TO_INDEX:
        raise ValueError(f"Unknown amino acid: {aa}")
    return AA_TO_INDEX[aa]


# =============================================================================
# RESIDUE PROPERTIES (MPIPI-GG STYLE)
# =============================================================================

# Hydrophobicity scale (Kyte-Doolittle inspired, normalized)
# More negative = more hydrophobic
HYDROPHOBICITY = {
    "A":  0.17, "C":  0.24, "D": -0.78, "E": -0.64, "F":  0.61,
    "G":  0.01, "H": -0.40, "I":  0.73, "K": -1.10, "L":  0.53,
    "M":  0.26, "N": -0.60, "P": -0.07, "Q": -0.69, "R": -1.80,
    "S": -0.26, "T": -0.18, "V":  0.54, "W":  0.37, "Y":  0.02,
}

# Charge at pH 7 (simplified)
CHARGE = {
    "A": 0, "C": 0, "D": -1, "E": -1, "F": 0,
    "G": 0, "H": 0, "I": 0, "K": +1, "L": 0,
    "M": 0, "N": 0, "P": 0, "Q": 0, "R": +1,
    "S": 0, "T": 0, "V": 0, "W": 0, "Y": 0,
}

# Aromaticity (for π-π and cation-π)
AROMATICITY = {
    "A": 0.0, "C": 0.0, "D": 0.0, "E": 0.0, "F": 1.0,
    "G": 0.0, "H": 0.5, "I": 0.0, "K": 0.0, "L": 0.0,
    "M": 0.0, "N": 0.0, "P": 0.0, "Q": 0.0, "R": 0.0,
    "S": 0.0, "T": 0.0, "V": 0.0, "W": 1.0, "Y": 0.9,
}

# Cationic character (for cation-π)
CATIONICITY = {
    "A": 0.0, "C": 0.0, "D": 0.0, "E": 0.0, "F": 0.0,
    "G": 0.0, "H": 0.3, "I": 0.0, "K": 1.0, "L": 0.0,
    "M": 0.0, "N": 0.0, "P": 0.0, "Q": 0.0, "R": 1.0,
    "S": 0.0, "T": 0.0, "V": 0.0, "W": 0.0, "Y": 0.0,
}


# =============================================================================
# MPIPI-GG STYLE INTERACTION MATRIX
# =============================================================================

@dataclass
class ForceFieldParameters:
    """
    Parameters for the MPIPI-GG style force field.

    All energies in units of kT.
    """
    # Aromatic-aromatic (π-π stacking)
    epsilon_aromatic: float = -0.65  # Y-Y, F-F, W-W base energy

    # Cation-π interactions
    epsilon_cation_pi: float = -0.50  # R/K with Y/F/W

    # Electrostatic
    epsilon_attractive: float = -0.30  # Opposite charges (R/K with D/E)
    epsilon_repulsive: float = +0.20   # Like charges

    # Hydrophobic
    epsilon_hydrophobic: float = -0.15  # Hydrophobic-hydrophobic scaling

    # Polar/H-bond proxy
    epsilon_polar: float = -0.05  # Weak polar interactions

    # Background
    epsilon_background: float = 0.0  # Neutral interactions

    # Special: Glycine spacer effect (reduces local interactions)
    glycine_factor: float = 0.3  # Glycine weakens neighboring interactions

    # Proline disruption (breaks secondary structure)
    proline_penalty: float = +0.10  # Slight repulsion


def build_interaction_matrix(params: Optional[ForceFieldParameters] = None) -> np.ndarray:
    """
    Build the 20x20 residue-residue interaction energy matrix.

    The matrix is symmetric: E(i,j) = E(j,i)

    Parameters
    ----------
    params : Optional[ForceFieldParameters]
        Force field parameters. Uses defaults if None.

    Returns
    -------
    np.ndarray
        20x20 interaction matrix, indexed by AA_ORDER
    """
    if params is None:
        params = ForceFieldParameters()

    matrix = np.zeros((N_AMINO_ACIDS, N_AMINO_ACIDS), dtype=np.float64)

    for i, aa_i in enumerate(AA_ORDER):
        for j, aa_j in enumerate(AA_ORDER):
            energy = _compute_pairwise_energy(aa_i, aa_j, params)
            matrix[i, j] = energy

    # Ensure symmetry (should already be symmetric, but enforce)
    matrix = 0.5 * (matrix + matrix.T)

    return matrix


def _compute_pairwise_energy(aa_i: str, aa_j: str, params: ForceFieldParameters) -> float:
    """
    Compute pairwise interaction energy between two amino acids.

    Parameters
    ----------
    aa_i : str
        First amino acid
    aa_j : str
        Second amino acid
    params : ForceFieldParameters
        Force field parameters

    Returns
    -------
    float
        Interaction energy in kT
    """
    energy = 0.0

    # Get properties
    arom_i, arom_j = AROMATICITY[aa_i], AROMATICITY[aa_j]
    cat_i, cat_j = CATIONICITY[aa_i], CATIONICITY[aa_j]
    charge_i, charge_j = CHARGE[aa_i], CHARGE[aa_j]
    hydro_i, hydro_j = HYDROPHOBICITY[aa_i], HYDROPHOBICITY[aa_j]

    # 1. Aromatic-aromatic interactions (π-π stacking)
    aromatic_term = arom_i * arom_j * params.epsilon_aromatic
    energy += aromatic_term

    # 2. Cation-π interactions
    cation_pi_term = (cat_i * arom_j + cat_j * arom_i) * params.epsilon_cation_pi
    energy += cation_pi_term

    # 3. Electrostatic interactions
    if charge_i != 0 and charge_j != 0:
        if charge_i * charge_j < 0:  # Opposite charges
            energy += params.epsilon_attractive
        else:  # Like charges
            energy += params.epsilon_repulsive

    # 4. Hydrophobic interactions (both hydrophobic)
    if hydro_i > 0.2 and hydro_j > 0.2:
        hydro_term = np.sqrt(hydro_i * hydro_j) * params.epsilon_hydrophobic
        energy += hydro_term

    # 5. Polar interactions (weak, S/T/N/Q)
    polar_residues = {"S", "T", "N", "Q"}
    if aa_i in polar_residues and aa_j in polar_residues:
        energy += params.epsilon_polar

    # 6. Glycine effects (weakens interactions)
    if aa_i == "G" or aa_j == "G":
        energy *= params.glycine_factor

    # 7. Proline penalties
    if aa_i == "P" or aa_j == "P":
        energy += params.proline_penalty

    return energy


# =============================================================================
# PRE-COMPUTED MATRICES
# =============================================================================

# Default MPIPI-GG interaction matrix
_DEFAULT_MATRIX: Optional[np.ndarray] = None


def get_default_matrix() -> np.ndarray:
    """Get the default interaction matrix (lazy initialization)."""
    global _DEFAULT_MATRIX
    if _DEFAULT_MATRIX is None:
        _DEFAULT_MATRIX = build_interaction_matrix()
    return _DEFAULT_MATRIX.copy()


# =============================================================================
# INTERACTION LOOKUP
# =============================================================================

def get_interaction_energy(
    aa_i: str,
    aa_j: str,
    matrix: Optional[np.ndarray] = None
) -> float:
    """
    Look up interaction energy between two amino acids.

    Parameters
    ----------
    aa_i : str
        First amino acid (one-letter code)
    aa_j : str
        Second amino acid (one-letter code)
    matrix : Optional[np.ndarray]
        Interaction matrix. Uses default if None.

    Returns
    -------
    float
        Interaction energy in kT
    """
    if matrix is None:
        matrix = get_default_matrix()

    i = aa_index(aa_i)
    j = aa_index(aa_j)
    return matrix[i, j]


def get_interaction_vector(aa: str, matrix: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Get the interaction profile for a single amino acid.

    Returns the row/column of the interaction matrix for this residue.

    Parameters
    ----------
    aa : str
        Amino acid (one-letter code)
    matrix : Optional[np.ndarray]
        Interaction matrix. Uses default if None.

    Returns
    -------
    np.ndarray
        20-element vector of interactions with all amino acids
    """
    if matrix is None:
        matrix = get_default_matrix()

    i = aa_index(aa)
    return matrix[i, :].copy()


# =============================================================================
# SPECIALIZED INTERACTION MATRICES
# =============================================================================

def build_aromatic_only_matrix(params: Optional[ForceFieldParameters] = None) -> np.ndarray:
    """
    Build matrix with only aromatic (π-π) interactions.

    Useful for isolating the contribution of aromatic stacking.
    """
    if params is None:
        params = ForceFieldParameters()

    matrix = np.zeros((N_AMINO_ACIDS, N_AMINO_ACIDS), dtype=np.float64)

    for i, aa_i in enumerate(AA_ORDER):
        for j, aa_j in enumerate(AA_ORDER):
            arom_i = AROMATICITY[aa_i]
            arom_j = AROMATICITY[aa_j]
            matrix[i, j] = arom_i * arom_j * params.epsilon_aromatic

    return matrix


def build_cation_pi_only_matrix(params: Optional[ForceFieldParameters] = None) -> np.ndarray:
    """
    Build matrix with only cation-π interactions.

    Useful for isolating R/K interactions with aromatics.
    """
    if params is None:
        params = ForceFieldParameters()

    matrix = np.zeros((N_AMINO_ACIDS, N_AMINO_ACIDS), dtype=np.float64)

    for i, aa_i in enumerate(AA_ORDER):
        for j, aa_j in enumerate(AA_ORDER):
            cat_i = CATIONICITY[aa_i]
            cat_j = CATIONICITY[aa_j]
            arom_i = AROMATICITY[aa_i]
            arom_j = AROMATICITY[aa_j]
            matrix[i, j] = (cat_i * arom_j + cat_j * arom_i) * params.epsilon_cation_pi

    return matrix


def build_electrostatic_only_matrix(params: Optional[ForceFieldParameters] = None) -> np.ndarray:
    """
    Build matrix with only electrostatic interactions.
    """
    if params is None:
        params = ForceFieldParameters()

    matrix = np.zeros((N_AMINO_ACIDS, N_AMINO_ACIDS), dtype=np.float64)

    for i, aa_i in enumerate(AA_ORDER):
        for j, aa_j in enumerate(AA_ORDER):
            charge_i = CHARGE[aa_i]
            charge_j = CHARGE[aa_j]
            if charge_i != 0 and charge_j != 0:
                if charge_i * charge_j < 0:
                    matrix[i, j] = params.epsilon_attractive
                else:
                    matrix[i, j] = params.epsilon_repulsive

    return matrix


# =============================================================================
# MATRIX ANALYSIS
# =============================================================================

def get_most_attractive_pairs(
    matrix: Optional[np.ndarray] = None,
    n_pairs: int = 10
) -> list:
    """
    Get the most attractive amino acid pairs.

    Parameters
    ----------
    matrix : Optional[np.ndarray]
        Interaction matrix
    n_pairs : int
        Number of pairs to return

    Returns
    -------
    list
        List of (aa_i, aa_j, energy) tuples
    """
    if matrix is None:
        matrix = get_default_matrix()

    # Get upper triangle (avoid duplicates)
    pairs = []
    for i in range(N_AMINO_ACIDS):
        for j in range(i, N_AMINO_ACIDS):
            pairs.append((INDEX_TO_AA[i], INDEX_TO_AA[j], matrix[i, j]))

    # Sort by energy (most negative first)
    pairs.sort(key=lambda x: x[2])
    return pairs[:n_pairs]


def get_residue_stickiness(
    aa: str,
    matrix: Optional[np.ndarray] = None
) -> float:
    """
    Compute the mean interaction strength (stickiness) of a residue.

    More negative = stickier.

    Parameters
    ----------
    aa : str
        Amino acid
    matrix : Optional[np.ndarray]
        Interaction matrix

    Returns
    -------
    float
        Mean interaction energy across all partners
    """
    if matrix is None:
        matrix = get_default_matrix()

    i = aa_index(aa)
    return float(np.mean(matrix[i, :]))


def compute_stickiness_ranking(matrix: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Rank all amino acids by their mean stickiness.

    Returns
    -------
    Dict[str, float]
        Amino acid -> mean stickiness, sorted from stickiest to least sticky
    """
    if matrix is None:
        matrix = get_default_matrix()

    stickiness = {}
    for aa in AA_ORDER:
        stickiness[aa] = get_residue_stickiness(aa, matrix)

    # Sort by stickiness
    return dict(sorted(stickiness.items(), key=lambda x: x[1]))


# =============================================================================
# MATRIX I/O
# =============================================================================

def matrix_to_dict(matrix: np.ndarray) -> Dict[Tuple[str, str], float]:
    """
    Convert interaction matrix to dictionary format.

    Useful for serialization or inspection.
    """
    result = {}
    for i, aa_i in enumerate(AA_ORDER):
        for j, aa_j in enumerate(AA_ORDER):
            result[(aa_i, aa_j)] = float(matrix[i, j])
    return result


def print_matrix_summary(matrix: Optional[np.ndarray] = None) -> None:
    """Print a summary of the interaction matrix."""
    if matrix is None:
        matrix = get_default_matrix()

    print("Interaction Matrix Summary")
    print("=" * 50)
    print(f"Shape: {matrix.shape}")
    print(f"Min energy: {matrix.min():.3f} kT")
    print(f"Max energy: {matrix.max():.3f} kT")
    print(f"Mean energy: {matrix.mean():.3f} kT")

    print("\nMost attractive pairs:")
    for aa_i, aa_j, energy in get_most_attractive_pairs(matrix, 5):
        print(f"  {aa_i}-{aa_j}: {energy:.3f} kT")

    print("\nStickiest residues:")
    ranking = compute_stickiness_ranking(matrix)
    for i, (aa, stick) in enumerate(ranking.items()):
        if i < 5:
            print(f"  {aa}: {stick:.3f} kT (mean)")


# =============================================================================
# EXPORTS
# =============================================================================

# Default parameters
DEFAULT_PARAMS = ForceFieldParameters()

# Default matrix (lazy loaded)
INTERACTION_MATRIX = get_default_matrix()


if __name__ == "__main__":
    # Demo
    print_matrix_summary()

    print("\n\nKey interaction energies for FUS LCD:")
    print("Y-Y:", get_interaction_energy("Y", "Y"), "kT")
    print("Y-R:", get_interaction_energy("Y", "R"), "kT (cation-π)")
    print("Y-S:", get_interaction_energy("Y", "S"), "kT")
    print("G-G:", get_interaction_energy("G", "G"), "kT")
    print("R-D:", get_interaction_energy("R", "D"), "kT (salt bridge)")
