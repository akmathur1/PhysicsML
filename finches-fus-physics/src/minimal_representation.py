"""
minimal_representation.py - Finding invariant representations for IDP energetics.

The key insight: We want a MINIMAL set of features θ such that:

    E_total = ∫∫ ε(r_i, r_j) ρ(r_i, r_j) dr_i dr_j  ≈  f(θ)

Two sequences with the SAME θ should have the SAME integrated energy,
regardless of the specific arrangement of residues.

This is the "sufficient statistic" for phase separation thermodynamics.

Mathematical Framework:
----------------------
Full representation:     N×N interaction map I[i,j] = ε(aa_i, aa_j)
Integrated quantity:     E = Σᵢⱼ I[i,j] × w(|i-j|)
Minimal representation:  θ = {n_αβ} = counts of interaction types

The question: What is the MINIMAL θ that preserves E?

Approaches:
1. Composition-based: θ = {n_A, n_C, n_D, ...} (20 numbers)
2. Interaction-type: θ = {n_YY, n_YR, n_RR, ...} (reduced set)
3. Sticker-based: θ = {n_sticker, f_sticker, <ε_ss>}
4. Optimal: Use PCA/SVD to find minimal basis
"""

from __future__ import annotations
import numpy as np
from scipy import linalg
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass

from .sequences import SequenceRecord, AROMATIC_RESIDUES, CATIONIC_RESIDUES
from .forcefield import get_default_matrix, AA_ORDER, aa_index


# =============================================================================
# THE FUNDAMENTAL INTEGRAL
# =============================================================================

def compute_total_energy_integral(
    sequence: str,
    interaction_matrix: Optional[np.ndarray] = None,
    weight_function: Optional[Callable[[int], float]] = None
) -> float:
    """
    Compute the total integrated energy:

        E = Σᵢ Σⱼ ε(aa_i, aa_j) × w(|i-j|)

    This is THE quantity we want our minimal representation to preserve.

    Parameters
    ----------
    sequence : str
        Amino acid sequence
    interaction_matrix : Optional[np.ndarray]
        20×20 force field matrix
    weight_function : Optional[Callable]
        w(|i-j|) weighting. Default: uniform (w=1 for all)
        Could be: exponential decay, contact probability, etc.

    Returns
    -------
    float
        Total integrated energy
    """
    if interaction_matrix is None:
        interaction_matrix = get_default_matrix()

    if weight_function is None:
        weight_function = lambda d: 1.0  # Uniform weighting

    n = len(sequence)
    indices = np.array([aa_index(aa) for aa in sequence])

    total_energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):  # Upper triangle only (symmetric)
            epsilon_ij = interaction_matrix[indices[i], indices[j]]
            w_ij = weight_function(abs(j - i))
            total_energy += epsilon_ij * w_ij

    return total_energy


def compute_energy_with_contact_probability(
    sequence: str,
    interaction_matrix: Optional[np.ndarray] = None,
    persistence_length: float = 5.0
) -> float:
    """
    More physical: weight by contact probability from polymer physics.

    P(contact | |i-j|) ∝ |i-j|^(-3/2) for Gaussian chain

    This is closer to what actually matters for phase separation.
    """
    def contact_probability(d: int) -> float:
        if d < 3:
            return 0.0  # Excluded volume
        # Gaussian chain contact probability
        return (d / persistence_length) ** (-1.5)

    return compute_total_energy_integral(sequence, interaction_matrix, contact_probability)


# =============================================================================
# MINIMAL REPRESENTATIONS
# =============================================================================

@dataclass
class MinimalRepresentation:
    """A minimal representation θ of a sequence."""
    name: str
    theta: np.ndarray  # The minimal parameter vector
    dimension: int     # Dimensionality of θ

    def predict_energy(self, coefficients: np.ndarray) -> float:
        """E ≈ θ · c"""
        return np.dot(self.theta, coefficients)


def composition_representation(sequence: str) -> MinimalRepresentation:
    """
    Representation 1: Amino acid composition (20D)

    θ = [n_A, n_C, n_D, ..., n_Y]

    This is sufficient if E = Σ_α Σ_β n_α n_β ε_αβ / N
    (mean-field approximation)
    """
    counts = np.zeros(20)
    for aa in sequence:
        counts[aa_index(aa)] += 1

    return MinimalRepresentation(
        name="composition",
        theta=counts,
        dimension=20
    )


def pair_count_representation(
    sequence: str,
    interaction_matrix: Optional[np.ndarray] = None
) -> MinimalRepresentation:
    """
    Representation 2: Pair interaction counts (210D for unique pairs)

    θ = [n_AA, n_AC, n_AD, ..., n_YY]

    E = Σ_αβ n_αβ × ε_αβ  (EXACT if no distance weighting)
    """
    if interaction_matrix is None:
        interaction_matrix = get_default_matrix()

    n = len(sequence)
    indices = np.array([aa_index(aa) for aa in sequence])

    # Count pairs (210 unique pairs for 20 amino acids)
    pair_counts = np.zeros((20, 20))
    for i in range(n):
        for j in range(i + 1, n):
            pair_counts[indices[i], indices[j]] += 1
            pair_counts[indices[j], indices[i]] += 1

    # Flatten upper triangle
    theta = pair_counts[np.triu_indices(20)]

    return MinimalRepresentation(
        name="pair_counts",
        theta=theta,
        dimension=210
    )


def sticker_representation(sequence: str) -> MinimalRepresentation:
    """
    Representation 3: Sticker-based (very low D)

    θ = [n_sticker, n_linker, n_sticker_sticker_pairs, n_sticker_linker_pairs, ...]

    This is the "sticker-spacer" minimal model.
    """
    sticker_residues = AROMATIC_RESIDUES | CATIONIC_RESIDUES

    n = len(sequence)
    is_sticker = np.array([aa in sticker_residues for aa in sequence])

    n_sticker = np.sum(is_sticker)
    n_linker = n - n_sticker

    # Count pair types
    n_ss = 0  # sticker-sticker
    n_ll = 0  # linker-linker
    n_sl = 0  # sticker-linker

    for i in range(n):
        for j in range(i + 1, n):
            if is_sticker[i] and is_sticker[j]:
                n_ss += 1
            elif not is_sticker[i] and not is_sticker[j]:
                n_ll += 1
            else:
                n_sl += 1

    theta = np.array([n_sticker, n_linker, n_ss, n_ll, n_sl], dtype=float)

    return MinimalRepresentation(
        name="sticker_based",
        theta=theta,
        dimension=5
    )


def interaction_class_representation(sequence: str) -> MinimalRepresentation:
    """
    Representation 4: Interaction class counts

    Classes:
    - Aromatic-Aromatic (π-π)
    - Cation-Aromatic (cation-π)
    - Charge-Charge (electrostatic)
    - Hydrophobic-Hydrophobic
    - Polar-Polar
    - Other

    θ = [n_pipi, n_catpi, n_elec, n_hydro, n_polar, n_other]
    """
    n = len(sequence)

    # Residue classifications
    aromatic = AROMATIC_RESIDUES
    cationic = CATIONIC_RESIDUES
    anionic = {'D', 'E'}
    hydrophobic = {'A', 'V', 'L', 'I', 'M', 'F', 'W'}
    polar = {'S', 'T', 'N', 'Q', 'C', 'Y'}  # Y is both aromatic and polar

    counts = {
        'pipi': 0,      # Aromatic-Aromatic
        'catpi': 0,     # Cation-Aromatic
        'elec_att': 0,  # Opposite charges
        'elec_rep': 0,  # Same charges
        'hydro': 0,     # Hydrophobic-Hydrophobic
        'polar': 0,     # Polar-Polar
        'other': 0,
    }

    for i in range(n):
        for j in range(i + 1, n):
            aa_i, aa_j = sequence[i], sequence[j]

            # Classify this pair
            if aa_i in aromatic and aa_j in aromatic:
                counts['pipi'] += 1
            elif (aa_i in cationic and aa_j in aromatic) or (aa_j in cationic and aa_i in aromatic):
                counts['catpi'] += 1
            elif (aa_i in cationic and aa_j in anionic) or (aa_j in cationic and aa_i in anionic):
                counts['elec_att'] += 1
            elif (aa_i in cationic and aa_j in cationic) or (aa_i in anionic and aa_j in anionic):
                counts['elec_rep'] += 1
            elif aa_i in hydrophobic and aa_j in hydrophobic:
                counts['hydro'] += 1
            elif aa_i in polar and aa_j in polar:
                counts['polar'] += 1
            else:
                counts['other'] += 1

    theta = np.array(list(counts.values()), dtype=float)

    return MinimalRepresentation(
        name="interaction_class",
        theta=theta,
        dimension=7
    )


# =============================================================================
# FINDING THE OPTIMAL MINIMAL REPRESENTATION
# =============================================================================

def find_optimal_representation(
    sequences: List[str],
    n_components: int = 3,
    interaction_matrix: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find the OPTIMAL minimal representation using SVD/PCA.

    Given a set of sequences, find the minimal basis that captures
    the variance in their interaction maps.

    This answers: "What is the lowest-dimensional θ that preserves E?"

    Parameters
    ----------
    sequences : List[str]
        Training sequences
    n_components : int
        Dimensionality of minimal representation
    interaction_matrix : Optional[np.ndarray]
        Force field

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (projections, components, explained_variance_ratio)
    """
    if interaction_matrix is None:
        interaction_matrix = get_default_matrix()

    # Build composition matrix (sequences × 20 amino acids)
    X = np.zeros((len(sequences), 20))
    for i, seq in enumerate(sequences):
        for aa in seq:
            X[i, aa_index(aa)] += 1
        X[i] /= len(seq)  # Normalize to fractions

    # Compute energies for each sequence
    energies = np.array([
        compute_total_energy_integral(seq, interaction_matrix) / len(seq)**2
        for seq in sequences
    ])

    # SVD to find principal components
    X_centered = X - X.mean(axis=0)
    U, S, Vt = linalg.svd(X_centered, full_matrices=False)

    # Project onto top components
    projections = U[:, :n_components] * S[:n_components]
    components = Vt[:n_components, :]

    # Explained variance
    total_var = np.sum(S**2)
    explained_var = S[:n_components]**2 / total_var

    return projections, components, explained_var


# =============================================================================
# THE KEY THEOREM: ENERGY INVARIANCE
# =============================================================================

def test_representation_invariance(
    sequences: List[str],
    representation_func: Callable[[str], MinimalRepresentation],
    interaction_matrix: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Test if a representation preserves the energy integral.

    THE KEY QUESTION:
    If θ(seq1) = θ(seq2), does E(seq1) ≈ E(seq2)?

    This tests whether the representation is a SUFFICIENT STATISTIC
    for the energy.

    Parameters
    ----------
    sequences : List[str]
        Test sequences
    representation_func : Callable
        Function that computes θ from sequence

    Returns
    -------
    Dict[str, float]
        R² score and other metrics
    """
    if interaction_matrix is None:
        interaction_matrix = get_default_matrix()

    # Compute representations and energies
    thetas = []
    energies = []

    for seq in sequences:
        rep = representation_func(seq)
        thetas.append(rep.theta)
        E = compute_total_energy_integral(seq, interaction_matrix)
        energies.append(E / len(seq)**2)  # Normalize by N²

    thetas = np.array(thetas)
    energies = np.array(energies)

    # Fit linear model: E = θ · c
    # Solve least squares: c = (θᵀθ)⁻¹ θᵀ E
    coefficients, residuals, rank, s = np.linalg.lstsq(thetas, energies, rcond=None)

    # Predictions
    E_predicted = thetas @ coefficients

    # R² score
    ss_res = np.sum((energies - E_predicted)**2)
    ss_tot = np.sum((energies - energies.mean())**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # RMSE
    rmse = np.sqrt(np.mean((energies - E_predicted)**2))

    return {
        'r_squared': r_squared,
        'rmse': rmse,
        'dimension': thetas.shape[1],
        'coefficients': coefficients,
    }


# =============================================================================
# GENERATE TEST SEQUENCES FOR VALIDATION
# =============================================================================

def generate_shuffled_sequences(base_sequence: str, n_shuffles: int = 100) -> List[str]:
    """
    Generate shuffled versions of a sequence.

    These have IDENTICAL composition but DIFFERENT arrangements.
    If composition is sufficient, they should have the same energy.
    """
    sequences = [base_sequence]
    seq_list = list(base_sequence)

    for _ in range(n_shuffles):
        np.random.shuffle(seq_list)
        sequences.append(''.join(seq_list))

    return sequences


def generate_compositional_variants(
    base_composition: Dict[str, int],
    n_variants: int = 100
) -> List[str]:
    """
    Generate sequences with the same composition but random order.
    """
    # Build base sequence from composition
    base_seq = ''
    for aa, count in base_composition.items():
        base_seq += aa * count

    return generate_shuffled_sequences(base_seq, n_variants)


# =============================================================================
# THE MINIMAL REPRESENTATION FINDER
# =============================================================================

def find_minimal_sufficient_representation(
    sequences: List[str],
    max_dimension: int = 20
) -> Dict[str, any]:
    """
    Find the minimal dimensionality θ that is SUFFICIENT for energy prediction.

    We want the smallest dim(θ) such that R² > 0.99.

    This is THE answer to "what is the minimal representation?"
    """
    results = {}

    # Test each representation
    representations = [
        ('composition_20D', composition_representation),
        ('sticker_5D', sticker_representation),
        ('interaction_class_7D', interaction_class_representation),
        ('pair_counts_210D', pair_count_representation),
    ]

    for name, rep_func in representations:
        metrics = test_representation_invariance(sequences, rep_func)
        results[name] = {
            'dimension': metrics['dimension'],
            'r_squared': metrics['r_squared'],
            'rmse': metrics['rmse'],
        }

    # Find minimal sufficient representation
    sufficient = [(name, r['dimension'], r['r_squared'])
                  for name, r in results.items() if r['r_squared'] > 0.95]

    if sufficient:
        minimal = min(sufficient, key=lambda x: x[1])
        results['minimal_sufficient'] = {
            'name': minimal[0],
            'dimension': minimal[1],
            'r_squared': minimal[2],
        }

    return results


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    from .sequences import FUS_LCD_SEQUENCE, VARIANTS

    print("="*70)
    print("FINDING THE MINIMAL REPRESENTATION")
    print("="*70)

    # Test on FUS variants
    sequences = [v.sequence for v in VARIANTS.values()]
    names = list(VARIANTS.keys())

    print("\n1. Energy integrals for FUS variants:")
    print("-"*50)
    for name, seq in zip(names, sequences):
        E = compute_total_energy_integral(seq)
        E_contact = compute_energy_with_contact_probability(seq)
        print(f"{name:12s}: E_uniform={E:10.2f}, E_contact={E_contact:10.4f}")

    print("\n2. Testing representation sufficiency:")
    print("-"*50)

    # Generate shuffled sequences for testing
    test_seqs = generate_shuffled_sequences(FUS_LCD_SEQUENCE, 50)

    results = find_minimal_sufficient_representation(test_seqs)
    for name, metrics in results.items():
        if name != 'minimal_sufficient':
            print(f"{name:25s}: dim={metrics['dimension']:3d}, R²={metrics['r_squared']:.4f}")

    if 'minimal_sufficient' in results:
        print(f"\n>>> MINIMAL SUFFICIENT: {results['minimal_sufficient']}")
