"""
sequences.py - FUS LCD sequence management and mutation utilities.

This module provides:
- Canonical FUS LCD sequence (residues 1-214, UniProt P35637)
- Safe mutation utilities with validation
- Pre-defined variant registry for systematic studies
- Sequence analysis utilities (composition, aromatic positions)

The FUS low-complexity domain (LCD) is an intrinsically disordered region
enriched in tyrosine (Y), glycine (G), serine (S), and glutamine (Q).
Tyrosine residues serve as "stickers" driving phase separation.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import numpy as np


# =============================================================================
# CANONICAL FUS LCD SEQUENCE (Residues 1-214, UniProt P35637)
# =============================================================================

# Human FUS protein, N-terminal low-complexity domain
# This is the experimentally characterized prion-like domain (residues 1-214)
# Sequence from UniProt P35637
FUS_LCD_SEQUENCE = (
    "MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQS"
    "QNTGYGTQSTPQGYGSTGGYGSSQSSQSSYGQQSSYPGYGQQPAPSSTSGSYGSSSQSSSY"
    "GQPQSGSYSQQPSYGGQQQSYGQQQSYNPPQGYGQQNQYNSSSSGGGGGGGGGNYGQDQSS"
    "MSSGGGSGGGYGNQDQSGGGGSGGYGQQDRG"
)

# Validate sequence length (214 residues for FUS LCD)
assert len(FUS_LCD_SEQUENCE) == 214, f"FUS LCD should be 214 residues, got {len(FUS_LCD_SEQUENCE)}"


# =============================================================================
# AMINO ACID CLASSIFICATIONS
# =============================================================================

AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
AROMATIC_RESIDUES = {"Y", "F", "W"}
CATIONIC_RESIDUES = {"R", "K"}
ANIONIC_RESIDUES = {"D", "E"}
POLAR_RESIDUES = {"S", "T", "N", "Q", "C"}
HYDROPHOBIC_RESIDUES = {"A", "V", "L", "I", "M", "P", "G"}

# One-letter to three-letter code mapping
AA_3LETTER = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR"
}


# =============================================================================
# SEQUENCE DATACLASS
# =============================================================================

@dataclass
class SequenceRecord:
    """
    Container for a protein sequence with metadata.

    Attributes
    ----------
    name : str
        Identifier for this sequence variant
    sequence : str
        One-letter amino acid sequence
    mutations : List[str]
        List of mutations applied (e.g., ["Y144S", "Y149S"])
    description : str
        Human-readable description of the variant
    parent : Optional[str]
        Name of the parent sequence (None for WT)
    """
    name: str
    sequence: str
    mutations: List[str] = field(default_factory=list)
    description: str = ""
    parent: Optional[str] = None

    def __post_init__(self):
        """Validate sequence on creation."""
        validate_sequence(self.sequence)

    @property
    def length(self) -> int:
        """Sequence length."""
        return len(self.sequence)

    @property
    def tyrosine_positions(self) -> List[int]:
        """0-indexed positions of tyrosine residues."""
        return [i for i, aa in enumerate(self.sequence) if aa == "Y"]

    @property
    def tyrosine_count(self) -> int:
        """Number of tyrosine residues."""
        return self.sequence.count("Y")

    @property
    def aromatic_positions(self) -> List[int]:
        """0-indexed positions of aromatic residues (Y, F, W)."""
        return [i for i, aa in enumerate(self.sequence) if aa in AROMATIC_RESIDUES]

    @property
    def aromatic_count(self) -> int:
        """Number of aromatic residues."""
        return sum(1 for aa in self.sequence if aa in AROMATIC_RESIDUES)

    @property
    def composition(self) -> Dict[str, int]:
        """Amino acid composition dictionary."""
        return {aa: self.sequence.count(aa) for aa in AMINO_ACIDS if self.sequence.count(aa) > 0}

    @property
    def aromatic_fraction(self) -> float:
        """Fraction of residues that are aromatic."""
        return self.aromatic_count / self.length

    def get_residue(self, position: int) -> str:
        """
        Get residue at 1-indexed position (biological convention).

        Parameters
        ----------
        position : int
            1-indexed residue position

        Returns
        -------
        str
            One-letter amino acid code
        """
        if position < 1 or position > self.length:
            raise IndexError(f"Position {position} out of range [1, {self.length}]")
        return self.sequence[position - 1]

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def to_fasta(self) -> str:
        """Format as FASTA string."""
        header = f">{self.name}"
        if self.mutations:
            header += f" mutations={','.join(self.mutations)}"
        if self.description:
            header += f" | {self.description}"
        # Wrap sequence at 60 characters
        wrapped = "\n".join(
            self.sequence[i:i+60] for i in range(0, len(self.sequence), 60)
        )
        return f"{header}\n{wrapped}"

    def __repr__(self) -> str:
        return f"SequenceRecord(name='{self.name}', length={self.length}, Y={self.tyrosine_count})"


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def validate_sequence(sequence: str) -> bool:
    """
    Validate that a sequence contains only standard amino acids.

    Parameters
    ----------
    sequence : str
        Amino acid sequence to validate

    Returns
    -------
    bool
        True if valid

    Raises
    ------
    ValueError
        If sequence contains invalid characters
    """
    invalid = set(sequence.upper()) - AMINO_ACIDS
    if invalid:
        raise ValueError(f"Invalid amino acid(s) in sequence: {invalid}")
    return True


def validate_mutation(mutation: str, sequence: str) -> Tuple[str, int, str]:
    """
    Parse and validate a mutation string.

    Parameters
    ----------
    mutation : str
        Mutation in format "X123Y" where X is original, 123 is position, Y is new
    sequence : str
        Reference sequence to validate against

    Returns
    -------
    Tuple[str, int, str]
        (original_aa, position_1indexed, new_aa)

    Raises
    ------
    ValueError
        If mutation format is invalid or doesn't match sequence
    """
    if len(mutation) < 3:
        raise ValueError(f"Invalid mutation format: {mutation}")

    original = mutation[0].upper()
    new = mutation[-1].upper()

    try:
        position = int(mutation[1:-1])
    except ValueError:
        raise ValueError(f"Invalid position in mutation: {mutation}")

    if original not in AMINO_ACIDS:
        raise ValueError(f"Invalid original amino acid '{original}' in {mutation}")
    if new not in AMINO_ACIDS:
        raise ValueError(f"Invalid new amino acid '{new}' in {mutation}")

    if position < 1 or position > len(sequence):
        raise ValueError(f"Position {position} out of range for sequence length {len(sequence)}")

    actual = sequence[position - 1]
    if actual != original:
        raise ValueError(
            f"Mutation {mutation} specifies {original} at position {position}, "
            f"but sequence has {actual}"
        )

    return original, position, new


# =============================================================================
# MUTATION UTILITIES
# =============================================================================

def apply_mutation(sequence: str, mutation: str) -> str:
    """
    Apply a single point mutation to a sequence.

    Parameters
    ----------
    sequence : str
        Original amino acid sequence
    mutation : str
        Mutation in format "X123Y"

    Returns
    -------
    str
        Mutated sequence
    """
    original, position, new = validate_mutation(mutation, sequence)
    seq_list = list(sequence)
    seq_list[position - 1] = new
    return "".join(seq_list)


def apply_mutations(sequence: str, mutations: List[str]) -> str:
    """
    Apply multiple point mutations to a sequence.

    Mutations are validated against the ORIGINAL sequence,
    then all applied simultaneously.

    Parameters
    ----------
    sequence : str
        Original amino acid sequence
    mutations : List[str]
        List of mutations in format "X123Y"

    Returns
    -------
    str
        Mutated sequence
    """
    # First validate all mutations against original sequence
    parsed = [validate_mutation(m, sequence) for m in mutations]

    # Check for conflicting mutations at same position
    positions = [p[1] for p in parsed]
    if len(positions) != len(set(positions)):
        raise ValueError("Multiple mutations at the same position")

    # Apply all mutations
    seq_list = list(sequence)
    for original, position, new in parsed:
        seq_list[position - 1] = new

    return "".join(seq_list)


def mutate_all_residues(
    sequence: str,
    target_residue: str,
    replacement: str,
    exclude_positions: Optional[Set[int]] = None
) -> Tuple[str, List[str]]:
    """
    Mutate all occurrences of a target residue to a replacement.

    Parameters
    ----------
    sequence : str
        Original sequence
    target_residue : str
        Residue to replace (e.g., "Y")
    replacement : str
        Replacement residue (e.g., "S")
    exclude_positions : Optional[Set[int]]
        1-indexed positions to exclude from mutation

    Returns
    -------
    Tuple[str, List[str]]
        (mutated_sequence, list_of_mutations)
    """
    if exclude_positions is None:
        exclude_positions = set()

    mutations = []
    for i, aa in enumerate(sequence):
        pos = i + 1  # 1-indexed
        if aa == target_residue and pos not in exclude_positions:
            mutations.append(f"{target_residue}{pos}{replacement}")

    mutated = apply_mutations(sequence, mutations)
    return mutated, mutations


# =============================================================================
# SEQUENCE ANALYSIS
# =============================================================================

def find_positions(sequence: str, residues: Set[str]) -> List[int]:
    """
    Find all positions (1-indexed) of specified residues.

    Parameters
    ----------
    sequence : str
        Amino acid sequence
    residues : Set[str]
        Set of residue types to find

    Returns
    -------
    List[int]
        1-indexed positions
    """
    return [i + 1 for i, aa in enumerate(sequence) if aa in residues]


def compute_sequence_properties(sequence: str) -> Dict:
    """
    Compute biophysical properties of a sequence.

    Parameters
    ----------
    sequence : str
        Amino acid sequence

    Returns
    -------
    Dict
        Dictionary of computed properties
    """
    n = len(sequence)

    # Counts
    n_aromatic = sum(1 for aa in sequence if aa in AROMATIC_RESIDUES)
    n_cationic = sum(1 for aa in sequence if aa in CATIONIC_RESIDUES)
    n_anionic = sum(1 for aa in sequence if aa in ANIONIC_RESIDUES)
    n_polar = sum(1 for aa in sequence if aa in POLAR_RESIDUES)
    n_hydrophobic = sum(1 for aa in sequence if aa in HYDROPHOBIC_RESIDUES)

    # Net charge (simple model, pH 7)
    net_charge = n_cationic - n_anionic

    # Tyrosine-specific
    n_tyr = sequence.count("Y")
    tyr_positions = find_positions(sequence, {"Y"})

    # Mean spacing between tyrosines
    if len(tyr_positions) > 1:
        spacings = np.diff(tyr_positions)
        mean_tyr_spacing = float(np.mean(spacings))
    else:
        mean_tyr_spacing = None

    return {
        "length": n,
        "n_aromatic": n_aromatic,
        "n_cationic": n_cationic,
        "n_anionic": n_anionic,
        "n_polar": n_polar,
        "n_hydrophobic": n_hydrophobic,
        "net_charge": net_charge,
        "n_tyrosine": n_tyr,
        "tyrosine_positions": tyr_positions,
        "mean_tyrosine_spacing": mean_tyr_spacing,
        "fraction_aromatic": n_aromatic / n,
        "fraction_tyrosine": n_tyr / n,
    }


# =============================================================================
# VARIANT REGISTRY
# =============================================================================

def create_wt_record() -> SequenceRecord:
    """Create the wild-type FUS LCD sequence record."""
    return SequenceRecord(
        name="WT",
        sequence=FUS_LCD_SEQUENCE,
        mutations=[],
        description="Wild-type FUS LCD (residues 1-214)",
        parent=None
    )


def create_y143s_record() -> SequenceRecord:
    """Create single Y143S mutant."""
    mutated = apply_mutation(FUS_LCD_SEQUENCE, "Y143S")
    return SequenceRecord(
        name="Y143S",
        sequence=mutated,
        mutations=["Y143S"],
        description="Single tyrosine-to-serine mutation at position 143",
        parent="WT"
    )


def create_multi_y_to_s_record() -> SequenceRecord:
    """
    Create multi-Y→S variant (aromatic sticker disruption).

    Mutates all tyrosines to serine, removing all aromatic stickers.
    This is a severe perturbation expected to abolish phase separation.
    """
    mutated, mutations = mutate_all_residues(FUS_LCD_SEQUENCE, "Y", "S")
    return SequenceRecord(
        name="AllY_to_S",
        sequence=mutated,
        mutations=mutations,
        description="All tyrosines mutated to serine (aromatic sticker disruption)",
        parent="WT"
    )


def create_y_to_a_record() -> SequenceRecord:
    """
    Create Y→A variant (aromatic knockout control).

    Alanine is a small hydrophobic residue that removes aromatic character
    while maintaining similar hydrophobicity. Used as a control to test
    whether aromatic pi-stacking specifically drives interactions.
    """
    mutated, mutations = mutate_all_residues(FUS_LCD_SEQUENCE, "Y", "A")
    return SequenceRecord(
        name="AllY_to_A",
        sequence=mutated,
        mutations=mutations,
        description="All tyrosines mutated to alanine (aromatic knockout control)",
        parent="WT"
    )


def create_y_to_f_record() -> SequenceRecord:
    """
    Create Y→F variant (conservative control).

    Phenylalanine retains aromatic character but lacks the hydroxyl group.
    This tests whether the tyrosine OH contributes to interactions
    (e.g., via hydrogen bonding or cation-pi modulation).
    """
    mutated, mutations = mutate_all_residues(FUS_LCD_SEQUENCE, "Y", "F")
    return SequenceRecord(
        name="AllY_to_F",
        sequence=mutated,
        mutations=mutations,
        description="All tyrosines mutated to phenylalanine (conservative aromatic control)",
        parent="WT"
    )


def build_variant_registry() -> Dict[str, SequenceRecord]:
    """
    Build the complete variant registry.

    Returns
    -------
    Dict[str, SequenceRecord]
        Dictionary mapping variant names to SequenceRecord objects
    """
    return {
        "WT": create_wt_record(),
        "Y143S": create_y143s_record(),
        "AllY_to_S": create_multi_y_to_s_record(),
        "AllY_to_A": create_y_to_a_record(),
        "AllY_to_F": create_y_to_f_record(),
    }


# Global variant registry (lazy initialization)
_VARIANT_REGISTRY: Optional[Dict[str, SequenceRecord]] = None


def get_variant_registry() -> Dict[str, SequenceRecord]:
    """
    Get the global variant registry (lazy initialization).

    Returns
    -------
    Dict[str, SequenceRecord]
        The variant registry
    """
    global _VARIANT_REGISTRY
    if _VARIANT_REGISTRY is None:
        _VARIANT_REGISTRY = build_variant_registry()
    return _VARIANT_REGISTRY


def get_variant(name: str) -> SequenceRecord:
    """
    Get a specific variant by name.

    Parameters
    ----------
    name : str
        Variant name (e.g., "WT", "Y144S")

    Returns
    -------
    SequenceRecord
        The requested variant

    Raises
    ------
    KeyError
        If variant not found
    """
    registry = get_variant_registry()
    if name not in registry:
        available = ", ".join(registry.keys())
        raise KeyError(f"Unknown variant '{name}'. Available: {available}")
    return registry[name]


def list_variants() -> List[str]:
    """List all available variant names."""
    return list(get_variant_registry().keys())


# =============================================================================
# SERIALIZATION
# =============================================================================

def save_sequences(
    output_dir: Path,
    variants: Optional[Dict[str, SequenceRecord]] = None
) -> None:
    """
    Save sequence records to files.

    Creates:
    - Individual FASTA files for each variant
    - A combined JSON file with all metadata

    Parameters
    ----------
    output_dir : Path
        Directory to save files
    variants : Optional[Dict[str, SequenceRecord]]
        Variants to save (default: all from registry)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if variants is None:
        variants = get_variant_registry()

    # Save individual FASTA files
    for name, record in variants.items():
        fasta_path = output_dir / f"{name}.fasta"
        with open(fasta_path, "w") as f:
            f.write(record.to_fasta())

    # Save combined JSON
    json_path = output_dir / "variants.json"
    data = {name: record.to_dict() for name, record in variants.items()}
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


def load_sequences(input_dir: Path) -> Dict[str, SequenceRecord]:
    """
    Load sequence records from JSON file.

    Parameters
    ----------
    input_dir : Path
        Directory containing variants.json

    Returns
    -------
    Dict[str, SequenceRecord]
        Loaded variant registry
    """
    json_path = Path(input_dir) / "variants.json"
    with open(json_path, "r") as f:
        data = json.load(f)

    return {
        name: SequenceRecord(**record_data)
        for name, record_data in data.items()
    }


# =============================================================================
# CONVENIENCE EXPORTS
# =============================================================================

# Pre-built variant registry for direct import
VARIANTS = get_variant_registry()

# Direct access to wild-type
WT = get_variant("WT")


if __name__ == "__main__":
    # Demo usage
    print("FUS LCD Sequence Analysis")
    print("=" * 60)

    for name, record in VARIANTS.items():
        print(f"\n{name}:")
        print(f"  Length: {record.length}")
        print(f"  Tyrosines: {record.tyrosine_count}")
        print(f"  Aromatics: {record.aromatic_count}")
        if record.mutations:
            print(f"  Mutations: {', '.join(record.mutations[:5])}...")

    print("\n\nWT Sequence Properties:")
    props = compute_sequence_properties(WT.sequence)
    for key, value in props.items():
        if key != "tyrosine_positions":
            print(f"  {key}: {value}")
