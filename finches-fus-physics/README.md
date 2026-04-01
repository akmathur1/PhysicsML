# FINCHES-FUS-Physics

**Physics-based analysis pipeline for FUS low-complexity domain (LCD) phase separation**

This repository implements a FINCHES/MPIPI-GG style computational workflow for analyzing the sequence determinants of FUS LCD phase separation behavior.

## Overview

The FUS protein contains an N-terminal low-complexity domain (LCD, residues 1-214) that drives liquid-liquid phase separation (LLPS). This pipeline provides tools to:

1. **Generate interaction maps** based on MPIPI-GG style coarse-grained force fields
2. **Identify sticker and linker regions** through energetic and sequence-based criteria
3. **Compute biophysical metrics** that correlate with phase separation propensity
4. **Visualize mutation effects** through difference maps and comparative analysis

## Workflow

```
Sequence → FINCHES-style intermaps → Sliding-window smoothing →
Sticker segmentation → Difference maps → Coarse-grained metrics
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/finches-fus-physics.git
cd finches-fus-physics

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Add src to Python path (for notebook imports)
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Project Structure

```
finches-fus-physics/
├── notebooks/
│   ├── 01_sequences_and_mutations.ipynb    # Sequence handling and variants
│   ├── 02_forcefield_and_intermaps.ipynb   # Force field and interaction maps
│   ├── 03_sliding_window_and_sticker_masks.ipynb  # Sticker identification
│   ├── 04_difference_maps_and_profiles.ipynb      # Mutation effect analysis
│   ├── 05_sticker_linker_metrics.ipynb     # Biophysical metrics
│   └── 06_minimal_model_and_figures.ipynb  # Publication figures
│
├── src/
│   ├── __init__.py
│   ├── sequences.py      # FUS LCD sequence and mutation utilities
│   ├── forcefield.py     # MPIPI-GG style residue-residue interactions
│   ├── intermaps.py      # Interaction map generation and processing
│   ├── segmentation.py   # Sticker-linker segmentation
│   ├── metrics.py        # Biophysical metrics computation
│   └── plotting.py       # Publication-quality visualization
│
├── data/
│   ├── sequences/        # FASTA files and variant registry
│   └── outputs/          # Computed intermaps, metrics, etc.
│
├── figures/              # Generated publication figures
├── requirements.txt
└── README.md
```

## Variant Registry

The pipeline includes 5 pre-defined FUS LCD variants:

| Variant | Description | Tyrosines | Purpose |
|---------|-------------|-----------|---------|
| WT | Wild-type FUS LCD | 27 | Reference |
| Y144S | Single Y→S mutation | 26 | Minimal perturbation |
| AllY_to_S | All Y→S mutations | 0 | Sticker disruption |
| AllY_to_A | All Y→A mutations | 0 | Aromatic knockout |
| AllY_to_F | All Y→F mutations | 0 (27 F) | Conservative control |

## Usage

### Quick Start

```python
from src.sequences import VARIANTS, get_variant
from src.intermaps import compute_all_intermaps, InterMapConfig
from src.segmentation import identify_stickers_hybrid, create_sticker_mask
from src.metrics import compute_all_metrics

# Load variants
wt = get_variant("WT")
print(f"WT FUS LCD: {wt.length} residues, {wt.tyrosine_count} tyrosines")

# Generate interaction maps
config = InterMapConfig(smooth=True, sigma=2.0, normalize=False)
intermaps = compute_all_intermaps(VARIANTS, config=config)

# Compute sticker masks
from src.intermaps import compute_interaction_profile
profile = compute_interaction_profile(intermaps['WT'])
sticker_bool = identify_stickers_hybrid(profile, wt, energy_threshold=-0.15)
sticker_mask = create_sticker_mask(sticker_bool)

print(f"Sticker fraction: {sticker_mask.sticker_fraction:.2%}")
```

### Running Notebooks

```bash
cd notebooks
jupyter lab
```

Execute notebooks in order (01 → 06) for the complete analysis pipeline.

## Key Concepts

### Force Field

The MPIPI-GG style force field captures key interactions:

- **Aromatic-aromatic (π-π)**: ~-0.65 kT for Y-Y, F-F, W-W
- **Cation-π**: ~-0.50 kT for R/K with aromatics
- **Electrostatic**: ~-0.30 kT for opposite charges
- **Hydrophobic**: ~-0.15 kT for nonpolar contacts

### Sticker-Linker Model

- **Stickers**: Residues with strong attractive interactions (aromatics, cations)
- **Linkers**: Flexible spacers between stickers
- **Sticker fraction** correlates with phase separation propensity

### Metrics

| Metric | Symbol | Description |
|--------|--------|-------------|
| Sticker fraction | f_sticker | Fraction of residues classified as stickers |
| ΔG proxy | ΔG | Weighted sum of interaction energies |
| Kappa | κ | Linker length uniformity |
| Omega | Ω | Sticker spacing uniformity |
| SCD | - | Sequence charge decoration |

## Output Files

- `data/outputs/intermaps.npz`: Computed interaction maps
- `data/outputs/sticker_masks.npz`: Boolean sticker masks
- `data/outputs/difference_maps.npz`: Variant - WT difference maps
- `data/outputs/metrics.json`: Comprehensive metrics table
- `figures/*.png, *.pdf`: Publication-quality figures

## References

This implementation is inspired by:

1. **FINCHES**: Holehouse lab sequence analysis tools
2. **MPIPI**: Dignon et al. (2018) coarse-grained model for IDPs
3. **Sticker-linker theory**: Choi, Holehouse, Pappu (2020)

## Citation

If you use this code, please cite:

```
@software{finches_fus_physics,
  author = {Mathur, Arjun},
  title = {FINCHES-FUS-Physics: Physics-based FUS LCD analysis},
  year = {2024},
  url = {https://github.com/your-username/finches-fus-physics}
}
```

## License

MIT License - see LICENSE file for details.

## Contact

For questions or collaboration, contact: [your-email@example.com]
