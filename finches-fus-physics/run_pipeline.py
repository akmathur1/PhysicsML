#!/usr/bin/env python3
"""
run_pipeline.py - Execute the complete FUS LCD analysis pipeline.

This script runs all computations and generates outputs/figures.
"""

import sys
import json
import numpy as np
from pathlib import Path

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("FINCHES-FUS-Physics: Complete Pipeline Execution")
print("=" * 70)

# =============================================================================
# STEP 1: Sequences and Variants
# =============================================================================
print("\n[1/8] Loading sequences and variants...")

from src.sequences import (
    VARIANTS, FUS_LCD_SEQUENCE, get_variant,
    compute_sequence_properties, save_sequences
)

# Display sequence info
wt = get_variant("WT")
print(f"  FUS LCD sequence length: {len(FUS_LCD_SEQUENCE)} residues")
print(f"  Wild-type tyrosines: {wt.tyrosine_count}")

# Save sequences
output_dir = Path("data/sequences")
output_dir.mkdir(parents=True, exist_ok=True)
save_sequences(output_dir, VARIANTS)
print(f"  Saved {len(VARIANTS)} variants to {output_dir}/")

# Print variant summary
print("\n  Variant Registry:")
for name, variant in VARIANTS.items():
    print(f"    {name:12s}: {variant.tyrosine_count:2d} Tyr, {variant.aromatic_count:2d} aromatic")

# =============================================================================
# STEP 2: Force Field and Interaction Maps
# =============================================================================
print("\n[2/8] Computing interaction maps...")

from src.forcefield import get_default_matrix, get_most_attractive_pairs
from src.intermaps import (
    compute_all_intermaps, InterMapConfig,
    compute_map_statistics, save_intermaps
)

# Show force field info
matrix = get_default_matrix()
print(f"  Force field matrix shape: {matrix.shape}")
top_pairs = get_most_attractive_pairs(matrix, 3)
print(f"  Most attractive pairs: {[(p[0]+'-'+p[1], f'{p[2]:.3f}') for p in top_pairs]}")

# Compute intermaps (unnormalized for physical energies)
config = InterMapConfig(smooth=True, sigma=2.0, normalize=False)
intermaps = compute_all_intermaps(VARIANTS, config=config)

# Save intermaps
output_path = Path("data/outputs")
output_path.mkdir(parents=True, exist_ok=True)
save_intermaps(intermaps, str(output_path / "intermaps.npz"))
print(f"  Saved interaction maps to {output_path}/intermaps.npz")

# Print statistics
print("\n  Interaction Map Statistics:")
for name, imap in intermaps.items():
    stats = compute_map_statistics(imap)
    print(f"    {name:12s}: mean={stats['mean']:.4f}, min={stats['min']:.4f}, max={stats['max']:.4f}")

# =============================================================================
# STEP 3: Sticker-Linker Segmentation
# =============================================================================
print("\n[3/8] Computing sticker-linker segmentation...")

from src.segmentation import (
    compute_interaction_profile,
    identify_stickers_by_sequence, create_sticker_mask,
    compute_linker_statistics, build_sticker_linker_chain
)

sticker_masks = {}
chains = {}
profiles = {}

for name, variant in VARIANTS.items():
    # Compute profile
    profile = compute_interaction_profile(intermaps[name])
    profiles[name] = profile

    # Identify stickers based on sequence chemistry (aromatics + cations)
    # This directly identifies Y, F, W (aromatics) and R, K (cations) as stickers
    sticker_bool = identify_stickers_by_sequence(
        variant, include_aromatics=True, include_cations=True
    )
    sticker_masks[name] = create_sticker_mask(sticker_bool)
    chains[name] = build_sticker_linker_chain(variant, sticker_masks[name])

# Save sticker masks
mask_dict = {name: mask.mask for name, mask in sticker_masks.items()}
np.savez_compressed(output_path / "sticker_masks.npz", **mask_dict)
print(f"  Saved sticker masks to {output_path}/sticker_masks.npz")

# Print segmentation summary
print("\n  Sticker-Linker Summary:")
for name, mask in sticker_masks.items():
    stats = compute_linker_statistics(mask)
    print(f"    {name:12s}: {mask.n_stickers:2d} stickers ({mask.sticker_fraction:.1%}), "
          f"<L>={stats['mean_length']:.1f}")

# =============================================================================
# STEP 4: Difference Maps
# =============================================================================
print("\n[4/8] Computing difference maps...")

from src.intermaps import compute_all_difference_maps

diff_maps = compute_all_difference_maps(intermaps, reference_name='WT')

# Save difference maps
np.savez_compressed(output_path / "difference_maps.npz", **diff_maps)
print(f"  Saved difference maps to {output_path}/difference_maps.npz")

# Print difference statistics
print("\n  Difference Map Statistics (Variant - WT):")
for name, dmap in diff_maps.items():
    total_delta = np.sum(dmap) / 2
    rmsd = np.sqrt(np.mean(dmap**2))
    print(f"    {name:12s}: total_Δ={total_delta:+.1f}, RMSD={rmsd:.4f}")

# =============================================================================
# STEP 5: Biophysical Metrics
# =============================================================================
print("\n[5/8] Computing topology engine (Phase 1)...")

from src.topology import compute_topology_metrics, compute_percolation_sweep
from src.homology import compute_homology_metrics
from src.entropy import compute_entropy_metrics

all_topology = {}
all_homology = {}
all_entropy = {}
all_percolation = {}

for name in VARIANTS:
    variant = VARIANTS[name]
    seq = variant.sequence
    imap = intermaps[name]
    mask = sticker_masks[name]

    all_topology[name] = compute_topology_metrics(imap, mask, sequence=seq)
    all_homology[name] = compute_homology_metrics(imap, mask)
    all_entropy[name] = compute_entropy_metrics(imap, mask, seq)
    all_percolation[name] = compute_percolation_sweep(imap, mask)

# Print topology summary
print("\n  Topology Metrics:")
print("  " + "-" * 90)
print(f"  {'Variant':12s} {'Contacts':>9s} {'Density':>8s} {'Clust.C':>8s} {'<Deg>':>7s} {'Comp.':>6s} {'Perc.Thr':>9s}")
print("  " + "-" * 90)
for name, t in all_topology.items():
    print(f"  {name:12s} {t.sticker_n_contacts:9d} {t.sticker_graph_density:8.4f} "
          f"{t.sticker_clustering_coefficient:8.4f} {t.sticker_mean_degree:7.2f} "
          f"{t.sticker_n_components:6d} {t.percolation_threshold:9.4f}")

# Print homology summary
print("\n  Persistent Homology:")
print("  " + "-" * 80)
print(f"  {'Variant':12s} {'H0_total':>9s} {'H0_max':>8s} {'H1_feat':>8s} {'H1_total':>9s} {'B0_AUC':>8s} {'B1_AUC':>8s}")
print("  " + "-" * 80)
for name, h in all_homology.items():
    print(f"  {name:12s} {h.h0_total_persistence:9.3f} {h.h0_max_persistence:8.3f} "
          f"{h.h1_n_features:8d} {h.h1_total_persistence:9.3f} "
          f"{h.betti_0_auc:8.2f} {h.betti_1_auc:8.2f}")

# Print entropy summary
print("\n  Entropy Metrics:")
print("  " + "-" * 70)
print(f"  {'Variant':12s} {'Spacing':>9s} {'Norm.Sp':>8s} {'Block':>8s} {'Interact':>9s} {'Compos':>8s}")
print("  " + "-" * 70)
for name, e in all_entropy.items():
    print(f"  {name:12s} {e.spacing_entropy:9.3f} {e.normalized_spacing_entropy:8.3f} "
          f"{e.block_entropy:8.3f} {e.interaction_entropy:9.3f} {e.composition_entropy:8.3f}")

# Save topology data
import json as _json
topo_dict = {name: t.to_dict() for name, t in all_topology.items()}
homo_dict = {name: h.to_dict() for name, h in all_homology.items()}
entr_dict = {name: e.to_dict() for name, e in all_entropy.items()}
with open(output_path / "topology_metrics.json", 'w') as f:
    _json.dump({"topology": topo_dict, "homology": homo_dict, "entropy": entr_dict}, f, indent=2)
print(f"\n  Saved topology_metrics.json to {output_path}/")

print("\n[6/8] Computing biophysical metrics...")

from src.metrics import compute_all_metrics

all_metrics = {}
for name in VARIANTS:
    all_metrics[name] = compute_all_metrics(
        name, VARIANTS[name], intermaps[name], sticker_masks[name]
    )

# Save metrics as JSON
metrics_dict = {name: m.to_dict() for name, m in all_metrics.items()}
with open(output_path / "metrics.json", 'w') as f:
    json.dump(metrics_dict, f, indent=2)
print(f"  Saved metrics to {output_path}/metrics.json")

# Print metrics table
print("\n  Comprehensive Metrics Table:")
print("  " + "-" * 80)
print(f"  {'Variant':12s} {'n_Y':>5s} {'f_arom':>8s} {'f_stick':>8s} {'<L>':>6s} {'ΔG':>10s} {'E_ss':>10s}")
print("  " + "-" * 80)
for name, m in all_metrics.items():
    print(f"  {name:12s} {m.n_tyrosine:5d} {m.aromatic_fraction:8.3f} {m.sticker_fraction:8.3f} "
          f"{m.mean_linker_length:6.1f} {m.delta_G_proxy:10.4f} {m.sticker_energy:10.4f}")

# =============================================================================
# STEP 6: Generate Figures
# =============================================================================
print("\n[7/8] Generating publication figures...")

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from src.plotting import (
    plot_interaction_map, plot_difference_map,
    plot_sticker_linker_comparison, plot_profiles_comparison,
    save_figure, set_publication_style, COLOR_STICKER, COLOR_LINKER
)

set_publication_style()
figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)

# Figure 1: WT Interaction Map
fig = plot_interaction_map(
    intermaps['WT'],
    title="FUS LCD Wild-Type: Interaction Map",
    figsize=(8, 7)
)
save_figure(fig, figures_dir / "fig1_wt_intermap")
plt.close(fig)
print(f"  Saved fig1_wt_intermap.png/pdf")

# Figure 2: Interaction Map Comparison (WT vs AllY_to_S vs AllY_to_F)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
abs_max = max(abs(intermaps['WT'].min()), abs(intermaps['WT'].max()),
              abs(intermaps['AllY_to_S'].min()), abs(intermaps['AllY_to_S'].max()))

for ax, name, title in zip(axes, ['WT', 'AllY_to_S', 'AllY_to_F'], ['A. WT', 'B. All Y→S', 'C. All Y→F']):
    im = ax.imshow(intermaps[name], cmap='RdBu_r', vmin=-abs_max, vmax=abs_max)
    ax.set_title(title)
    ax.set_xlabel('Residue')
    ax.set_ylabel('Residue')

fig.colorbar(im, ax=axes, shrink=0.8, label='Energy (kT)')
plt.tight_layout()
save_figure(fig, figures_dir / "fig2_intermap_comparison")
plt.close(fig)
print(f"  Saved fig2_intermap_comparison.png/pdf")

# Figure 3: Difference Map (AllY_to_S - WT)
fig = plot_difference_map(
    diff_maps['AllY_to_S'],
    title="Difference Map: AllY_to_S − WT\n(Green = loss of attraction)",
    figsize=(8, 7)
)
save_figure(fig, figures_dir / "fig3_difference_map")
plt.close(fig)
print(f"  Saved fig3_difference_map.png/pdf")

# Figure 4: Sticker-Linker Segmentation
fig, axes = plt.subplots(5, 1, figsize=(14, 8), sharex=True)
for ax, (name, mask) in zip(axes, sticker_masks.items()):
    n = len(mask.mask)
    for i in range(n):
        color = COLOR_STICKER if mask.mask[i] else COLOR_LINKER
        ax.axvspan(i, i+1, facecolor=color, edgecolor='none')
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    m = all_metrics[name]
    ax.set_ylabel(name, rotation=0, ha='right', va='center')
    ax.set_title(f"f_sticker = {m.sticker_fraction:.2f}", loc='right', fontsize=10)

axes[-1].set_xlabel('Residue Position')
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=COLOR_STICKER, label='Sticker'),
                   Patch(facecolor=COLOR_LINKER, label='Linker')]
fig.legend(handles=legend_elements, loc='upper right')
plt.suptitle('Sticker-Linker Segmentation', fontsize=14)
plt.tight_layout()
save_figure(fig, figures_dir / "fig4_segmentation")
plt.close(fig)
print(f"  Saved fig4_segmentation.png/pdf")

# Figure 5: Interaction Profiles
fig, ax = plt.subplots(figsize=(14, 5))
colors = {'WT': '#2C3E50', 'AllY_to_S': '#E74C3C', 'AllY_to_F': '#3498DB'}
for name in ['WT', 'AllY_to_S', 'AllY_to_F']:
    x = np.arange(len(profiles[name])) + 1
    ax.plot(x, profiles[name], label=name, color=colors[name], linewidth=2)

# Mark tyrosine positions
for pos in wt.tyrosine_positions:
    ax.axvline(x=pos+1, color='#F39C12', alpha=0.2, linewidth=1)

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('Residue Position')
ax.set_ylabel('Mean Interaction Energy (kT)')
ax.set_title('Interaction Profiles (Orange = WT Tyr positions)')
ax.legend()
ax.set_xlim(1, len(profiles['WT']))
plt.tight_layout()
save_figure(fig, figures_dir / "fig5_profiles")
plt.close(fig)
print(f"  Saved fig5_profiles.png/pdf")

# Figure 6: Metrics Bar Charts
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
names = list(all_metrics.keys())
x = np.arange(len(names))

# Sticker fraction
axes[0,0].bar(x, [all_metrics[n].sticker_fraction for n in names], color='#E74C3C', edgecolor='black')
axes[0,0].set_xticks(x)
axes[0,0].set_xticklabels(names, rotation=45, ha='right')
axes[0,0].set_ylabel('Sticker Fraction')
axes[0,0].set_title('A. Sticker Fraction')

# Aromatic fraction
axes[0,1].bar(x, [all_metrics[n].aromatic_fraction for n in names], color='#9B59B6', edgecolor='black')
axes[0,1].set_xticks(x)
axes[0,1].set_xticklabels(names, rotation=45, ha='right')
axes[0,1].set_ylabel('Aromatic Fraction')
axes[0,1].set_title('B. Aromatic Content')

# ΔG proxy
axes[1,0].bar(x, [all_metrics[n].delta_G_proxy for n in names], color='#3498DB', edgecolor='black')
axes[1,0].set_xticks(x)
axes[1,0].set_xticklabels(names, rotation=45, ha='right')
axes[1,0].set_ylabel('ΔG Proxy (kT)')
axes[1,0].set_title('C. Phase Separation Driving Force')
axes[1,0].axhline(y=0, color='black', linewidth=0.5)

# Sticker energy
axes[1,1].bar(x, [all_metrics[n].sticker_energy for n in names], color='#2ECC71', edgecolor='black')
axes[1,1].set_xticks(x)
axes[1,1].set_xticklabels(names, rotation=45, ha='right')
axes[1,1].set_ylabel('E_sticker-sticker (kT)')
axes[1,1].set_title('D. Sticker Interaction Strength')
axes[1,1].axhline(y=0, color='black', linewidth=0.5)

plt.tight_layout()
save_figure(fig, figures_dir / "fig6_metrics")
plt.close(fig)
print(f"  Saved fig6_metrics.png/pdf")

# =============================================================================
# STEP 8: Topology Engine Figures (Phase 1)
# =============================================================================
print("\n[8/8] Generating topology figures (Phase 1)...")

from src.plotting import (
    plot_topology_summary, plot_persistence_diagram,
    plot_entropy_comparison, plot_contact_graph,
)
from src.topology import build_sticker_contact_graph
from src.homology import interaction_to_distance, compute_H0_persistence, compute_H1_persistence, compute_betti_numbers

# Figure 7: Topology Summary (multi-panel bar chart)
fig = plot_topology_summary(
    list(all_topology.keys()),
    all_topology,
)
save_figure(fig, figures_dir / "fig7_topology_summary")
plt.close(fig)
print(f"  Saved fig7_topology_summary.png/pdf")

# Figure 8: Percolation curves (WT vs AllY_to_S vs AllY_to_F)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
perc_colors = {'WT': '#2C3E50', 'AllY_to_S': '#E74C3C', 'AllY_to_F': '#3498DB'}
for ax, name in zip(axes, ['WT', 'AllY_to_S', 'AllY_to_F']):
    perc = all_percolation[name]
    ax.plot(perc.thresholds, perc.fraction_connected, color=perc_colors[name], linewidth=2)
    ax.axvline(x=perc.percolation_threshold, color='gray', linestyle='--', alpha=0.7)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
    ax.set_xlabel('Energy Threshold (kT)')
    ax.set_ylabel('Fraction in Giant Component')
    ax.set_title(f'{name}\n(perc. thr. = {perc.percolation_threshold:.3f})')
    ax.set_ylim(-0.05, 1.05)
plt.suptitle('Percolation Analysis: Connectivity Transition', fontsize=14)
plt.tight_layout()
save_figure(fig, figures_dir / "fig8_percolation")
plt.close(fig)
print(f"  Saved fig8_percolation.png/pdf")

# Figure 9: Persistence diagrams (WT)
wt_dist = interaction_to_distance(intermaps['WT'], sticker_masks['WT'])
wt_h0 = compute_H0_persistence(wt_dist)
wt_h1 = compute_H1_persistence(wt_dist)
fig = plot_persistence_diagram(
    wt_h0.pairs, wt_h1.pairs,
    title="WT FUS LCD: Persistence Diagram"
)
save_figure(fig, figures_dir / "fig9_persistence_diagram")
plt.close(fig)
print(f"  Saved fig9_persistence_diagram.png/pdf")

# Figure 10: Betti curves comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, name in zip(axes, ['WT', 'AllY_to_S', 'AllY_to_F']):
    dist_mat = interaction_to_distance(intermaps[name], sticker_masks[name])
    betti = compute_betti_numbers(dist_mat)
    ax.plot(betti["thresholds"], betti["betti_0"], color='#3498DB', linewidth=2, label='$\\beta_0$')
    ax.fill_between(betti["thresholds"], betti["betti_0"], alpha=0.1, color='#3498DB')
    ax.plot(betti["thresholds"], betti["betti_1"], color='#E74C3C', linewidth=2, label='$\\beta_1$')
    ax.fill_between(betti["thresholds"], betti["betti_1"], alpha=0.1, color='#E74C3C')
    ax.set_xlabel('Filtration Threshold')
    ax.set_ylabel('Betti Number')
    ax.set_title(name)
    ax.legend(fontsize=9)
plt.suptitle('Betti Number Curves', fontsize=14)
plt.tight_layout()
save_figure(fig, figures_dir / "fig10_betti_curves")
plt.close(fig)
print(f"  Saved fig10_betti_curves.png/pdf")

# Figure 11: Entropy comparison
fig = plot_entropy_comparison(
    list(all_entropy.keys()),
    all_entropy,
)
plt.suptitle('Sticker Arrangement Entropy', fontsize=14)
plt.tight_layout()
save_figure(fig, figures_dir / "fig11_entropy")
plt.close(fig)
print(f"  Saved fig11_entropy.png/pdf")

# Figure 12: WT sticker contact graph
wt_sticker_graph = build_sticker_contact_graph(
    intermaps['WT'], sticker_masks['WT'], sequence=wt.sequence
)
fig = plot_contact_graph(
    wt_sticker_graph.adjacency,
    sticker_positions=sticker_masks['WT'].positions,
    node_labels=wt_sticker_graph.node_labels,
    title=f"WT Sticker Contact Graph ({wt_sticker_graph.n_edges} edges, {wt_sticker_graph.n_nodes} nodes)"
)
save_figure(fig, figures_dir / "fig12_contact_graph")
plt.close(fig)
print(f"  Saved fig12_contact_graph.png/pdf")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("PIPELINE COMPLETE (Phase 1: Topology Engine)")
print("=" * 70)

print("\nGenerated Files:")
print(f"  data/sequences/     - FASTA files and variants.json")
print(f"  data/outputs/       - intermaps.npz, sticker_masks.npz, difference_maps.npz, metrics.json")
print(f"  data/outputs/       - topology_metrics.json (NEW: topology + homology + entropy)")
print(f"  figures/            - 12 publication figures (PNG + PDF)")

print("\nKey Results (Original):")
print(f"  WT:         {all_metrics['WT'].n_stickers:2d} stickers, f={all_metrics['WT'].sticker_fraction:.2f}, ΔG={all_metrics['WT'].delta_G_proxy:.4f}")
print(f"  AllY_to_S:  {all_metrics['AllY_to_S'].n_stickers:2d} stickers, f={all_metrics['AllY_to_S'].sticker_fraction:.2f}, ΔG={all_metrics['AllY_to_S'].delta_G_proxy:.4f}")
print(f"  AllY_to_F:  {all_metrics['AllY_to_F'].n_stickers:2d} stickers, f={all_metrics['AllY_to_F'].sticker_fraction:.2f}, ΔG={all_metrics['AllY_to_F'].delta_G_proxy:.4f}")

print("\nKey Results (Phase 1 — Topology Engine):")
for name in ['WT', 'AllY_to_S', 'AllY_to_F']:
    t = all_topology[name]
    h = all_homology[name]
    e = all_entropy[name]
    print(f"  {name:12s}: clust={t.sticker_clustering_coefficient:.3f}, "
          f"perc_thr={t.percolation_threshold:.3f}, "
          f"H1_cycles={h.h1_n_features}, "
          f"spacing_H={e.normalized_spacing_entropy:.3f}")

print("\nInterpretation:")
print("  - WT FUS LCD has ~30 stickers (tyrosines + arginines) driving phase separation")
print("  - Y→S mutations abolish sticker character → weakens phase separation")
print("  - Y→F mutations preserve aromatic character → maintains phase separation")
print("  - Aromatic (π-π) interactions are the key driver, not tyrosine-specific chemistry")
print("  Phase 1 additions:")
print("  - Sticker network topology captures emergent structure beyond pairwise energetics")
print("  - Percolation threshold maps connectivity onset → LLPS propensity indicator")
print("  - Persistent homology reveals global network features (cycles, robustness)")
print("  - Spacing entropy quantifies sticker arrangement regularity")

print("\n✓ All computations successful!")
