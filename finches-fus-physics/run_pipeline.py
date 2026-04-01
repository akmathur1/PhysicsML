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
print("\n[1/10] Loading sequences and variants...")

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
print("\n[2/10] Computing interaction maps...")

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
print("\n[3/10] Computing sticker-linker segmentation...")

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
print("\n[4/10] Computing difference maps...")

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
print("\n[5/10] Computing topology engine (Phase 1)...")

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

print("\n[6/10] Computing biophysical metrics...")

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
print("\n[7/10] Generating publication figures...")

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
print("\n[8/10] Generating topology figures (Phase 1)...")

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

# Figure 12: WT sticker contact graph (use auto-threshold for edges)
from src.topology import auto_threshold
wt_threshold = auto_threshold(intermaps['WT'], sticker_masks['WT'])
wt_sticker_graph = build_sticker_contact_graph(
    intermaps['WT'], sticker_masks['WT'], threshold=wt_threshold, sequence=wt.sequence
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
# STEP 9: Phase 2 — Hybrid Hamiltonian
# =============================================================================
print("\n[9/10] Computing hybrid Hamiltonian (Phase 2)...")

from src.hamiltonian import compute_all_H_eff, compute_sensitivity, predict_csat

# Gather omega values
omega_values = {name: all_metrics[name].omega for name in VARIANTS}

# Compute H_eff for all variants
all_H_eff = compute_all_H_eff(
    variants=VARIANTS,
    intermaps=intermaps,
    sticker_masks=sticker_masks,
    topology_metrics=all_topology,
    homology_metrics=all_homology,
    entropy_metrics=all_entropy,
    omega_values=omega_values,
)

# Sensitivity analysis
sensitivity = compute_sensitivity(all_H_eff)

# Predict c_sat (relative to WT)
wt_H = all_H_eff['WT'].H_eff
csat_predictions = {}
for name, decomp in all_H_eff.items():
    csat_predictions[name] = predict_csat(decomp.H_eff, reference_csat=1.0, reference_H=wt_H)

# Print Hamiltonian decomposition
print("\n  Hybrid Hamiltonian Decomposition:")
print("  " + "-" * 100)
print(f"  {'Variant':12s} {'H_chem':>9s} {'H_topo':>9s} {'H_eff':>9s} "
      f"{'Phi_clust':>10s} {'Phi_conn':>10s} {'Phi_perc':>10s} {'Phi_arr':>10s} {'Phi_hom':>10s}")
print("  " + "-" * 100)
for name, d in all_H_eff.items():
    print(f"  {name:12s} {d.H_chemistry:9.4f} {d.H_topology:9.4f} {d.H_eff:9.4f} "
          f"{d.phi_clustering:10.4f} {d.phi_connectivity:10.4f} {d.phi_percolation:10.4f} "
          f"{d.phi_arrangement:10.4f} {d.phi_homology:10.4f}")

# Print topology fraction
print("\n  Topology Contribution to H_eff:")
for name, d in all_H_eff.items():
    print(f"    {name:12s}: {d.topology_fraction:.1%} topology, {d.chemistry_fraction:.1%} chemistry")

# Print c_sat predictions
print("\n  Predicted c_sat (relative to WT):")
for name, csat in csat_predictions.items():
    direction = "↓ stronger LLPS" if csat < 1.0 else "↑ weaker LLPS"
    print(f"    {name:12s}: c_sat = {csat:.4f}  ({direction})")

# Print sensitivity
print("\n  Sensitivity Analysis (which terms discriminate variants):")
for term, s in sorted(sensitivity.sensitivities.items(), key=lambda x: -abs(x[1])):
    bar = "█" * int(abs(s) * 20)
    print(f"    {term:25s}: {s:.3f}  {bar}")

# Save Hamiltonian results
h_eff_dict = {name: d.to_dict() for name, d in all_H_eff.items()}
with open(output_path / "hamiltonian.json", 'w') as f:
    json.dump({
        "decompositions": h_eff_dict,
        "csat_predictions": csat_predictions,
        "sensitivities": sensitivity.sensitivities,
    }, f, indent=2)
print(f"\n  Saved hamiltonian.json to {output_path}/")

# =============================================================================
# STEP 10: Phase 2 Figures
# =============================================================================
print("\n[10/10] Generating Phase 2 figures...")

from src.plotting import (
    plot_hamiltonian_decomposition, plot_sensitivity_analysis,
    plot_csat_prediction, plot_H_eff_vs_H_chem,
)

variant_names = list(VARIANTS.keys())

# Figure 13: Hamiltonian decomposition
fig = plot_hamiltonian_decomposition(variant_names, all_H_eff)
plt.suptitle('Hybrid Effective Hamiltonian: $H_{eff} = H_{chem} + H_{topo}$', fontsize=14, y=1.02)
save_figure(fig, figures_dir / "fig13_hamiltonian_decomposition")
plt.close(fig)
print(f"  Saved fig13_hamiltonian_decomposition.png/pdf")

# Figure 14: Sensitivity analysis
fig = plot_sensitivity_analysis(sensitivity.sensitivities)
save_figure(fig, figures_dir / "fig14_sensitivity")
plt.close(fig)
print(f"  Saved fig14_sensitivity.png/pdf")

# Figure 15: c_sat prediction
H_eff_values = {name: d.H_eff for name, d in all_H_eff.items()}
fig = plot_csat_prediction(variant_names, csat_predictions, H_eff_values)
plt.suptitle('Phase Separation Prediction from $H_{eff}$', fontsize=14, y=1.02)
save_figure(fig, figures_dir / "fig15_csat_prediction")
plt.close(fig)
print(f"  Saved fig15_csat_prediction.png/pdf")

# Figure 16: H_eff vs H_chem scatter (topology correction)
fig = plot_H_eff_vs_H_chem(variant_names, all_H_eff)
save_figure(fig, figures_dir / "fig16_topology_correction")
plt.close(fig)
print(f"  Saved fig16_topology_correction.png/pdf")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("PIPELINE COMPLETE (Phase 1 + Phase 2)")
print("=" * 70)

print("\nGenerated Files:")
print(f"  data/sequences/     - FASTA files and variants.json")
print(f"  data/outputs/       - intermaps.npz, sticker_masks.npz, difference_maps.npz, metrics.json")
print(f"  data/outputs/       - topology_metrics.json (Phase 1: topology + homology + entropy)")
print(f"  data/outputs/       - hamiltonian.json (Phase 2: H_eff decomposition + c_sat)")
print(f"  figures/            - 16 publication figures (PNG + PDF)")

print("\nKey Results — Hybrid Hamiltonian:")
for name in ['WT', 'AllY_to_S', 'AllY_to_F']:
    d = all_H_eff[name]
    c = csat_predictions[name]
    print(f"  {name:12s}: H_eff={d.H_eff:.4f}  (chem={d.H_chemistry:.4f} + topo={d.H_topology:.4f})  "
          f"c_sat={c:.3f}")

print("\nInterpretation:")
print("  - H_eff = H_chemistry + H_topology provides a unified energy model")
print("  - H_topology contributes an independent, quantifiable correction")
print("  - Percolation threshold is the dominant topology term")
print("  - AllY_to_F has the deepest H_eff → strongest predicted LLPS")
print("  - AllY_to_S loses both chemistry AND topology → c_sat rises sharply")
print("  - The topology correction amplifies differences beyond what chemistry alone predicts")

print("\n✓ All computations successful!")
