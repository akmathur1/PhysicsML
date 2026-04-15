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
print("\n[1/14] Loading sequences and variants...")

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
print("\n[2/14] Computing interaction maps...")

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
print("\n[3/14] Computing sticker-linker segmentation...")

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
print("\n[4/14] Computing difference maps...")

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
print("\n[5/14] Computing topology engine (Phase 1)...")

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

print("\n[6/14] Computing biophysical metrics...")

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
print("\n[7/14] Generating publication figures...")

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
    n = intermaps[name].shape[0]
    im = ax.imshow(intermaps[name], cmap='RdBu_r', vmin=-abs_max, vmax=abs_max, origin='lower')
    ax.set_title(title)
    ax.set_xlabel('Residue')
    ax.set_ylabel('Residue')
    tick_interval = max(1, n // 10) * 10
    ticks = np.arange(0, n, tick_interval)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks + 1)
    ax.set_yticklabels(ticks + 1)

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
print("\n[8/14] Generating topology figures (Phase 1)...")

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
# STEP 9: Expanded Variant Library (Phase 2)
# =============================================================================
print("\n[9/14] Building expanded variant library (Phase 2)...")

from src.variants import build_expanded_registry, get_experimental_data
from src.hamiltonian import compute_all_H_eff, compute_sensitivity, predict_csat
from src.calibration import (
    CalibrationInput, calibrate_coupling_constants,
    compute_robustness, compare_models,
)

# Build expanded registry (original 5 + ~15 new)
EXPANDED = build_expanded_registry(n_single_sites=6, n_shuffles=3)
print(f"  Expanded variant library: {len(EXPANDED)} variants")

# List new variants
original_names = set(VARIANTS.keys())
new_names = sorted(set(EXPANDED.keys()) - original_names)
print(f"  New variants: {', '.join(new_names)}")

# Compute interaction maps, sticker masks, topology, homology, entropy for ALL expanded variants
from src.intermaps import generate_finches_intermap
from src.segmentation import identify_stickers_by_sequence, create_sticker_mask
from src.topology import compute_topology_metrics
from src.homology import compute_homology_metrics
from src.entropy import compute_entropy_metrics
from src.metrics import compute_omega

exp_intermaps = {}
exp_sticker_masks = {}
exp_topology = {}
exp_homology = {}
exp_entropy = {}
exp_omega = {}

config_exp = InterMapConfig(smooth=True, sigma=2.0, normalize=False)
for name, variant in EXPANDED.items():
    if name in VARIANTS:
        # Reuse already-computed data for original 5
        exp_intermaps[name] = intermaps[name]
        exp_sticker_masks[name] = sticker_masks[name]
        exp_topology[name] = all_topology[name]
        exp_homology[name] = all_homology[name]
        exp_entropy[name] = all_entropy[name]
        exp_omega[name] = all_metrics[name].omega
    else:
        imap = generate_finches_intermap(variant, config=config_exp)
        exp_intermaps[name] = imap

        sticker_bool = identify_stickers_by_sequence(
            variant, include_aromatics=True, include_cations=True
        )
        mask = create_sticker_mask(sticker_bool)
        exp_sticker_masks[name] = mask

        seq = variant.sequence
        exp_topology[name] = compute_topology_metrics(imap, mask, sequence=seq)
        exp_homology[name] = compute_homology_metrics(imap, mask)
        exp_entropy[name] = compute_entropy_metrics(imap, mask, seq)
        exp_omega[name] = compute_omega(mask)

print(f"  Computed interaction maps + topology for {len(EXPANDED)} variants")

# =============================================================================
# STEP 10: Hamiltonian + Calibration (Phase 2)
# =============================================================================
print("\n[10/14] Computing H_eff + calibration (Phase 2)...")

# Compute H_eff for ALL expanded variants (with default params first)
all_H_eff = compute_all_H_eff(
    variants=EXPANDED,
    intermaps=exp_intermaps,
    sticker_masks=exp_sticker_masks,
    topology_metrics=exp_topology,
    homology_metrics=exp_homology,
    entropy_metrics=exp_entropy,
    omega_values=exp_omega,
)

# Print H_eff for all variants
print("\n  H_eff Decomposition (default params, all variants):")
print("  " + "-" * 65)
print(f"  {'Variant':16s} {'H_chem':>9s} {'H_topo':>9s} {'H_eff':>9s} {'Topo%':>7s}")
print("  " + "-" * 65)
for name in EXPANDED:
    d = all_H_eff[name]
    print(f"  {name:16s} {d.H_chemistry:9.4f} {d.H_topology:9.4f} {d.H_eff:9.4f} {d.topology_fraction:6.1%}")

# --- Calibration against experimental c_sat ---
exp_data = get_experimental_data()
calibration_inputs = []
for name, edata in exp_data.items():
    if name in EXPANDED and edata.csat_relative is not None:
        calibration_inputs.append(CalibrationInput(
            name=name,
            intermap=exp_intermaps[name],
            sticker_mask=exp_sticker_masks[name],
            topology=exp_topology[name],
            homology=exp_homology[name],
            entropy=exp_entropy[name],
            omega=exp_omega[name],
            csat_experimental=edata.csat_relative,
        ))

print(f"\n  Calibrating against {len(calibration_inputs)} variants with experimental c_sat...")
print(f"  Calibration set: {[ci.name for ci in calibration_inputs]}")

cal_result = calibrate_coupling_constants(calibration_inputs, n_restarts=10)

print(f"\n  Calibration Results:")
print(f"    Loss: {cal_result.loss_initial:.4f} → {cal_result.loss_final:.4f}")
print(f"    Rank corr: {cal_result.rank_correlation_initial:.3f} → {cal_result.rank_correlation_final:.3f}")
print(f"    Optimal coupling constants:")
p = cal_result.optimal_params
print(f"      alpha_clustering:   {p.alpha_clustering:.4f}")
print(f"      alpha_connectivity: {p.alpha_connectivity:.4f}")
print(f"      alpha_percolation:  {p.alpha_percolation:.4f}")
print(f"      alpha_arrangement:  {p.alpha_arrangement:.4f}")
print(f"      alpha_homology:     {p.alpha_homology:.4f}")

# Re-compute H_eff with calibrated params
all_H_eff_cal = compute_all_H_eff(
    variants=EXPANDED,
    intermaps=exp_intermaps,
    sticker_masks=exp_sticker_masks,
    topology_metrics=exp_topology,
    homology_metrics=exp_homology,
    entropy_metrics=exp_entropy,
    omega_values=exp_omega,
    params=cal_result.optimal_params,
)

# Predict c_sat with calibrated H_eff
wt_H_cal = all_H_eff_cal['WT'].H_eff
csat_calibrated = {}
for name, decomp in all_H_eff_cal.items():
    csat_calibrated[name] = predict_csat(decomp.H_eff, reference_csat=1.0, reference_H=wt_H_cal)

print("\n  Calibrated c_sat predictions:")
print(f"  {'Variant':16s} {'c_sat_pred':>11s} {'c_sat_exp':>10s} {'H_eff':>9s}")
print("  " + "-" * 50)
for name in EXPANDED:
    cpred = csat_calibrated[name]
    cexp = exp_data[name].csat_relative if name in exp_data else None
    cexp_str = f"{cexp:.2f}" if cexp is not None else "—"
    print(f"  {name:16s} {cpred:11.4f} {cexp_str:>10s} {all_H_eff_cal[name].H_eff:9.4f}")

# Sensitivity analysis with calibrated params
sensitivity_cal = compute_sensitivity(all_H_eff_cal)

# --- Model comparison: H_chem-only vs H_eff ---
comparison = compare_models(calibration_inputs, params=cal_result.optimal_params)
print(f"\n  Model Comparison (chemistry-only vs H_eff):")
print(f"    Rank correlation: H_chem={comparison.rank_corr_chem_only:.3f}, H_eff={comparison.rank_corr_H_eff:.3f}")
print(f"    RMSE(log c_sat): H_chem={comparison.rmse_log_chem_only:.3f}, H_eff={comparison.rmse_log_H_eff:.3f}")
improvement = comparison.rank_corr_H_eff - comparison.rank_corr_chem_only
if improvement > 0:
    print(f"    → H_eff improves rank correlation by {improvement:.3f}")

# --- Robustness analysis ---
print("\n  Parameter robustness sweep...")
robustness = compute_robustness(calibration_inputs, base_params=cal_result.optimal_params)
for pname, stable in robustness.rank_stable.items():
    status = "STABLE" if stable else "SENSITIVE"
    print(f"    {pname:25s}: {status}")

# =============================================================================
# STEP 11: Multi-Task Learning (MTL) — addresses overfitting
# =============================================================================
print("\n[11/14] Training multi-task network (MTL)...")

from src.mtl_model import (
    extract_features, build_training_data, train_and_predict,
    MTLConfig,
)
from src.metrics import compute_all_metrics

# Extract features for all expanded variants
mtl_features = {}
mtl_h_chem = {}
for name in EXPANDED:
    variant = EXPANDED[name]
    # Compute quick metrics for feature extraction
    if name in all_metrics:
        m = all_metrics[name]
    else:
        m = compute_all_metrics(name, variant, exp_intermaps[name], exp_sticker_masks[name],
                                include_topology=False)
    mtl_features[name] = extract_features(m, exp_topology[name], exp_homology[name], exp_entropy[name])
    # H_chemistry from the Hamiltonian decomposition (available for ALL variants)
    mtl_h_chem[name] = all_H_eff[name].H_chemistry

# Experimental c_sat (sparse: only 3 variants)
mtl_csat_exp = {}
for name, edata in exp_data.items():
    if name in EXPANDED and edata.csat_relative is not None:
        mtl_csat_exp[name] = edata.csat_relative

# Phase separation labels (dense: ALL 20 variants get a label)
# Based on: does the variant have meaningful sticker content?
mtl_phase_labels = {}
for name in EXPANDED:
    variant = EXPANDED[name]
    n_stickers = exp_sticker_masks[name].n_stickers
    if name in exp_data and exp_data[name].phase_separates is not None:
        mtl_phase_labels[name] = exp_data[name].phase_separates
    else:
        # Heuristic: phase separates if it has >5 stickers
        mtl_phase_labels[name] = n_stickers > 5

# Build training data
training_data = build_training_data(
    variant_names=list(EXPANDED.keys()),
    features=mtl_features,
    h_chem_values=mtl_h_chem,
    csat_experimental=mtl_csat_exp,
    phase_labels=mtl_phase_labels,
)

print(f"  Training samples: {len(training_data)}")
print(f"  c_sat labels: {sum(1 for s in training_data if s.csat_target is not None)}")
print(f"  Phase labels: {sum(1 for s in training_data if s.phase_target is not None)}")
print(f"  H_chem labels: {sum(1 for s in training_data if s.h_chem_target is not None)}")

# Train MTL network
mtl_config = MTLConfig(
    hidden_1=16, hidden_2=8,
    lr=0.005, n_epochs=3000,
    lambda_h_eff=0.0,    # no H_eff ground truth — this is what we're predicting
    lambda_csat=1.0,      # sparse c_sat supervision
    lambda_phase=0.5,     # dense phase classification
    lambda_h_chem=0.5,    # dense physics anchor
    seed=42,
)

mtl_net, mtl_result = train_and_predict(training_data, config=mtl_config)

print(f"\n  MTL Training: final loss = {mtl_result.training_history[-1]['loss']:.4f}")

# Print MTL predictions
print("\n  MTL Predictions:")
print(f"  {'Variant':16s} {'c_sat_MTL':>10s} {'c_sat_exp':>10s} {'Phase_P':>8s} {'H_chem_pred':>12s} {'H_chem_true':>12s}")
print("  " + "-" * 75)
for name in EXPANDED:
    pred = mtl_result.predictions[name]
    cexp = mtl_csat_exp.get(name)
    cexp_str = f"{cexp:.2f}" if cexp is not None else "—"
    hc_true = mtl_h_chem[name]
    print(f"  {name:16s} {pred['csat_relative']:10.3f} {cexp_str:>10s} {pred['phase_prob']:8.3f} "
          f"{pred['H_chemistry']:12.4f} {hc_true:12.4f}")

# Compare MTL c_sat accuracy vs Nelder-Mead on the 3 labeled variants
print("\n  MTL vs Nelder-Mead on experimental c_sat:")
for name in mtl_csat_exp:
    c_exp = mtl_csat_exp[name]
    c_nm = csat_calibrated[name]
    c_mtl = mtl_result.predictions[name]['csat_relative']
    print(f"    {name:12s}: exp={c_exp:.2f}  NM={c_nm:.3f}  MTL={c_mtl:.3f}")

# Save all Phase 2 results
phase2_results = {
    "calibration": cal_result.to_dict(),
    "model_comparison": comparison.to_dict(),
    "robustness": robustness.to_dict(),
    "H_eff_calibrated": {name: d.to_dict() for name, d in all_H_eff_cal.items()},
    "csat_calibrated": csat_calibrated,
    "sensitivity_calibrated": sensitivity_cal.sensitivities,
    "mtl": mtl_result.to_dict(),
}
with open(output_path / "phase2_results.json", 'w') as f:
    json.dump(phase2_results, f, indent=2, default=str)
print(f"\n  Saved phase2_results.json to {output_path}/")

# =============================================================================
# STEP 12: Phase 2 Figures (original 5 variants)
# =============================================================================
print("\n[12/14] Generating Phase 2 figures...")

from src.plotting import (
    plot_hamiltonian_decomposition, plot_sensitivity_analysis,
    plot_csat_prediction, plot_H_eff_vs_H_chem,
)

variant_names_orig = list(VARIANTS.keys())

# Figure 13: Hamiltonian decomposition (original 5, calibrated)
H_eff_orig = {n: all_H_eff_cal[n] for n in variant_names_orig}
fig = plot_hamiltonian_decomposition(variant_names_orig, H_eff_orig)
plt.suptitle('Hybrid Effective Hamiltonian (calibrated)', fontsize=14, y=1.02)
save_figure(fig, figures_dir / "fig13_hamiltonian_decomposition")
plt.close(fig)
print(f"  Saved fig13_hamiltonian_decomposition.png/pdf")

# Figure 14: Sensitivity analysis (calibrated, all expanded variants)
fig = plot_sensitivity_analysis(sensitivity_cal.sensitivities)
save_figure(fig, figures_dir / "fig14_sensitivity")
plt.close(fig)
print(f"  Saved fig14_sensitivity.png/pdf")

# Figure 15: c_sat prediction vs experimental
H_eff_values_orig = {name: all_H_eff_cal[name].H_eff for name in variant_names_orig}
csat_orig = {n: csat_calibrated[n] for n in variant_names_orig}
fig = plot_csat_prediction(variant_names_orig, csat_orig, H_eff_values_orig)
plt.suptitle('Phase Separation Prediction (calibrated)', fontsize=14, y=1.02)
save_figure(fig, figures_dir / "fig15_csat_prediction")
plt.close(fig)
print(f"  Saved fig15_csat_prediction.png/pdf")

# Figure 16: H_eff vs H_chem scatter (calibrated)
fig = plot_H_eff_vs_H_chem(variant_names_orig, H_eff_orig)
save_figure(fig, figures_dir / "fig16_topology_correction")
plt.close(fig)
print(f"  Saved fig16_topology_correction.png/pdf")

# =============================================================================
# STEP 13: Expanded Variant Figures
# =============================================================================
print("\n[13/14] Generating expanded variant figures...")

# Figure 17: Shuffled vs WT — same composition, different topology
shuffled_names = [n for n in EXPANDED if n.startswith("Shuffled")] + ["WT"]
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(shuffled_names))
h_chem_vals = [all_H_eff_cal[n].H_chemistry for n in shuffled_names]
h_topo_vals = [all_H_eff_cal[n].H_topology for n in shuffled_names]
width = 0.35
ax.bar(x - width/2, h_chem_vals, width, label='$H_{chemistry}$', color='#3498DB', edgecolor='black')
ax.bar(x + width/2, h_topo_vals, width, label='$H_{topology}$', color='#E74C3C', edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels(shuffled_names, rotation=45, ha='right')
ax.set_ylabel('Energy (kT)')
ax.set_title('Shuffled vs WT: Same Composition, Different Topology')
ax.legend()
ax.axhline(y=0, color='black', linewidth=0.5)
plt.tight_layout()
save_figure(fig, figures_dir / "fig17_shuffled_comparison")
plt.close(fig)
print(f"  Saved fig17_shuffled_comparison.png/pdf")

# Figure 18: Progressive Y→S titration
prog_names = ["WT"] + [n for n in EXPANDED if n.endswith("Y_to_S") and n[0].isdigit()]
prog_names = sorted(prog_names, key=lambda n: all_H_eff_cal[n].H_eff, reverse=False)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
x = np.arange(len(prog_names))
h_effs = [all_H_eff_cal[n].H_eff for n in prog_names]
csats = [csat_calibrated[n] for n in prog_names]
ax1.bar(x, h_effs, color='#2C3E50', edgecolor='black')
ax1.set_xticks(x)
ax1.set_xticklabels(prog_names, rotation=45, ha='right')
ax1.set_ylabel('$H_{eff}$ (kT)')
ax1.set_title('Progressive Y→S Titration: $H_{eff}$')
ax2.bar(x, csats, color='#E74C3C', edgecolor='black')
ax2.set_xticks(x)
ax2.set_xticklabels(prog_names, rotation=45, ha='right')
ax2.set_ylabel('Predicted $c_{sat}$ (relative)')
ax2.set_title('Progressive Y→S Titration: $c_{sat}$')
ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
save_figure(fig, figures_dir / "fig18_progressive_titration")
plt.close(fig)
print(f"  Saved fig18_progressive_titration.png/pdf")

# Figure 19: Block variants — clustered vs evenly-spaced
block_names = ["WT", "Y_Clustered", "Y_EvenSpaced"]
block_names = [n for n in block_names if n in EXPANDED]
if len(block_names) >= 2:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    x = np.arange(len(block_names))
    h_effs = [all_H_eff_cal[n].H_eff for n in block_names]
    h_topos = [all_H_eff_cal[n].H_topology for n in block_names]
    ax1.bar(x, h_effs, color='#9B59B6', edgecolor='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels(block_names, rotation=45, ha='right')
    ax1.set_ylabel('$H_{eff}$ (kT)')
    ax1.set_title('Sticker Arrangement: $H_{eff}$')
    ax2.bar(x, h_topos, color='#E74C3C', edgecolor='black')
    ax2.set_xticks(x)
    ax2.set_xticklabels(block_names, rotation=45, ha='right')
    ax2.set_ylabel('$H_{topology}$ (kT)')
    ax2.set_title('Sticker Arrangement: Topology Contribution')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    plt.tight_layout()
    save_figure(fig, figures_dir / "fig19_block_arrangement")
    plt.close(fig)
    print(f"  Saved fig19_block_arrangement.png/pdf")

# =============================================================================
# STEP 14: MTL Figures
# =============================================================================
print("\n[14/14] Generating MTL figures...")

# Figure 20: MTL c_sat predictions vs experimental (all variants)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: bar chart of MTL c_sat for all variants
all_variant_names = list(EXPANDED.keys())
mtl_csats = [mtl_result.predictions[n]['csat_relative'] for n in all_variant_names]
colors_mtl = ['#2C3E50' if n in mtl_csat_exp else '#95A5A6' for n in all_variant_names]
x = np.arange(len(all_variant_names))
ax1.bar(x, mtl_csats, color=colors_mtl, edgecolor='black')
# Overlay experimental points
for i, name in enumerate(all_variant_names):
    if name in mtl_csat_exp:
        ax1.scatter(i, mtl_csat_exp[name], color='#E74C3C', s=100, zorder=5,
                    marker='*', label='Experimental' if i == 0 else None)
ax1.set_xticks(x)
ax1.set_xticklabels(all_variant_names, rotation=60, ha='right', fontsize=8)
ax1.set_ylabel('$c_{sat}$ (relative to WT)')
ax1.set_title('MTL Predicted $c_{sat}$ (dark = labeled, gray = unlabeled)')
ax1.set_yscale('log')
ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

# Right: phase classification probabilities
phase_probs = [mtl_result.predictions[n]['phase_prob'] for n in all_variant_names]
phase_colors = ['#2ECC71' if p > 0.5 else '#E74C3C' for p in phase_probs]
ax2.bar(x, phase_probs, color=phase_colors, edgecolor='black')
ax2.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
ax2.set_xticks(x)
ax2.set_xticklabels(all_variant_names, rotation=60, ha='right', fontsize=8)
ax2.set_ylabel('P(phase separates)')
ax2.set_title('MTL Phase Classification')
ax2.set_ylim(0, 1.05)
plt.tight_layout()
save_figure(fig, figures_dir / "fig20_mtl_predictions")
plt.close(fig)
print(f"  Saved fig20_mtl_predictions.png/pdf")

# Figure 21: H_chem reconstruction accuracy
fig, ax = plt.subplots(figsize=(7, 6))
hc_true = [mtl_h_chem[n] for n in all_variant_names]
hc_pred = [mtl_result.predictions[n]['H_chemistry'] for n in all_variant_names]
ax.scatter(hc_true, hc_pred, s=60, c='#3498DB', edgecolors='black', zorder=5)
for i, name in enumerate(all_variant_names):
    ax.annotate(name, (hc_true[i], hc_pred[i]), textcoords="offset points",
                xytext=(5, 5), fontsize=7)
lims = [min(min(hc_true), min(hc_pred)) * 1.1, max(max(hc_true), max(hc_pred)) * 0.9]
ax.plot(lims, lims, 'k--', alpha=0.3)
ax.set_xlabel('True $H_{chemistry}$ (kT)')
ax.set_ylabel('MTL Predicted $H_{chemistry}$ (kT)')
ax.set_title('Physics Anchor: $H_{chem}$ Reconstruction')
plt.tight_layout()
save_figure(fig, figures_dir / "fig21_hchem_reconstruction")
plt.close(fig)
print(f"  Saved fig21_hchem_reconstruction.png/pdf")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("PIPELINE COMPLETE (Phase 2: Calibrated + MTL Hybrid Hamiltonian)")
print("=" * 70)

print(f"\nVariant Library: {len(EXPANDED)} variants")
print(f"Nelder-Mead calibrated against: {len(calibration_inputs)} experimental c_sat values")
print(f"MTL trained on: {len(training_data)} variants "
      f"({sum(1 for s in training_data if s.csat_target is not None)} c_sat, "
      f"{sum(1 for s in training_data if s.phase_target is not None)} phase, "
      f"{sum(1 for s in training_data if s.h_chem_target is not None)} H_chem labels)")

print("\nKey Results — MTL vs Nelder-Mead on experimental c_sat:")
for name in ['WT', 'AllY_to_S', 'AllY_to_F']:
    c_exp = mtl_csat_exp.get(name, None)
    c_nm = csat_calibrated[name]
    c_mtl = mtl_result.predictions[name]['csat_relative']
    cexp_str = f"{c_exp:.2f}" if c_exp else "—"
    print(f"  {name:12s}: exp={cexp_str}  NM={c_nm:.3f}  MTL={c_mtl:.3f}")

print(f"\nGenerated Files:")
print(f"  data/outputs/phase2_results.json  — calibration, robustness, MTL, model comparison")
print(f"  figures/                          — 21 publication figures (PNG + PDF)")

print("\n✓ All computations successful!")
