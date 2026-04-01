"""
plotting.py - Publication-quality visualization for FUS LCD analysis.

This module provides matplotlib-based plotting utilities for:
- Interaction map heatmaps
- Difference maps
- Sticker-linker diagrams
- Interaction profiles
- Metric comparisons

All plots follow publication standards:
- Clear labeling
- Consistent color schemes
- Appropriate font sizes
- Vector-compatible output
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

from .sequences import SequenceRecord, AROMATIC_RESIDUES
from .segmentation import StickerMask


# =============================================================================
# PLOT CONFIGURATION
# =============================================================================

# Publication-quality defaults
FIGURE_DPI = 150
SAVE_DPI = 300

# Color schemes
CMAP_INTERACTION = "RdBu_r"  # Red = attractive (negative), Blue = repulsive
CMAP_DIFFERENCE = "PiYG"     # Green = more attractive in variant
CMAP_SEQUENTIAL = "viridis"

# Sticker-linker colors
COLOR_STICKER = "#E74C3C"  # Red
COLOR_LINKER = "#3498DB"   # Blue
COLOR_TYROSINE = "#F39C12" # Orange
COLOR_AROMATIC = "#9B59B6" # Purple

# Font sizes
FONTSIZE_TITLE = 14
FONTSIZE_LABEL = 12
FONTSIZE_TICK = 10
FONTSIZE_LEGEND = 10


def set_publication_style():
    """Set matplotlib parameters for publication-quality figures."""
    plt.rcParams.update({
        "figure.dpi": FIGURE_DPI,
        "savefig.dpi": SAVE_DPI,
        "font.size": FONTSIZE_TICK,
        "axes.titlesize": FONTSIZE_TITLE,
        "axes.labelsize": FONTSIZE_LABEL,
        "xtick.labelsize": FONTSIZE_TICK,
        "ytick.labelsize": FONTSIZE_TICK,
        "legend.fontsize": FONTSIZE_LEGEND,
        "figure.figsize": (8, 6),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,  # TrueType fonts for PDF
        "ps.fonttype": 42,
    })


# =============================================================================
# INTERACTION MAP PLOTS
# =============================================================================

def plot_interaction_map(
    intermap: np.ndarray,
    title: str = "Interaction Map",
    cmap: str = CMAP_INTERACTION,
    symmetric: bool = True,
    colorbar_label: str = "Interaction Energy (kT)",
    ax: Optional[plt.Axes] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Tuple[float, float] = (8, 7),
) -> plt.Figure:
    """
    Plot an interaction map as a heatmap.

    Parameters
    ----------
    intermap : np.ndarray
        NxN interaction map
    title : str
        Plot title
    cmap : str
        Colormap name
    symmetric : bool
        If True, center colormap at 0
    colorbar_label : str
        Label for colorbar
    ax : Optional[plt.Axes]
        Existing axes to plot on
    vmin, vmax : Optional[float]
        Color scale limits
    figsize : Tuple[float, float]
        Figure size

    Returns
    -------
    plt.Figure
        Figure object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Determine color limits
    if symmetric and vmin is None and vmax is None:
        abs_max = max(abs(intermap.min()), abs(intermap.max()))
        vmin, vmax = -abs_max, abs_max

    im = ax.imshow(
        intermap,
        cmap=cmap,
        aspect="equal",
        origin="upper",
        vmin=vmin,
        vmax=vmax,
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(colorbar_label)

    # Labels
    ax.set_xlabel("Residue Position")
    ax.set_ylabel("Residue Position")
    ax.set_title(title)

    # Tick marks at intervals
    n = intermap.shape[0]
    tick_interval = max(1, n // 10) * 10
    ticks = np.arange(0, n, tick_interval)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks + 1)  # 1-indexed
    ax.set_yticklabels(ticks + 1)

    plt.tight_layout()
    return fig


def plot_interaction_maps_comparison(
    intermaps: Dict[str, np.ndarray],
    titles: Optional[Dict[str, str]] = None,
    ncols: int = 3,
    cmap: str = CMAP_INTERACTION,
    figsize: Optional[Tuple[float, float]] = None,
    shared_scale: bool = True,
) -> plt.Figure:
    """
    Plot multiple interaction maps side by side.

    Parameters
    ----------
    intermaps : Dict[str, np.ndarray]
        Dictionary of name -> intermap
    titles : Optional[Dict[str, str]]
        Custom titles (default: use keys)
    ncols : int
        Number of columns
    cmap : str
        Colormap
    figsize : Optional[Tuple[float, float]]
        Figure size
    shared_scale : bool
        Use same color scale for all maps

    Returns
    -------
    plt.Figure
        Figure object
    """
    names = list(intermaps.keys())
    n = len(names)
    nrows = (n + ncols - 1) // ncols

    if figsize is None:
        figsize = (5 * ncols, 4.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    # Determine shared scale
    if shared_scale:
        all_vals = np.concatenate([im.ravel() for im in intermaps.values()])
        abs_max = max(abs(all_vals.min()), abs(all_vals.max()))
        vmin, vmax = -abs_max, abs_max
    else:
        vmin, vmax = None, None

    for idx, name in enumerate(names):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]

        imap = intermaps[name]
        title = titles[name] if titles and name in titles else name

        im = ax.imshow(
            imap,
            cmap=cmap,
            aspect="equal",
            origin="upper",
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_title(title)
        ax.set_xlabel("Residue")
        ax.set_ylabel("Residue")

        # Simplified ticks
        n_res = imap.shape[0]
        tick_interval = max(1, n_res // 5) * 10
        ticks = np.arange(0, n_res, tick_interval)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

    # Remove empty axes
    for idx in range(n, nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row, col].axis("off")

    # Shared colorbar
    if shared_scale:
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
        fig.colorbar(im, cax=cbar_ax, label="Energy (kT)")

    plt.tight_layout()
    return fig


# =============================================================================
# DIFFERENCE MAP PLOTS
# =============================================================================

def plot_difference_map(
    diff_map: np.ndarray,
    title: str = "Difference Map (Variant - WT)",
    cmap: str = CMAP_DIFFERENCE,
    colorbar_label: str = "ΔEnergy (kT)",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (8, 7),
) -> plt.Figure:
    """
    Plot a difference map.

    Positive values (green): variant more attractive
    Negative values (pink): WT more attractive

    Parameters
    ----------
    diff_map : np.ndarray
        NxN difference map
    title : str
        Plot title
    cmap : str
        Colormap
    colorbar_label : str
        Colorbar label
    ax : Optional[plt.Axes]
        Existing axes
    figsize : Tuple[float, float]
        Figure size

    Returns
    -------
    plt.Figure
        Figure object
    """
    return plot_interaction_map(
        diff_map,
        title=title,
        cmap=cmap,
        symmetric=True,
        colorbar_label=colorbar_label,
        ax=ax,
        figsize=figsize,
    )


def plot_difference_maps_panel(
    diff_maps: Dict[str, np.ndarray],
    reference_name: str = "WT",
    ncols: int = 2,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """
    Plot panel of difference maps relative to reference.

    Parameters
    ----------
    diff_maps : Dict[str, np.ndarray]
        Dictionary of variant name -> difference map
    reference_name : str
        Name of reference (for titles)
    ncols : int
        Number of columns
    figsize : Optional[Tuple[float, float]]
        Figure size

    Returns
    -------
    plt.Figure
        Figure object
    """
    titles = {
        name: f"{name} − {reference_name}"
        for name in diff_maps
    }
    return plot_interaction_maps_comparison(
        diff_maps,
        titles=titles,
        ncols=ncols,
        cmap=CMAP_DIFFERENCE,
        figsize=figsize,
        shared_scale=True,
    )


# =============================================================================
# INTERACTION PROFILE PLOTS
# =============================================================================

def plot_interaction_profile(
    profile: np.ndarray,
    sequence: Optional[Union[str, SequenceRecord]] = None,
    sticker_mask: Optional[np.ndarray] = None,
    title: str = "Mean Interaction Profile",
    ylabel: str = "Mean Interaction Energy",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (12, 4),
    highlight_tyrosines: bool = True,
) -> plt.Figure:
    """
    Plot 1D interaction profile along sequence.

    Parameters
    ----------
    profile : np.ndarray
        1D interaction profile
    sequence : Optional[Union[str, SequenceRecord]]
        Sequence for highlighting residues
    sticker_mask : Optional[np.ndarray]
        Sticker mask for shading
    title : str
        Plot title
    ylabel : str
        Y-axis label
    ax : Optional[plt.Axes]
        Existing axes
    figsize : Tuple[float, float]
        Figure size
    highlight_tyrosines : bool
        Mark tyrosine positions

    Returns
    -------
    plt.Figure
        Figure object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    n = len(profile)
    x = np.arange(n) + 1  # 1-indexed

    # Plot profile
    ax.plot(x, profile, "k-", linewidth=1.5, label="Profile")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    # Shade sticker regions
    if sticker_mask is not None:
        sticker_region = profile.copy()
        sticker_region[~sticker_mask] = np.nan
        ax.fill_between(
            x, 0, sticker_region,
            alpha=0.3, color=COLOR_STICKER, label="Stickers"
        )

    # Highlight tyrosines
    if highlight_tyrosines and sequence is not None:
        if isinstance(sequence, SequenceRecord):
            seq = sequence.sequence
        else:
            seq = sequence
        tyr_pos = [i + 1 for i, aa in enumerate(seq) if aa == "Y"]
        tyr_vals = [profile[i - 1] for i in tyr_pos]
        ax.scatter(
            tyr_pos, tyr_vals,
            c=COLOR_TYROSINE, s=50, zorder=5, marker="v", label="Tyrosine"
        )

    ax.set_xlabel("Residue Position")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(1, n)
    ax.legend(loc="upper right")

    plt.tight_layout()
    return fig


def plot_profiles_comparison(
    profiles: Dict[str, np.ndarray],
    title: str = "Interaction Profile Comparison",
    ylabel: str = "Mean Interaction Energy",
    figsize: Tuple[float, float] = (12, 5),
) -> plt.Figure:
    """
    Plot multiple profiles on same axes.

    Parameters
    ----------
    profiles : Dict[str, np.ndarray]
        Dictionary of name -> profile
    title : str
        Plot title
    ylabel : str
        Y-axis label
    figsize : Tuple[float, float]
        Figure size

    Returns
    -------
    plt.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(profiles)))

    for (name, profile), color in zip(profiles.items(), colors):
        n = len(profile)
        x = np.arange(n) + 1
        ax.plot(x, profile, linewidth=1.5, label=name, color=color)

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Residue Position")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right")

    plt.tight_layout()
    return fig


# =============================================================================
# STICKER-LINKER DIAGRAM
# =============================================================================

def plot_sticker_linker_diagram(
    sticker_mask: Union[np.ndarray, StickerMask],
    sequence: Optional[Union[str, SequenceRecord]] = None,
    title: str = "Sticker-Linker Segmentation",
    figsize: Tuple[float, float] = (14, 2),
    show_residue_labels: bool = False,
) -> plt.Figure:
    """
    Plot sticker-linker diagram as colored bar.

    Parameters
    ----------
    sticker_mask : Union[np.ndarray, StickerMask]
        Sticker mask
    sequence : Optional[Union[str, SequenceRecord]]
        Sequence for residue labels
    title : str
        Plot title
    figsize : Tuple[float, float]
        Figure size
    show_residue_labels : bool
        Show single-letter codes

    Returns
    -------
    plt.Figure
        Figure object
    """
    if isinstance(sticker_mask, StickerMask):
        mask = sticker_mask.mask
    else:
        mask = sticker_mask

    n = len(mask)

    fig, ax = plt.subplots(figsize=figsize)

    # Draw colored rectangles
    patches = []
    for i in range(n):
        color = COLOR_STICKER if mask[i] else COLOR_LINKER
        rect = Rectangle((i, 0), 1, 1, facecolor=color, edgecolor="none")
        patches.append(rect)

    collection = PatchCollection(patches, match_original=True)
    ax.add_collection(collection)

    # Axis setup
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Residue Position")
    ax.set_yticks([])
    ax.set_title(title)

    # Tick labels
    tick_interval = max(1, n // 10) * 10
    ticks = np.arange(0, n + 1, tick_interval)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_STICKER, label="Sticker"),
        Patch(facecolor=COLOR_LINKER, label="Linker"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    return fig


def plot_sticker_linker_comparison(
    masks: Dict[str, Union[np.ndarray, StickerMask]],
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """
    Plot sticker-linker diagrams for multiple variants.

    Parameters
    ----------
    masks : Dict[str, Union[np.ndarray, StickerMask]]
        Dictionary of name -> mask
    figsize : Optional[Tuple[float, float]]
        Figure size

    Returns
    -------
    plt.Figure
        Figure object
    """
    n_variants = len(masks)
    if figsize is None:
        figsize = (14, 1.5 * n_variants)

    fig, axes = plt.subplots(n_variants, 1, figsize=figsize, sharex=True)
    if n_variants == 1:
        axes = [axes]

    for ax, (name, mask) in zip(axes, masks.items()):
        if isinstance(mask, StickerMask):
            mask_arr = mask.mask
        else:
            mask_arr = mask

        n = len(mask_arr)

        # Draw bars
        for i in range(n):
            color = COLOR_STICKER if mask_arr[i] else COLOR_LINKER
            ax.axvspan(i, i + 1, facecolor=color, edgecolor="none")

        ax.set_xlim(0, n)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(name, rotation=0, ha="right", va="center")

    axes[-1].set_xlabel("Residue Position")

    # Shared legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_STICKER, label="Sticker"),
        Patch(facecolor=COLOR_LINKER, label="Linker"),
    ]
    fig.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    return fig


# =============================================================================
# METRICS PLOTS
# =============================================================================

def plot_metrics_bar(
    metrics: Dict[str, float],
    title: str = "Metric Comparison",
    ylabel: str = "Value",
    figsize: Tuple[float, float] = (8, 5),
    color: str = "#3498DB",
    highlight_wt: bool = True,
) -> plt.Figure:
    """
    Plot bar chart of metrics across variants.

    Parameters
    ----------
    metrics : Dict[str, float]
        Dictionary of variant name -> metric value
    title : str
        Plot title
    ylabel : str
        Y-axis label
    figsize : Tuple[float, float]
        Figure size
    color : str
        Bar color
    highlight_wt : bool
        Highlight WT bar differently

    Returns
    -------
    plt.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    names = list(metrics.keys())
    values = list(metrics.values())
    x = np.arange(len(names))

    colors = [COLOR_TYROSINE if name == "WT" and highlight_wt else color for name in names]
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)

    plt.tight_layout()
    return fig


def plot_metrics_multi_panel(
    metrics_dict: Dict[str, Dict[str, float]],
    metric_names: List[str],
    ncols: int = 3,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """
    Plot multiple metrics as panels.

    Parameters
    ----------
    metrics_dict : Dict[str, Dict[str, float]]
        Nested dict: metric_name -> {variant: value}
    metric_names : List[str]
        Metrics to plot
    ncols : int
        Number of columns
    figsize : Optional[Tuple[float, float]]
        Figure size

    Returns
    -------
    plt.Figure
        Figure object
    """
    n_metrics = len(metric_names)
    nrows = (n_metrics + ncols - 1) // ncols

    if figsize is None:
        figsize = (4 * ncols, 3 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_metrics == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    for idx, metric_name in enumerate(metric_names):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]

        data = metrics_dict.get(metric_name, {})
        names = list(data.keys())
        values = list(data.values())
        x = np.arange(len(names))

        ax.bar(x, values, color="#3498DB", edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_title(metric_name)
        ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)

    # Remove empty axes
    for idx in range(n_metrics, nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row, col].axis("off")

    plt.tight_layout()
    return fig


# =============================================================================
# FORCE FIELD VISUALIZATION
# =============================================================================

def plot_interaction_matrix(
    matrix: np.ndarray,
    aa_order: str = "ACDEFGHIKLMNPQRSTVWY",
    title: str = "Residue-Residue Interaction Matrix",
    cmap: str = CMAP_INTERACTION,
    figsize: Tuple[float, float] = (10, 8),
) -> plt.Figure:
    """
    Plot the 20x20 amino acid interaction matrix.

    Parameters
    ----------
    matrix : np.ndarray
        20x20 interaction matrix
    aa_order : str
        Amino acid ordering
    title : str
        Plot title
    cmap : str
        Colormap
    figsize : Tuple[float, float]
        Figure size

    Returns
    -------
    plt.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    abs_max = max(abs(matrix.min()), abs(matrix.max()))
    im = ax.imshow(matrix, cmap=cmap, aspect="equal", vmin=-abs_max, vmax=abs_max)

    # Labels
    ax.set_xticks(np.arange(len(aa_order)))
    ax.set_yticks(np.arange(len(aa_order)))
    ax.set_xticklabels(list(aa_order))
    ax.set_yticklabels(list(aa_order))

    ax.set_xlabel("Amino Acid j")
    ax.set_ylabel("Amino Acid i")
    ax.set_title(title)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Interaction Energy (kT)")

    plt.tight_layout()
    return fig


# =============================================================================
# SAVE UTILITIES
# =============================================================================

def save_figure(
    fig: plt.Figure,
    path: Union[str, Path],
    formats: List[str] = ["png", "pdf"],
    dpi: int = SAVE_DPI,
) -> None:
    """
    Save figure in multiple formats.

    Parameters
    ----------
    fig : plt.Figure
        Figure to save
    path : Union[str, Path]
        Output path (without extension)
    formats : List[str]
        File formats to save
    dpi : int
        Resolution for raster formats
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        fig.savefig(
            path.with_suffix(f".{fmt}"),
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )


def create_figure_panel(
    n_panels: int,
    ncols: int = 3,
    panel_size: Tuple[float, float] = (4, 4),
    **kwargs
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a figure with multiple panels.

    Parameters
    ----------
    n_panels : int
        Number of panels
    ncols : int
        Number of columns
    panel_size : Tuple[float, float]
        Size of each panel

    Returns
    -------
    Tuple[plt.Figure, np.ndarray]
        Figure and axes array
    """
    nrows = (n_panels + ncols - 1) // ncols
    figsize = (panel_size[0] * ncols, panel_size[1] * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)

    if n_panels == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    return fig, axes


# =============================================================================
# TOPOLOGY VISUALIZATIONS (Phase 1)
# =============================================================================

def plot_contact_graph(
    adjacency: np.ndarray,
    sticker_positions: Optional[np.ndarray] = None,
    node_labels: Optional[List[str]] = None,
    title: str = "Contact Graph",
    figsize: Tuple[float, float] = (8, 8),
    node_color: str = COLOR_STICKER,
    edge_color: str = "#CCCCCC",
    edge_alpha: float = 0.3,
) -> plt.Figure:
    """
    Plot a contact graph using a circular layout.

    Parameters
    ----------
    adjacency : np.ndarray
        Boolean or weighted adjacency matrix
    sticker_positions : Optional[np.ndarray]
        Original sequence positions (for labeling)
    node_labels : Optional[List[str]]
        Residue labels
    title : str
        Plot title
    figsize : Tuple
        Figure size
    node_color : str
        Node color
    edge_color : str
        Edge color
    edge_alpha : float
        Edge transparency

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    n = adjacency.shape[0]

    if n == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No nodes", ha='center', va='center', transform=ax.transAxes)
        return fig

    # Circular layout
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)

    # Draw edges
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j]:
                ax.plot([x[i], x[j]], [y[i], y[j]],
                        color=edge_color, alpha=edge_alpha, linewidth=0.5)

    # Draw nodes
    ax.scatter(x, y, s=60, c=node_color, edgecolors='black', linewidths=0.5, zorder=5)

    # Labels
    if node_labels and n <= 40:
        for i in range(n):
            label = node_labels[i]
            if sticker_positions is not None:
                label = f"{node_labels[i]}{sticker_positions[i]}"
            offset = 1.12
            ax.text(x[i] * offset, y[i] * offset, label,
                    ha='center', va='center', fontsize=6)

    ax.set_title(title)
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect('equal')
    ax.axis('off')

    return fig


def plot_percolation_curve(
    thresholds: np.ndarray,
    fraction_connected: np.ndarray,
    n_components: np.ndarray,
    percolation_threshold: float,
    title: str = "Percolation Analysis",
    figsize: Tuple[float, float] = (10, 5),
) -> plt.Figure:
    """
    Plot percolation transition: connectivity vs threshold.

    Parameters
    ----------
    thresholds : np.ndarray
        Energy thresholds
    fraction_connected : np.ndarray
        Fraction of nodes in largest component
    n_components : np.ndarray
        Number of connected components
    percolation_threshold : float
        Estimated critical threshold
    title : str
        Plot title
    figsize : Tuple
        Figure size

    Returns
    -------
    plt.Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left: fraction in giant component
    ax1.plot(thresholds, fraction_connected, color='#2C3E50', linewidth=2)
    ax1.axvline(x=percolation_threshold, color=COLOR_STICKER,
                linestyle='--', alpha=0.7, label=f'Perc. threshold = {percolation_threshold:.3f}')
    ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Energy Threshold (kT)')
    ax1.set_ylabel('Fraction in Largest Component')
    ax1.set_title('Giant Component Transition')
    ax1.legend(fontsize=9)
    ax1.set_ylim(-0.05, 1.05)

    # Right: number of components
    ax2.plot(thresholds, n_components, color='#3498DB', linewidth=2)
    ax2.axvline(x=percolation_threshold, color=COLOR_STICKER,
                linestyle='--', alpha=0.7)
    ax2.set_xlabel('Energy Threshold (kT)')
    ax2.set_ylabel('Number of Components')
    ax2.set_title('Component Count')

    plt.suptitle(title)
    plt.tight_layout()
    return fig


def plot_persistence_diagram(
    h0_pairs: np.ndarray,
    h1_pairs: Optional[np.ndarray] = None,
    title: str = "Persistence Diagram",
    figsize: Tuple[float, float] = (7, 7),
) -> plt.Figure:
    """
    Plot persistence diagram (birth vs death).

    Points far from the diagonal represent long-lived topological features.

    Parameters
    ----------
    h0_pairs : np.ndarray
        H0 (connectivity) birth-death pairs
    h1_pairs : Optional[np.ndarray]
        H1 (cycles) birth-death pairs
    title : str
        Plot title
    figsize : Tuple
        Figure size

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Determine plot range (exclude infinite deaths)
    all_finite = []
    if len(h0_pairs) > 0:
        finite_h0 = h0_pairs[np.isfinite(h0_pairs[:, 1])]
        if len(finite_h0) > 0:
            all_finite.append(finite_h0)
    if h1_pairs is not None and len(h1_pairs) > 0:
        finite_h1 = h1_pairs[np.isfinite(h1_pairs[:, 1])]
        if len(finite_h1) > 0:
            all_finite.append(finite_h1)

    if all_finite:
        all_vals = np.concatenate(all_finite)
        vmin = float(np.min(all_vals))
        vmax = float(np.max(all_vals)) * 1.1
    else:
        vmin, vmax = 0, 1

    # Diagonal line (birth = death)
    ax.plot([vmin, vmax], [vmin, vmax], 'k--', alpha=0.3, linewidth=1)

    # H0 points
    if len(h0_pairs) > 0:
        finite_mask = np.isfinite(h0_pairs[:, 1])
        if np.any(finite_mask):
            ax.scatter(h0_pairs[finite_mask, 0], h0_pairs[finite_mask, 1],
                       c='#3498DB', s=40, alpha=0.7, label='H0 (components)',
                       edgecolors='black', linewidths=0.5)
        # Infinite deaths: plot as triangles at top
        if np.any(~finite_mask):
            ax.scatter(h0_pairs[~finite_mask, 0],
                       np.full(np.sum(~finite_mask), vmax),
                       c='#3498DB', s=60, marker='^', alpha=0.7,
                       edgecolors='black', linewidths=0.5)

    # H1 points
    if h1_pairs is not None and len(h1_pairs) > 0:
        finite_mask = np.isfinite(h1_pairs[:, 1])
        if np.any(finite_mask):
            ax.scatter(h1_pairs[finite_mask, 0], h1_pairs[finite_mask, 1],
                       c=COLOR_STICKER, s=40, alpha=0.7, label='H1 (cycles)',
                       edgecolors='black', linewidths=0.5)

    ax.set_xlabel('Birth')
    ax.set_ylabel('Death')
    ax.set_title(title)
    ax.legend()
    ax.set_aspect('equal')

    return fig


def plot_betti_curves(
    thresholds: np.ndarray,
    betti_0: np.ndarray,
    betti_1: np.ndarray,
    title: str = "Betti Number Curves",
    figsize: Tuple[float, float] = (10, 5),
) -> plt.Figure:
    """
    Plot Betti numbers as functions of filtration threshold.

    Parameters
    ----------
    thresholds : np.ndarray
        Filtration thresholds
    betti_0 : np.ndarray
        Betti-0 (components) at each threshold
    betti_1 : np.ndarray
        Betti-1 (cycles) at each threshold
    title : str
        Plot title
    figsize : Tuple
        Figure size

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.plot(thresholds, betti_0, color='#3498DB', linewidth=2, label='$\\beta_0$ (components)')
    ax.fill_between(thresholds, betti_0, alpha=0.15, color='#3498DB')
    ax.plot(thresholds, betti_1, color=COLOR_STICKER, linewidth=2, label='$\\beta_1$ (cycles)')
    ax.fill_between(thresholds, betti_1, alpha=0.15, color=COLOR_STICKER)

    ax.set_xlabel('Filtration Threshold')
    ax.set_ylabel('Betti Number')
    ax.set_title(title)
    ax.legend()

    return fig


def plot_topology_summary(
    variant_names: List[str],
    topology_metrics: Dict,
    figsize: Tuple[float, float] = (14, 10),
) -> plt.Figure:
    """
    Multi-panel bar chart of topology metrics across variants.

    Parameters
    ----------
    variant_names : List[str]
        Names of variants
    topology_metrics : Dict
        Dict mapping variant name -> TopologyMetrics-like object with to_dict()
    figsize : Tuple
        Figure size

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    x = np.arange(len(variant_names))

    metrics_to_plot = [
        ("sticker_clustering_coefficient", "Sticker Clustering Coeff.", '#E74C3C'),
        ("sticker_mean_degree", "Sticker Mean Degree", '#3498DB'),
        ("sticker_n_contacts", "Sticker Contacts", '#2ECC71'),
        ("percolation_threshold", "Percolation Threshold", '#9B59B6'),
        ("sticker_graph_density", "Sticker Graph Density", '#F39C12'),
        ("sticker_n_components", "Connected Components", '#2C3E50'),
    ]

    for ax, (key, label, color) in zip(axes.flat, metrics_to_plot):
        vals = [topology_metrics[name].to_dict()[key] for name in variant_names]
        ax.bar(x, vals, color=color, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(variant_names, rotation=45, ha='right')
        ax.set_ylabel(label)
        ax.set_title(label)

    plt.tight_layout()
    return fig


def plot_entropy_comparison(
    variant_names: List[str],
    entropy_metrics: Dict,
    figsize: Tuple[float, float] = (12, 5),
) -> plt.Figure:
    """
    Bar chart comparing entropy metrics across variants.

    Parameters
    ----------
    variant_names : List[str]
        Variant names
    entropy_metrics : Dict
        Dict mapping variant name -> EntropyMetrics-like object with to_dict()
    figsize : Tuple
        Figure size

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    x = np.arange(len(variant_names))

    metrics_to_plot = [
        ("normalized_spacing_entropy", "Norm. Spacing Entropy", '#E74C3C'),
        ("block_entropy", "Block Entropy (bits)", '#3498DB'),
        ("composition_entropy", "Composition Entropy (bits)", '#9B59B6'),
    ]

    for ax, (key, label, color) in zip(axes, metrics_to_plot):
        vals = [entropy_metrics[name].to_dict()[key] for name in variant_names]
        ax.bar(x, vals, color=color, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(variant_names, rotation=45, ha='right')
        ax.set_ylabel(label)
        ax.set_title(label)

    plt.tight_layout()
    return fig


# Apply publication style on import
set_publication_style()


if __name__ == "__main__":
    # Demo plots
    print("Generating demo plots...")

    # Create synthetic data
    n = 100
    np.random.seed(42)
    demo_map = np.random.randn(n, n) * 0.3
    demo_map = (demo_map + demo_map.T) / 2  # Symmetrize

    # Demo interaction map
    fig = plot_interaction_map(demo_map, title="Demo Interaction Map")
    plt.show()

    print("Done!")
