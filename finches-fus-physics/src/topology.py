"""
topology.py - Contact graph construction and graph-theoretic analysis.

This module builds interaction networks from FINCHES interaction maps and
sticker masks, then computes graph metrics relevant to phase separation:

- Contact graph construction (threshold-based from interaction maps)
- Sticker interaction network (sticker-only subgraph)
- Graph metrics: clustering coefficient, path lengths, degree distribution
- Centrality measures: betweenness, eigenvector
- Percolation analysis: connectivity transition as a function of threshold
- Community detection via spectral methods

The key insight: network topology of sticker-sticker interactions
captures emergent structural features beyond pairwise energetics.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

from .segmentation import StickerMask


# =============================================================================
# CONTACT GRAPH CONSTRUCTION
# =============================================================================

@dataclass
class ContactGraph:
    """Lightweight graph representation from interaction/contact maps.

    Stores adjacency matrix and derived properties without requiring
    networkx for basic operations.
    """
    adjacency: np.ndarray           # NxN boolean or weighted adjacency
    weights: Optional[np.ndarray]   # NxN edge weights (interaction energies)
    node_labels: Optional[List[str]]  # residue identities
    n_nodes: int
    n_edges: int
    is_weighted: bool

    @property
    def density(self) -> float:
        """Graph density: fraction of possible edges present."""
        max_edges = self.n_nodes * (self.n_nodes - 1) / 2
        if max_edges == 0:
            return 0.0
        return self.n_edges / max_edges

    @property
    def degree_sequence(self) -> np.ndarray:
        """Degree of each node."""
        return np.sum(self.adjacency > 0, axis=1)


def build_contact_graph(
    intermap: np.ndarray,
    threshold: float = -0.2,
    min_separation: int = 4,
    sequence: Optional[str] = None,
) -> ContactGraph:
    """
    Build a contact graph from an interaction map.

    An edge exists between residues i and j if:
    1. E(i,j) < threshold (attractive interaction)
    2. |i - j| >= min_separation (not trivially close in sequence)

    Parameters
    ----------
    intermap : np.ndarray
        NxN interaction map (energies in kT)
    threshold : float
        Energy threshold for edge creation (more negative = stricter)
    min_separation : int
        Minimum sequence separation for edges
    sequence : Optional[str]
        Amino acid sequence for node labels

    Returns
    -------
    ContactGraph
        Contact graph representation
    """
    n = intermap.shape[0]

    # Build separation mask
    i_idx, j_idx = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    separation_mask = np.abs(i_idx - j_idx) >= min_separation

    # Build adjacency
    adjacency = (intermap < threshold) & separation_mask
    np.fill_diagonal(adjacency, False)  # no self-loops

    # Edge weights are interaction energies (only for edges)
    weights = np.where(adjacency, intermap, 0.0)

    # Count edges (upper triangle only for undirected)
    n_edges = int(np.sum(np.triu(adjacency, k=1)))

    labels = list(sequence) if sequence else None

    return ContactGraph(
        adjacency=adjacency.astype(bool),
        weights=weights,
        node_labels=labels,
        n_nodes=n,
        n_edges=n_edges,
        is_weighted=True,
    )


def build_sticker_contact_graph(
    intermap: np.ndarray,
    sticker_mask: Union[np.ndarray, StickerMask],
    threshold: float = -0.2,
    min_separation: int = 4,
    sequence: Optional[str] = None,
) -> ContactGraph:
    """
    Build a contact graph restricted to sticker residues only.

    This captures the multivalent interaction network that drives
    phase separation — the topology of sticker-sticker contacts.

    Parameters
    ----------
    intermap : np.ndarray
        NxN interaction map
    sticker_mask : Union[np.ndarray, StickerMask]
        Sticker mask
    threshold : float
        Energy threshold for edges
    min_separation : int
        Minimum sequence separation
    sequence : Optional[str]
        Full amino acid sequence

    Returns
    -------
    ContactGraph
        Sticker-only contact graph
    """
    if isinstance(sticker_mask, StickerMask):
        positions = sticker_mask.positions
        mask = sticker_mask.mask
    else:
        positions = np.where(sticker_mask)[0]
        mask = sticker_mask

    n_stickers = len(positions)
    if n_stickers == 0:
        return ContactGraph(
            adjacency=np.zeros((0, 0), dtype=bool),
            weights=None,
            node_labels=[],
            n_nodes=0,
            n_edges=0,
            is_weighted=False,
        )

    # Extract sticker-sticker submatrix
    sub_intermap = intermap[np.ix_(positions, positions)]

    # Sequence separations between sticker positions
    sep = np.abs(positions[:, None] - positions[None, :])

    adjacency = (sub_intermap < threshold) & (sep >= min_separation)
    np.fill_diagonal(adjacency, False)

    weights = np.where(adjacency, sub_intermap, 0.0)
    n_edges = int(np.sum(np.triu(adjacency, k=1)))

    if sequence:
        labels = [sequence[p] for p in positions]
    else:
        labels = None

    return ContactGraph(
        adjacency=adjacency.astype(bool),
        weights=weights,
        node_labels=labels,
        n_nodes=n_stickers,
        n_edges=n_edges,
        is_weighted=True,
    )


# =============================================================================
# GRAPH METRICS
# =============================================================================

def compute_clustering_coefficient(graph: ContactGraph) -> float:
    """
    Compute global clustering coefficient.

    C = 3 * (number of triangles) / (number of connected triples)

    Higher clustering indicates stickers form tightly interconnected
    subnetworks — important for robust phase separation.

    Parameters
    ----------
    graph : ContactGraph
        Contact graph

    Returns
    -------
    float
        Global clustering coefficient (0 to 1)
    """
    adj = graph.adjacency.astype(float)
    n = graph.n_nodes

    if n < 3:
        return 0.0

    # Count triangles: trace(A^3) / 6
    a2 = adj @ adj
    a3 = a2 @ adj
    n_triangles = np.trace(a3) / 6.0

    # Count connected triples (paths of length 2)
    # For each node, number of pairs of neighbors = C(degree, 2)
    degrees = np.sum(adj, axis=1)
    n_triples = np.sum(degrees * (degrees - 1)) / 2.0

    if n_triples < 1e-10:
        return 0.0

    return float(3.0 * n_triangles / n_triples)


def compute_local_clustering(graph: ContactGraph) -> np.ndarray:
    """
    Compute local clustering coefficient for each node.

    Parameters
    ----------
    graph : ContactGraph
        Contact graph

    Returns
    -------
    np.ndarray
        Local clustering coefficients (length n_nodes)
    """
    adj = graph.adjacency.astype(float)
    n = graph.n_nodes
    local_cc = np.zeros(n)

    for i in range(n):
        neighbors = np.where(adj[i] > 0)[0]
        k = len(neighbors)
        if k < 2:
            continue
        # Count edges among neighbors
        sub = adj[np.ix_(neighbors, neighbors)]
        n_edges_among = np.sum(sub) / 2.0
        local_cc[i] = 2.0 * n_edges_among / (k * (k - 1))

    return local_cc


def compute_average_path_length(graph: ContactGraph) -> float:
    """
    Compute average shortest path length using BFS.

    Only considers the largest connected component.

    Parameters
    ----------
    graph : ContactGraph
        Contact graph

    Returns
    -------
    float
        Average shortest path length (NaN if graph is disconnected with no large component)
    """
    adj = graph.adjacency
    n = graph.n_nodes

    if n < 2:
        return np.nan

    # Find largest connected component
    components = _find_connected_components(adj)
    if not components:
        return np.nan

    largest = max(components, key=len)
    if len(largest) < 2:
        return np.nan

    # BFS shortest paths within largest component
    nodes = sorted(largest)
    node_set = set(nodes)
    total_dist = 0.0
    n_pairs = 0

    for source in nodes:
        distances = _bfs_distances(adj, source, node_set)
        for target in nodes:
            if target > source and target in distances:
                total_dist += distances[target]
                n_pairs += 1

    if n_pairs == 0:
        return np.nan

    return total_dist / n_pairs


def compute_degree_distribution(graph: ContactGraph) -> Dict[str, float]:
    """
    Compute degree distribution statistics.

    Parameters
    ----------
    graph : ContactGraph
        Contact graph

    Returns
    -------
    Dict[str, float]
        mean_degree, std_degree, max_degree, min_degree
    """
    degrees = graph.degree_sequence

    if len(degrees) == 0:
        return {"mean_degree": 0.0, "std_degree": 0.0,
                "max_degree": 0.0, "min_degree": 0.0}

    return {
        "mean_degree": float(np.mean(degrees)),
        "std_degree": float(np.std(degrees)),
        "max_degree": float(np.max(degrees)),
        "min_degree": float(np.min(degrees)),
    }


def compute_betweenness_centrality(graph: ContactGraph) -> np.ndarray:
    """
    Compute betweenness centrality for each node.

    Identifies residues that serve as bridges in the interaction network.
    High-betweenness stickers are critical for network connectivity.

    Parameters
    ----------
    graph : ContactGraph
        Contact graph

    Returns
    -------
    np.ndarray
        Betweenness centrality for each node
    """
    adj = graph.adjacency
    n = graph.n_nodes
    centrality = np.zeros(n)

    if n < 3:
        return centrality

    # Brandes algorithm (simplified)
    for s in range(n):
        # BFS from s
        stack = []
        predecessors = [[] for _ in range(n)]
        sigma = np.zeros(n)
        sigma[s] = 1.0
        dist = np.full(n, -1)
        dist[s] = 0
        queue = [s]
        qi = 0

        while qi < len(queue):
            v = queue[qi]
            qi += 1
            stack.append(v)
            for w in range(n):
                if not adj[v, w]:
                    continue
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    predecessors[w].append(v)

        delta = np.zeros(n)
        while stack:
            w = stack.pop()
            for v in predecessors[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                centrality[w] += delta[w]

    # Normalize
    norm = (n - 1) * (n - 2)
    if norm > 0:
        centrality /= norm

    return centrality


# =============================================================================
# PERCOLATION ANALYSIS
# =============================================================================

@dataclass
class PercolationResult:
    """Results from percolation threshold analysis."""
    thresholds: np.ndarray          # energy thresholds tested
    n_components: np.ndarray        # number of connected components at each threshold
    largest_component_size: np.ndarray  # size of largest component
    fraction_connected: np.ndarray  # fraction of nodes in largest component
    percolation_threshold: float    # estimated critical threshold
    giant_component_onset: float    # threshold where largest component > 50% of nodes


def compute_percolation_sweep(
    intermap: np.ndarray,
    sticker_mask: Union[np.ndarray, StickerMask],
    n_thresholds: int = 50,
    threshold_range: Optional[Tuple[float, float]] = None,
    min_separation: int = 4,
) -> PercolationResult:
    """
    Sweep energy thresholds and track connectivity transition.

    This maps: percolation threshold <-> LLPS onset.
    The critical threshold where the network becomes connected
    is an indicator of phase separation propensity.

    Parameters
    ----------
    intermap : np.ndarray
        NxN interaction map
    sticker_mask : Union[np.ndarray, StickerMask]
        Sticker mask
    n_thresholds : int
        Number of thresholds to test
    threshold_range : Optional[Tuple[float, float]]
        (min, max) threshold range. Auto-determined if None.
    min_separation : int
        Minimum sequence separation for edges

    Returns
    -------
    PercolationResult
        Percolation analysis results
    """
    if isinstance(sticker_mask, StickerMask):
        positions = sticker_mask.positions
    else:
        positions = np.where(sticker_mask)[0]

    n_stickers = len(positions)

    if n_stickers < 2:
        thresholds = np.array([0.0])
        return PercolationResult(
            thresholds=thresholds,
            n_components=np.array([n_stickers]),
            largest_component_size=np.array([n_stickers]),
            fraction_connected=np.array([1.0]),
            percolation_threshold=0.0,
            giant_component_onset=0.0,
        )

    # Extract sticker-sticker submatrix
    sub_intermap = intermap[np.ix_(positions, positions)]
    sep = np.abs(positions[:, None] - positions[None, :])
    valid_pairs = (sep >= min_separation)
    np.fill_diagonal(valid_pairs, False)

    # Determine threshold range from sticker-sticker energies
    valid_energies = sub_intermap[valid_pairs]
    if len(valid_energies) == 0 or threshold_range is not None:
        if threshold_range is None:
            threshold_range = (-1.0, 0.0)
        thresholds = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)
    else:
        e_min = float(np.min(valid_energies))
        e_max = float(np.max(valid_energies))
        margin = 0.1 * (e_max - e_min) if e_max > e_min else 0.1
        thresholds = np.linspace(e_min - margin, e_max + margin, n_thresholds)

    n_components_arr = np.zeros(n_thresholds, dtype=int)
    largest_arr = np.zeros(n_thresholds, dtype=int)
    frac_arr = np.zeros(n_thresholds)

    for ti, thresh in enumerate(thresholds):
        adj = (sub_intermap < thresh) & valid_pairs
        components = _find_connected_components(adj)
        n_comp = len(components) if components else n_stickers
        largest = max(len(c) for c in components) if components else 0

        n_components_arr[ti] = n_comp
        largest_arr[ti] = largest
        frac_arr[ti] = largest / n_stickers if n_stickers > 0 else 0.0

    # Find percolation threshold: where n_components transitions
    # Use the threshold where fraction_connected first exceeds 0.5
    giant_onset = thresholds[0]
    for i, frac in enumerate(frac_arr):
        if frac >= 0.5:
            giant_onset = thresholds[i]
            break

    # Percolation threshold: steepest change in largest component fraction
    if len(frac_arr) > 1:
        dfrac = np.diff(frac_arr)
        perc_idx = np.argmax(dfrac)
        percolation_threshold = float(thresholds[perc_idx])
    else:
        percolation_threshold = float(thresholds[0])

    return PercolationResult(
        thresholds=thresholds,
        n_components=n_components_arr,
        largest_component_size=largest_arr,
        fraction_connected=frac_arr,
        percolation_threshold=percolation_threshold,
        giant_component_onset=float(giant_onset),
    )


# =============================================================================
# CONNECTED COMPONENTS (UNION-FIND)
# =============================================================================

def _find_connected_components(adjacency: np.ndarray) -> List[set]:
    """Find connected components using BFS."""
    n = adjacency.shape[0]
    visited = set()
    components = []

    for start in range(n):
        if start in visited:
            continue
        # BFS
        component = set()
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            for neighbor in range(n):
                if adjacency[node, neighbor] and neighbor not in visited:
                    queue.append(neighbor)
        if component:
            components.append(component)

    return components


def count_connected_components(graph: ContactGraph) -> int:
    """Count number of connected components."""
    components = _find_connected_components(graph.adjacency)
    return len(components)


def _bfs_distances(adjacency: np.ndarray, source: int, node_set: set) -> Dict[int, int]:
    """BFS shortest distances from source within node_set."""
    distances = {source: 0}
    queue = [source]
    qi = 0

    while qi < len(queue):
        node = queue[qi]
        qi += 1
        for neighbor in range(adjacency.shape[0]):
            if (adjacency[node, neighbor] and
                    neighbor in node_set and
                    neighbor not in distances):
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)

    return distances


# =============================================================================
# COMPREHENSIVE TOPOLOGY METRICS
# =============================================================================

@dataclass
class TopologyMetrics:
    """Complete topology metrics for a variant's interaction network."""
    # Full contact graph metrics
    n_contacts: int
    graph_density: float
    mean_degree: float
    max_degree: float

    # Sticker subgraph metrics
    sticker_n_contacts: int
    sticker_graph_density: float
    sticker_clustering_coefficient: float
    sticker_mean_degree: float
    sticker_avg_path_length: float
    sticker_n_components: int

    # Percolation
    percolation_threshold: float
    giant_component_onset: float

    # Centrality summary
    max_betweenness: float
    mean_betweenness: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "n_contacts": self.n_contacts,
            "graph_density": self.graph_density,
            "mean_degree": self.mean_degree,
            "max_degree": self.max_degree,
            "sticker_n_contacts": self.sticker_n_contacts,
            "sticker_graph_density": self.sticker_graph_density,
            "sticker_clustering_coefficient": self.sticker_clustering_coefficient,
            "sticker_mean_degree": self.sticker_mean_degree,
            "sticker_avg_path_length": self.sticker_avg_path_length,
            "sticker_n_components": self.sticker_n_components,
            "percolation_threshold": self.percolation_threshold,
            "giant_component_onset": self.giant_component_onset,
            "max_betweenness": self.max_betweenness,
            "mean_betweenness": self.mean_betweenness,
        }


def auto_threshold(
    intermap: np.ndarray,
    sticker_mask: Union[np.ndarray, StickerMask],
    percentile: float = 25.0,
    min_separation: int = 4,
) -> float:
    """
    Determine an adaptive energy threshold from the interaction map.

    Uses the given percentile of sticker-sticker interaction energies
    (excluding sequence-local pairs) as the threshold.

    Parameters
    ----------
    intermap : np.ndarray
        NxN interaction map
    sticker_mask : Union[np.ndarray, StickerMask]
        Sticker mask
    percentile : float
        Percentile of sticker-sticker energies to use as threshold
    min_separation : int
        Minimum sequence separation

    Returns
    -------
    float
        Adaptive energy threshold
    """
    if isinstance(sticker_mask, StickerMask):
        positions = sticker_mask.positions
    else:
        positions = np.where(sticker_mask)[0]

    if len(positions) < 2:
        return 0.0

    sub = intermap[np.ix_(positions, positions)]
    sep = np.abs(positions[:, None] - positions[None, :])
    valid = (sep >= min_separation)
    np.fill_diagonal(valid, False)
    upper = np.triu(valid, k=1)

    values = sub[upper]
    if len(values) == 0:
        return 0.0

    return float(np.percentile(values, percentile))


def compute_topology_metrics(
    intermap: np.ndarray,
    sticker_mask: Union[np.ndarray, StickerMask],
    threshold: Optional[float] = None,
    min_separation: int = 4,
    sequence: Optional[str] = None,
) -> TopologyMetrics:
    """
    Compute all topology metrics for a variant.

    Parameters
    ----------
    intermap : np.ndarray
        NxN interaction map
    sticker_mask : Union[np.ndarray, StickerMask]
        Sticker mask
    threshold : Optional[float]
        Energy threshold for contact edges. If None, uses adaptive
        threshold (25th percentile of sticker-sticker energies).
    min_separation : int
        Minimum sequence separation
    sequence : Optional[str]
        Amino acid sequence

    Returns
    -------
    TopologyMetrics
        Complete topology analysis
    """
    # Auto-determine threshold from data if not provided
    if threshold is None:
        threshold = auto_threshold(intermap, sticker_mask, min_separation=min_separation)

    # Full contact graph
    full_graph = build_contact_graph(
        intermap, threshold=threshold,
        min_separation=min_separation, sequence=sequence
    )
    full_deg = compute_degree_distribution(full_graph)

    # Sticker subgraph
    sticker_graph = build_sticker_contact_graph(
        intermap, sticker_mask, threshold=threshold,
        min_separation=min_separation, sequence=sequence
    )
    sticker_cc = compute_clustering_coefficient(sticker_graph)
    sticker_deg = compute_degree_distribution(sticker_graph)
    sticker_apl = compute_average_path_length(sticker_graph)
    sticker_n_comp = count_connected_components(sticker_graph)

    # Percolation
    perc = compute_percolation_sweep(
        intermap, sticker_mask, min_separation=min_separation
    )

    # Centrality (on sticker graph)
    betweenness = compute_betweenness_centrality(sticker_graph)

    return TopologyMetrics(
        n_contacts=full_graph.n_edges,
        graph_density=full_graph.density,
        mean_degree=full_deg["mean_degree"],
        max_degree=full_deg["max_degree"],
        sticker_n_contacts=sticker_graph.n_edges,
        sticker_graph_density=sticker_graph.density,
        sticker_clustering_coefficient=sticker_cc,
        sticker_mean_degree=sticker_deg["mean_degree"],
        sticker_avg_path_length=sticker_apl if not np.isnan(sticker_apl) else 0.0,
        sticker_n_components=sticker_n_comp,
        percolation_threshold=perc.percolation_threshold,
        giant_component_onset=perc.giant_component_onset,
        max_betweenness=float(np.max(betweenness)) if len(betweenness) > 0 else 0.0,
        mean_betweenness=float(np.mean(betweenness)) if len(betweenness) > 0 else 0.0,
    )
