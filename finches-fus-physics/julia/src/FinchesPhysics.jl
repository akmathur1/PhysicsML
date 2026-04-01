module FinchesPhysics

using LinearAlgebra
using SparseArrays
using StaticArrays

include("amino_acids.jl")
include("hamiltonian.jl")
include("topology.jl")
include("homology.jl")
include("bridge.jl")

export
    # Amino acid data
    AA_ORDER, AA_INDEX, N_AA,
    # Hamiltonian
    ForceFieldParams, build_interaction_matrix, compute_intermap,
    compute_intermap!, hamiltonian_energy, ∂H_∂topology,
    # Topology
    build_contact_graph, percolation_sweep, clustering_coefficient,
    betweenness_centrality, connected_components,
    # Homology
    compute_H0, compute_H1, betti_numbers,
    # Bridge
    intermap_from_python, topology_from_python, homology_from_python

end # module
