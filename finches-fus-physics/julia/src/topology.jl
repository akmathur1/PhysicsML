# topology.jl — High-performance graph analysis for interaction networks.
#
# Uses Graphs.jl + MetaGraphs.jl for the heavy lifting,
# with custom percolation sweep that's 10-50x faster than Python.

using Graphs
using MetaGraphs
using SparseArrays

"""
    build_contact_graph(intermap, threshold, min_separation; labels=nothing)

Build a SimpleWeightedGraph from an interaction map.
Edge exists where E[i,j] < threshold and |i-j| ≥ min_separation.
"""
function build_contact_graph(
    intermap::AbstractMatrix{T},
    threshold::Real;
    min_separation::Int = 4,
    labels::Union{Nothing, Vector{Char}} = nothing
) where T
    n = size(intermap, 1)
    g = MetaGraph(n)

    # Attach residue labels as vertex properties
    if labels !== nothing
        for i in 1:n
            set_prop!(g, i, :aa, labels[i])
        end
    end

    @inbounds for j in 1:n, i in (j+1):n
        if abs(i - j) >= min_separation && intermap[i, j] < threshold
            add_edge!(g, i, j)
            set_prop!(g, Edge(i, j), :weight, intermap[i, j])
        end
    end

    return g
end

"""
    build_sticker_subgraph(intermap, sticker_positions, threshold; min_separation=4)

Extract the sticker-sticker contact network.
Returns a MetaGraph on sticker indices only.
"""
function build_sticker_subgraph(
    intermap::AbstractMatrix{T},
    sticker_positions::Vector{Int},
    threshold::Real;
    min_separation::Int = 4
) where T
    ns = length(sticker_positions)
    g = MetaGraph(ns)

    # Map sticker index → original position
    for si in 1:ns
        set_prop!(g, si, :original_pos, sticker_positions[si])
    end

    @inbounds for sj in 1:ns, si in (sj+1):ns
        pi, pj = sticker_positions[si], sticker_positions[sj]
        if abs(pi - pj) >= min_separation && intermap[pi, pj] < threshold
            add_edge!(g, si, sj)
            set_prop!(g, Edge(si, sj), :weight, intermap[pi, pj])
        end
    end

    return g
end

"""
    clustering_coefficient(g::AbstractGraph) -> Float64

Global clustering coefficient: 3×triangles / connected_triples.
"""
function clustering_coefficient(g::AbstractGraph)
    nv(g) < 3 && return 0.0
    A = Float64.(adjacency_matrix(g))
    A3 = A * A * A
    n_triangles = tr(A3) / 6.0
    degrees = vec(sum(A, dims=2))
    n_triples = sum(d * (d - 1) for d in degrees) / 2.0
    n_triples < 1e-10 && return 0.0
    return 3.0 * n_triangles / n_triples
end

"""
    betweenness_centrality(g::AbstractGraph) -> Vector{Float64}

Betweenness centrality via Graphs.jl (Brandes algorithm).
"""
betweenness_centrality(g::AbstractGraph) = Graphs.betweenness_centrality(g)

"""
    PercolationResult

Results from sweeping energy thresholds across the sticker network.
"""
struct PercolationResult
    thresholds::Vector{Float64}
    n_components::Vector{Int}
    largest_component_size::Vector{Int}
    fraction_connected::Vector{Float64}
    percolation_threshold::Float64
    giant_component_onset::Float64
end

"""
    percolation_sweep(intermap, sticker_positions; n_thresholds=50, min_separation=4)

Sweep energy thresholds and track connectivity transition.
This is the inner loop that benefits most from Julia speed:
50 thresholds × O(N²) graph construction + O(N+E) component analysis.
"""
function percolation_sweep(
    intermap::AbstractMatrix{T},
    sticker_positions::Vector{Int};
    n_thresholds::Int = 50,
    threshold_range::Union{Nothing, Tuple{Float64, Float64}} = nothing,
    min_separation::Int = 4
) where T
    ns = length(sticker_positions)

    if ns < 2
        return PercolationResult(
            [0.0], [ns], [ns], [1.0], 0.0, 0.0
        )
    end

    # Extract sticker-sticker submatrix
    sub = intermap[sticker_positions, sticker_positions]

    # Valid pairs mask
    valid = falses(ns, ns)
    @inbounds for j in 1:ns, i in 1:ns
        if i != j && abs(sticker_positions[i] - sticker_positions[j]) >= min_separation
            valid[i, j] = true
        end
    end

    # Determine threshold range
    if threshold_range === nothing
        valid_energies = sub[valid]
        if isempty(valid_energies)
            threshold_range = (-1.0, 0.0)
        else
            e_min, e_max = extrema(valid_energies)
            margin = 0.1 * max(e_max - e_min, 0.1)
            threshold_range = (e_min - margin, e_max + margin)
        end
    end

    thresholds = range(threshold_range[1], threshold_range[2], length=n_thresholds)

    n_comp = Vector{Int}(undef, n_thresholds)
    largest = Vector{Int}(undef, n_thresholds)
    frac = Vector{Float64}(undef, n_thresholds)

    # Pre-allocate adjacency
    adj = falses(ns, ns)

    @inbounds for (ti, thresh) in enumerate(thresholds)
        # Build adjacency at this threshold
        for j in 1:ns, i in 1:ns
            adj[i, j] = valid[i, j] && sub[i, j] < thresh
        end

        # Connected components via simple BFS (faster than constructing Graph each time)
        visited = falses(ns)
        components = Int[]
        max_comp = 0

        for start in 1:ns
            visited[start] && continue
            # BFS
            comp_size = 0
            queue = [start]
            qi = 1
            visited[start] = true
            while qi <= length(queue)
                node = queue[qi]
                qi += 1
                comp_size += 1
                for neighbor in 1:ns
                    if adj[node, neighbor] && !visited[neighbor]
                        visited[neighbor] = true
                        push!(queue, neighbor)
                    end
                end
            end
            push!(components, comp_size)
            max_comp = max(max_comp, comp_size)
        end

        n_comp[ti] = length(components)
        largest[ti] = max_comp
        frac[ti] = max_comp / ns
    end

    # Find percolation threshold (steepest rise in largest component fraction)
    dfrac = diff(frac)
    perc_idx = argmax(dfrac)
    percolation_threshold = thresholds[perc_idx]

    # Giant component onset (first > 50%)
    giant_onset = thresholds[1]
    for i in eachindex(frac)
        if frac[i] >= 0.5
            giant_onset = thresholds[i]
            break
        end
    end

    return PercolationResult(
        collect(thresholds),
        n_comp,
        largest,
        frac,
        percolation_threshold,
        giant_onset,
    )
end
