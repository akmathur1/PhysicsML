# bridge.jl — PythonCall bridge functions.
#
# These are the entry points called from Python via juliacall.
# They accept/return plain arrays and dicts (no custom types crossing the boundary).

"""
    intermap_from_python(sequence, params_dict; sigma=2.0, smooth=true) -> Matrix

Compute interaction map callable from Python.
params_dict: Dict with force field parameter overrides (or empty for defaults).
"""
function intermap_from_python(
    sequence::String,
    params_dict::Dict{String, Any} = Dict{String, Any}();
    sigma::Float64 = 2.0,
    smooth::Bool = true
)
    # Build params from dict
    p = ForceFieldParams(;
        ε_aromatic     = get(params_dict, "epsilon_aromatic",   -0.65),
        ε_cation_pi    = get(params_dict, "epsilon_cation_pi",  -0.50),
        ε_attractive   = get(params_dict, "epsilon_attractive", -0.30),
        ε_repulsive    = get(params_dict, "epsilon_repulsive",   0.20),
        ε_hydrophobic  = get(params_dict, "epsilon_hydrophobic",-0.15),
        ε_polar        = get(params_dict, "epsilon_polar",      -0.05),
        glycine_factor = get(params_dict, "glycine_factor",      0.30),
        proline_penalty= get(params_dict, "proline_penalty",     0.10),
    )

    imap = compute_intermap(sequence, p)

    if smooth
        imap = gaussian_smooth(imap, sigma)
    end

    return imap
end

"""
    topology_from_python(intermap, sticker_positions, threshold; min_separation=4) -> Dict

Compute topology metrics callable from Python.
Returns a flat Dict{String, Any} for easy conversion.
"""
function topology_from_python(
    intermap::Matrix{Float64},
    sticker_positions::Vector{Int},
    threshold::Float64;
    min_separation::Int = 4,
    n_percolation_thresholds::Int = 50
)
    # Build sticker subgraph
    g = build_sticker_subgraph(intermap, sticker_positions, threshold;
                                min_separation=min_separation)

    cc = clustering_coefficient(g)
    bc = Graphs.betweenness_centrality(g)
    deg = degree(g)
    n_comp = length(connected_components(g))
    apl = if nv(g) >= 2 && is_connected(g)
        Float64(Graphs.mean(Graphs.dijkstra_shortest_paths(g, 1).dists))
    else
        0.0
    end

    # Percolation sweep
    perc = percolation_sweep(intermap, sticker_positions;
                             n_thresholds=n_percolation_thresholds,
                             min_separation=min_separation)

    return Dict{String, Any}(
        "sticker_clustering_coefficient" => cc,
        "sticker_mean_degree" => isempty(deg) ? 0.0 : Float64(mean(deg)),
        "sticker_max_degree" => isempty(deg) ? 0.0 : Float64(maximum(deg)),
        "sticker_n_contacts" => ne(g),
        "sticker_graph_density" => density(g),
        "sticker_n_components" => n_comp,
        "sticker_avg_path_length" => apl,
        "max_betweenness" => isempty(bc) ? 0.0 : maximum(bc),
        "mean_betweenness" => isempty(bc) ? 0.0 : mean(bc),
        "percolation_threshold" => perc.percolation_threshold,
        "giant_component_onset" => perc.giant_component_onset,
        "percolation_thresholds" => perc.thresholds,
        "percolation_fraction_connected" => perc.fraction_connected,
    )
end

"""
    homology_from_python(intermap, sticker_positions; min_separation=4) -> Dict

Compute persistent homology metrics callable from Python.
"""
function homology_from_python(
    intermap::Matrix{Float64},
    sticker_positions::Vector{Int};
    distance_method::Symbol = :negated,
    min_separation::Int = 4
)
    dist = interaction_to_distance(intermap, sticker_positions;
                                    method=distance_method,
                                    min_separation=min_separation)

    h0 = compute_H0(dist)
    h1 = compute_H1(dist)
    thresholds, β0, β1 = betti_numbers(dist)

    # Betti curve AUC
    b0_auc = 0.0
    b1_auc = 0.0
    if length(thresholds) > 1
        dt = diff(thresholds)
        b0_auc = sum(β0[1:end-1] .* dt)
        b1_auc = sum(β1[1:end-1] .* dt)
    end

    # H0 mean death (finite only)
    h0_deaths = [p.death for p in h0.pairs if isfinite(p.death)]
    h0_mean_death = isempty(h0_deaths) ? 0.0 : mean(h0_deaths)

    return Dict{String, Any}(
        "h0_total_persistence" => total_persistence(h0),
        "h0_max_persistence" => max_persistence(h0),
        "h0_n_features" => n_features(h0),
        "h0_mean_death" => h0_mean_death,
        "h1_total_persistence" => total_persistence(h1),
        "h1_max_persistence" => max_persistence(h1),
        "h1_n_features" => n_features(h1),
        "betti_0_auc" => b0_auc,
        "betti_1_auc" => b1_auc,
        "betti_thresholds" => thresholds,
        "betti_0" => β0,
        "betti_1" => β1,
    )
end
