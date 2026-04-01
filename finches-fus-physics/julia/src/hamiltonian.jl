# hamiltonian.jl — Differentiable MPIPI-GG Hamiltonian.
#
# Key design decisions:
# 1. All parameters live in a struct so Zygote can differentiate through them.
# 2. The interaction matrix is built functionally (no mutation) for AD.
# 3. Intermap generation uses vectorized indexing — O(N²) but fast.
# 4. ∂H/∂topology is the big unlock: gradient of Hamiltonian w.r.t.
#    topological features (sticker positions, graph connectivity).

using LinearAlgebra

"""
    ForceFieldParams

MPIPI-GG style force field parameters. All energies in kT.
Mutable only for convenience; the Hamiltonian evaluation is pure-functional.
"""
Base.@kwdef struct ForceFieldParams{T<:Real}
    ε_aromatic::T     = -0.65   # π-π stacking: Y-Y, F-F, W-W
    ε_cation_pi::T    = -0.50   # cation-π: R/K with Y/F/W
    ε_attractive::T   = -0.30   # opposite charge: R/K with D/E
    ε_repulsive::T    = +0.20   # like charges
    ε_hydrophobic::T  = -0.15   # hydrophobic-hydrophobic
    ε_polar::T        = -0.05   # weak polar (S/T/N/Q pairs)
    ε_background::T   =  0.00   # neutral baseline
    glycine_factor::T =  0.30   # glycine weakens neighbors
    proline_penalty::T = +0.10  # proline disrupts
end

"""
    pairwise_energy(i::Int, j::Int, p::ForceFieldParams)

Compute interaction energy between amino acids at indices i, j.
Written without mutation for Zygote compatibility.
"""
function pairwise_energy(i::Int, j::Int, p::ForceFieldParams{T}) where T
    arom_i, arom_j = AROMATICITY[i], AROMATICITY[j]
    cat_i, cat_j   = CATIONICITY[i], CATIONICITY[j]
    q_i, q_j       = CHARGE[i], CHARGE[j]
    h_i, h_j       = HYDROPHOBICITY[i], HYDROPHOBICITY[j]

    # 1. Aromatic-aromatic (π-π stacking)
    E = arom_i * arom_j * p.ε_aromatic

    # 2. Cation-π
    E += (cat_i * arom_j + cat_j * arom_i) * p.ε_cation_pi

    # 3. Electrostatic (smooth version for AD — no branching)
    #    sign(q_i*q_j) = +1 for like, -1 for opposite, 0 for neutral
    qq = q_i * q_j
    has_charge = (abs(q_i) > 0.5) * (abs(q_j) > 0.5)  # smooth indicator
    E += has_charge * (
        (qq < zero(T)) * p.ε_attractive +
        (qq > zero(T)) * p.ε_repulsive
    )

    # 4. Hydrophobic
    both_hydrophobic = (h_i > 0.2) * (h_j > 0.2)
    E += both_hydrophobic * sqrt(max(h_i * h_j, zero(T))) * p.ε_hydrophobic

    # 5. Polar
    E += POLAR_MASK[i] * POLAR_MASK[j] * p.ε_polar

    # 6. Glycine weakening
    is_gly = (i == GLY_IDX) | (j == GLY_IDX)
    E = is_gly ? E * p.glycine_factor : E

    # 7. Proline penalty
    is_pro = (i == PRO_IDX) | (j == PRO_IDX)
    E += is_pro * p.proline_penalty

    return E
end

"""
    build_interaction_matrix(p::ForceFieldParams) -> Matrix{Float64}

Build the 20×20 residue-residue interaction energy matrix.
Symmetric: E[i,j] = E[j,i].
"""
function build_interaction_matrix(p::ForceFieldParams{T} = ForceFieldParams()) where T
    M = Matrix{T}(undef, N_AA, N_AA)
    @inbounds for j in 1:N_AA, i in 1:N_AA
        M[i, j] = pairwise_energy(i, j, p)
    end
    # Enforce symmetry
    return T(0.5) .* (M .+ M')
end

"""
    compute_intermap(sequence::AbstractString, p::ForceFieldParams) -> Matrix

Compute NxN interaction map for a protein sequence.
Fully vectorized via index broadcasting.
"""
function compute_intermap(sequence::AbstractString, p::ForceFieldParams = ForceFieldParams())
    idx = seq_to_indices(sequence)
    M = build_interaction_matrix(p)
    return M[idx, idx]  # Advanced indexing: (N,N) from (20,20)
end

"""
    compute_intermap!(out::Matrix, idx::Vector{Int}, M::Matrix)

In-place intermap computation for pre-allocated buffers.
Use this in hot loops (parameter sweeps, Monte Carlo).
"""
function compute_intermap!(out::Matrix{T}, idx::Vector{Int}, M::Matrix{T}) where T
    n = length(idx)
    @inbounds for j in 1:n, i in 1:n
        out[i, j] = M[idx[i], idx[j]]
    end
    return out
end

"""
    hamiltonian_energy(sequence::AbstractString, p::ForceFieldParams;
                       min_separation::Int=4) -> Float64

Total Hamiltonian energy: sum of pairwise interactions beyond min_separation.
This is the quantity to differentiate w.r.t. parameters or topology features.

H = Σ_{|i-j| ≥ s} E(aa_i, aa_j; p)
"""
function hamiltonian_energy(
    sequence::AbstractString,
    p::ForceFieldParams = ForceFieldParams();
    min_separation::Int = 4
)
    imap = compute_intermap(sequence, p)
    n = size(imap, 1)
    H = zero(eltype(imap))
    @inbounds for j in 1:n, i in 1:n
        if abs(i - j) >= min_separation
            H += imap[i, j]
        end
    end
    # Each pair counted twice (symmetric), divide by 2
    return H / 2
end

"""
    ∂H_∂topology(sequence, sticker_mask, p; kwargs...)

Gradient of Hamiltonian w.r.t. topological features.

This is the key differentiable-physics unlock:
- How does total network energy change with sticker rearrangement?
- Which sticker positions most affect the Hamiltonian?

Returns a vector of ∂H/∂(sticker_i) for each sticker position,
computed as the sum of interaction energies involving that sticker.
"""
function ∂H_∂topology(
    sequence::AbstractString,
    sticker_positions::Vector{Int},
    p::ForceFieldParams = ForceFieldParams();
    min_separation::Int = 4
)
    imap = compute_intermap(sequence, p)
    n = size(imap, 1)
    ns = length(sticker_positions)

    grad = zeros(eltype(imap), ns)

    @inbounds for (si, pos) in enumerate(sticker_positions)
        # ∂H/∂(sticker at pos) = sum of interactions involving pos
        # beyond min_separation
        for j in 1:n
            if abs(pos - j) >= min_separation
                grad[si] += imap[pos, j]
            end
        end
    end

    return grad
end

# ─── Gaussian smoothing (pure Julia, no deps) ───────────────────────────

"""
    gaussian_smooth(M::Matrix, σ::Real) -> Matrix

2D Gaussian smoothing of interaction map. Simple separable filter.
"""
function gaussian_smooth(M::AbstractMatrix{T}, σ::Real) where T
    σ <= 0 && return copy(M)
    r = ceil(Int, 3σ)
    kernel = [exp(-x^2 / (2σ^2)) for x in -r:r]
    kernel ./= sum(kernel)

    # Separable: smooth rows, then columns
    n, m = size(M)
    tmp = similar(M)
    out = similar(M)

    # Smooth along rows
    @inbounds for i in 1:n
        for j in 1:m
            s = zero(T)
            w = zero(T)
            for k in -r:r
                jj = clamp(j + k, 1, m)
                s += kernel[k + r + 1] * M[i, jj]
                w += kernel[k + r + 1]
            end
            tmp[i, j] = s / w
        end
    end

    # Smooth along columns
    @inbounds for j in 1:m
        for i in 1:n
            s = zero(T)
            w = zero(T)
            for k in -r:r
                ii = clamp(i + k, 1, n)
                s += kernel[k + r + 1] * tmp[ii, j]
                w += kernel[k + r + 1]
            end
            out[i, j] = s / w
        end
    end

    return out
end
