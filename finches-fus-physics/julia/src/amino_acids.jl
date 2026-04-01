# amino_acids.jl — Amino acid properties and indexing for MPIPI-GG force field.
#
# All properties stored as StaticArrays for zero-allocation lookups
# and Zygote compatibility.

using StaticArrays

const AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
const N_AA = 20

# Build compile-time lookup: Char → index (1-based)
const AA_INDEX = let
    d = Dict{Char, Int}()
    for (i, c) in enumerate(AA_ORDER)
        d[c] = i
    end
    d
end

@inline aa_idx(c::Char) = AA_INDEX[c]
@inline aa_idx(s::AbstractString, i::Int) = AA_INDEX[s[i]]

# Residue properties as SVector{20, Float64} for AD compatibility
# Order matches AA_ORDER: A C D E F G H I K L M N P Q R S T V W Y

const HYDROPHOBICITY = SVector{20, Float64}(
    0.17, 0.24, -0.78, -0.64, 0.61,   # A C D E F
    0.01, -0.40, 0.73, -1.10, 0.53,    # G H I K L
    0.26, -0.60, -0.07, -0.69, -1.80,  # M N P Q R
   -0.26, -0.18, 0.54, 0.37, 0.02,    # S T V W Y
)

const CHARGE = SVector{20, Float64}(
    0, 0, -1, -1, 0,   # A C D E F
    0, 0,  0, +1, 0,   # G H I K L
    0, 0,  0,  0, +1,  # M N P Q R
    0, 0,  0,  0,  0,  # S T V W Y
)

const AROMATICITY = SVector{20, Float64}(
    0.0, 0.0, 0.0, 0.0, 1.0,   # A C D E F
    0.0, 0.5, 0.0, 0.0, 0.0,   # G H I K L
    0.0, 0.0, 0.0, 0.0, 0.0,   # M N P Q R
    0.0, 0.0, 0.0, 1.0, 0.9,   # S T V W Y
)

const CATIONICITY = SVector{20, Float64}(
    0.0, 0.0, 0.0, 0.0, 0.0,   # A C D E F
    0.0, 0.3, 0.0, 1.0, 0.0,   # G H I K L
    0.0, 0.0, 0.0, 0.0, 1.0,   # M N P Q R
    0.0, 0.0, 0.0, 0.0, 0.0,   # S T V W Y
)

# Boolean masks for polar residues (S=16, T=17, N=12, Q=14 in 1-indexed)
const POLAR_MASK = SVector{20, Bool}(
    false, false, false, false, false,  # A C D E F
    false, false, false, false, false,  # G H I K L
    false, true,  false, true,  false,  # M N P Q R
    true,  true,  false, false, false,  # S T V W Y
)

# Glycine index (6th in AA_ORDER)
const GLY_IDX = 6
# Proline index (13th in AA_ORDER)
const PRO_IDX = 13

"""
    seq_to_indices(sequence::AbstractString) -> Vector{Int}

Convert amino acid sequence to 1-based index array.
"""
function seq_to_indices(sequence::AbstractString)
    n = length(sequence)
    idx = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        idx[i] = AA_INDEX[sequence[i]]
    end
    return idx
end
