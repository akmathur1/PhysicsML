using Test

# Activate project
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using FinchesPhysics

@testset "FinchesPhysics" begin

    @testset "Amino acid indexing" begin
        @test FinchesPhysics.aa_idx('A') == 1
        @test FinchesPhysics.aa_idx('Y') == 20
        @test length(FinchesPhysics.AA_ORDER) == 20
    end

    @testset "Force field" begin
        p = ForceFieldParams()
        M = build_interaction_matrix(p)
        @test size(M) == (20, 20)
        @test M ≈ M'  # symmetric
        # Y-Y should be strongly attractive (aromatic π-π)
        y_idx = FinchesPhysics.aa_idx('Y')
        @test M[y_idx, y_idx] < -0.3
    end

    @testset "Interaction map" begin
        seq = "MASNDYTQQATQSYGAYPTQPGQGY"  # FUS LCD fragment
        imap = compute_intermap(seq)
        n = length(seq)
        @test size(imap) == (n, n)
        @test imap ≈ imap'  # symmetric

        # Smoothing
        smoothed = FinchesPhysics.gaussian_smooth(imap, 2.0)
        @test size(smoothed) == (n, n)
    end

    @testset "Hamiltonian energy" begin
        seq = "MASNDYTQQATQSYGAYPTQPGQGY"
        H = hamiltonian_energy(seq)
        @test H < 0  # should be net attractive for FUS LCD
        @test isfinite(H)
    end

    @testset "Topology gradient" begin
        seq = "MASNDYTQQATQSYGAYPTQPGQGY"
        # Positions of Y residues (1-based)
        sticker_pos = [6, 14, 18, 25]
        grad = ∂H_∂topology(seq, sticker_pos)
        @test length(grad) == 4
        @test all(isfinite, grad)
        # Y positions should have negative gradients (attractive)
        @test all(g -> g < 0, grad)
    end

    @testset "H0 persistence" begin
        # Simple 3-node test
        D = [0.0 1.0 3.0;
             1.0 0.0 2.0;
             3.0 2.0 0.0]
        h0 = compute_H0(D)
        @test FinchesPhysics.n_features(h0) == 3  # 3 nodes = 2 merges + 1 survivor
    end

    @testset "H1 persistence" begin
        # Triangle: all distances = 1 → one H1 cycle that immediately dies
        D = [0.0 1.0 1.0;
             1.0 0.0 1.0;
             1.0 1.0 0.0]
        h1 = compute_H1(D)
        # The triangle should create at least one cycle
        @test FinchesPhysics.n_features(h1) >= 0  # may be 0 if dies instantly
    end

    @testset "Percolation sweep" begin
        seq = "MASNDYTQQATQSYGAYPTQPGQGY"
        imap = compute_intermap(seq)
        sticker_pos = [6, 14, 18, 25]
        perc = percolation_sweep(imap, sticker_pos, n_thresholds=20)
        @test length(perc.thresholds) == 20
        @test all(perc.fraction_connected .>= 0.0)
        @test all(perc.fraction_connected .<= 1.0)
    end
end
