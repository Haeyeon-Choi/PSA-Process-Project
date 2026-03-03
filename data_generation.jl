using Pkg
Pkg.activate(".")

using PSASimulator
using DataFrames
using CSV
using QuasiMonteCarlo
using Random
using Base.Threads

# ----------------------------------------------------------
# Load demo data (material + isotherm parameters)
# ----------------------------------------------------------
include(joinpath(dirname(dirname(pathof(PSASimulator))),
                 "demo", "demo_data.jl"))

println("Running with $(Threads.nthreads()) threads")

# ==========================================================
# FAST forward dataset generator
# ==========================================================

function generate_dataset(mat_index; n_samples=200, N=4)

    filepath = joinpath(@__DIR__, "data", "dataset_material16_1000.csv")
    mkpath(dirname(filepath))

    # 파일 새로 만들기 + header
    open(filepath, "w") do io
        println(io, "P0,ndot,tads,alpha,beta,productivity,energy,purity,recovery")
    end

    println("Running with $(nthreads()) threads")

    # -------------------------------
    # Material setup
    # -------------------------------
    material_property = vec(SIMULATION_PARAMETERS[mat_index, :])
    isoPar = vec(ISOTHERM_PARAMETERS[mat_index, :])
    material = (material_property, isoPar)

    # -------------------------------
    # Sampling bounds
    # -------------------------------
    lb = [2.0e5, 0.5, 150.0, 0.15, 0.15]
    ub = [6.0e5, 2.0, 500.0, 0.35, 0.4]

    Random.seed!(1234)

    X = QuasiMonteCarlo.sample(
        n_samples,
        lb,
        ub,
        SobolSample()
    )
    # do it sequentially
    perm = sortperm(X[1, :])
    X = X[:, perm]  

    # -------------------------------
    # Fixed parameters
    # -------------------------------
    L  = 1.0    # fix geometry
    PI = 1e4    # fix intermediate pressure
    Pl = 1e4    # fix low pressure; P0 > PI >= Pl

    # ==========================================================
    # Multithreaded simulation loop
    # ==========================================================
    # open("data/dataset_material16.csv", "a") do io
    open(filepath, "a") do io2
        # @threads for i in 1:n_samples --> for multithread calculation
        for i in 1:n_samples

            P0, ndot, tads, alpha, beta = X[:, i]
            vars = [L, P0, ndot, tads, alpha, beta, PI, Pl]

            try
                res = PSASimulator.psacycle(
                    vars,
                    material;
                    N=N,
                    it_disp=false,
                    run_type=:EconomicEvaluation
                )

                prod   = -res.objectives[1]
                energy =  res.objectives[2]
                purity =  res.traj[:purity]
                recovery = res.traj[:recovery]

                println(io2, "$P0,$ndot,$tads,$alpha,$beta,$prod,$energy,$purity,$recovery")
                # println("saved sample,", "$P0,$ndot,$tads,$alpha,$beta,$prod,$energy,$purity,$recovery")
                if i % 10 == 0
                    flush(io2)
                    println("Saved $i samples")
                end

            catch e
                println("Failed sample $i")
            end
        end

    end

    # -------------------------------


end

# ==========================================================
# Run dataset generation
# ==========================================================

generate_dataset(16; n_samples=2000, N=10)

println("Dataset generation completed.")