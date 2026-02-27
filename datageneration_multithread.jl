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

function generate_dataset_fast(mat_index; n_samples=200, N=4)

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
    lb = [2.0e5, 0.5, 100.0, 0.15, 0.10]
    ub = [5.0e5, 1.8, 600.0, 0.40, 0.35]

    Random.seed!(1234)
    X = QuasiMonteCarlo.sample(n_samples, lb, ub, LatinHypercubeSample())

    # -------------------------------
    # Fixed parameters
    # -------------------------------
    L  = 1.0
    PI = 1e4
    Pl = 1e4

    # -------------------------------
    # Preallocate arrays
    # -------------------------------
    P0_vec       = Vector{Float64}(undef, n_samples)
    ndot_vec     = similar(P0_vec)
    tads_vec     = similar(P0_vec)
    alpha_vec    = similar(P0_vec)
    beta_vec     = similar(P0_vec)

    prod_vec     = fill(NaN, n_samples)
    energy_vec   = fill(NaN, n_samples)
    purity_vec   = fill(NaN, n_samples)
    recovery_vec = fill(NaN, n_samples)

    # ==========================================================
    # Multithreaded simulation loop
    # ==========================================================
    @threads for i in 1:n_samples

        sample_vec = X[:, i]

        P0    = sample_vec[1]
        ndot  = sample_vec[2]
        tads  = sample_vec[3]
        alpha = sample_vec[4]
        beta  = sample_vec[5]

        vars = [L, P0, ndot, tads, alpha, beta, PI, Pl]

        # Store inputs
        P0_vec[i]    = P0
        ndot_vec[i]  = ndot
        tads_vec[i]  = tads
        alpha_vec[i] = alpha
        beta_vec[i]  = beta

        try
            # -------------------------------------------------
            # EconomicEvaluation
            # -------------------------------------------------
            res = PSASimulator.psacycle(
                vars,
                material;
                N=N,
                it_disp=false,
                run_type=:EconomicEvaluation
            )

            # According to repo structure:
            # objectives typically contain productivity & energy
            prod_vec[i]   = -res.objectives[1]
            energy_vec[i] =  res.objectives[2]

            # we can get purity and recovery from the trajectory data
            purity_vec[i]   = res.traj[:purity]
            recovery_vec[i] = res.traj[:recovery]

            println("Complete!!")
            println("prod:", prod_vec[i], "purity:", purity_vec[i], "recovery:", recovery_vec[i])

        catch e
            # 실패한 샘플은 NaN 유지
        end
    end

    # -------------------------------
    # Build DataFrame
    # -------------------------------
    df = DataFrame(
        P0 = P0_vec,
        ndot = ndot_vec,
        tads = tads_vec,
        alpha = alpha_vec,
        beta = beta_vec,
        productivity = prod_vec,
        energy = energy_vec,
        purity = purity_vec,
        recovery = recovery_vec
    )

    return df
end

# ==========================================================
# Run dataset generation
# ==========================================================

df = generate_dataset_fast(16; n_samples=10, N=4)

CSV.write("data/dataset_material16_multithread.csv", df)

println("Dataset generation completed.")