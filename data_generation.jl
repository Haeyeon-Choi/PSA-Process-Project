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
# Dataset generator
# ==========================================================

function generate_dataset(mat_index; n_samples=2000, N=10)

    filepath = joinpath(@__DIR__, "data", "dataset_material$(mat_index).csv")
    mkpath(dirname(filepath))

    # Create file with header
    open(filepath, "w") do io
        println(io, "P0,ndot,tads,alpha,beta,PI,Pl,productivity,energy,purity,recovery")
    end

    # -------------------------------
    # Material setup
    # -------------------------------
    material_property = vec(SIMULATION_PARAMETERS[mat_index, :])
    isoPar = vec(ISOTHERM_PARAMETERS[mat_index, :])
    material = (material_property, isoPar)

    # -------------------------------
    # Sampling bounds
    # References:
    #   P0  : Haghpanah et al. (2013), Ind. Eng. Chem. Res. — 1.2–3.66 atm for post-combustion CO2 PSA
    #   Pl  : Zhang et al. (2008), Adsorption (Springer)   — optimal Pl = 0.025–0.1 bar for zeolite 13X VSA
    #   PI  : Kumar (1994), Chem. Eng. Sci.; Nilchan & Pantelides (1998), Adsorption
    #         — intermediate pressure ~0.15–0.5 bar; constrained to [Pl, 0.5*P0]
    #   ndot: pilot-scale superficial velocity 0.3–1.5 m/s (general PSA literature)
    #   tads: Haghpanah et al. (2013) — optimal tads in 60–200 s range; extended to 400 s
    #   alpha, beta: general PSA cycle optimization literature — 0.10–0.40
    # -------------------------------

    # Independent variable bounds (7 dimensions: P0, ndot, tads, alpha, beta, Pl, PI_frac)
    # PI_frac is sampled in [0,1] and mapped to [Pl, min(2e5, 0.5*P0)] after P0 and Pl are determined
    lb_indep = [1.0e5, 0.5,  60.0, 0.10, 0.10, 2.0e3]
    ub_indep = [4.0e5, 2.0, 400.0, 0.40, 0.40, 1.0e4]

    Random.seed!(1234)

    # Sample 7 dimensions in [0, 1] with Sobol sequence
    # Dims: [P0, ndot, tads, alpha, beta, Pl, PI_frac]
    X_unit = QuasiMonteCarlo.sample(n_samples, 7, SobolSample())

    # Transform independent variables to physical bounds
    X = similar(X_unit)
    for j in 1:6
        X[j, :] = lb_indep[j] .+ X_unit[j, :] .* (ub_indep[j] - lb_indep[j])
    end
    X[7, :] = X_unit[7, :]  # PI_frac stays in [0, 1] for now

    # Sort by P0 for sequential evaluation (cache-friendly)
    perm = sortperm(X[1, :])
    X = X[:, perm]

    # Fixed parameters
    L = 1.0  # column length [m], fixed at pilot-scale value

    # ==========================================================
    # Simulation loop
    # ==========================================================
    open(filepath, "a") do io
        for i in 1:n_samples

            P0    = X[1, i]
            ndot  = X[2, i]
            tads  = X[3, i]
            alpha = X[4, i]
            beta  = X[5, i]
            Pl    = X[6, i]

            # PI is sampled hierarchically: Pl <= PI <= min(2e5, 0.5*P0)
            # This enforces the physical pressure ordering constraint Pl <= PI <= P0
            PI_frac = X[7, i]
            PI_ub   = min(2.0e5, 0.5 * P0)
            PI      = Pl + PI_frac * (PI_ub - Pl)

            vars = [L, P0, ndot, tads, alpha, beta, PI, Pl]

            try
                res = PSASimulator.psacycle(
                    vars,
                    material;
                    N=N,
                    it_disp=false,
                    run_type=:EconomicEvaluation
                )

                prod     = -res.objectives[1]
                energy   =  res.objectives[2]
                purity   =  res.traj[:purity]
                recovery =  res.traj[:recovery]

                println(io, "$P0,$ndot,$tads,$alpha,$beta,$PI,$Pl,$prod,$energy,$purity,$recovery")

                if i % 10 == 0
                    flush(io)
                    println("Saved $i / $n_samples samples")
                end

            catch e
                println("Failed sample $i: $e")
            end
        end
    end

    println("Dataset saved to $filepath")
end

# ==========================================================
# Run dataset generation
# ==========================================================

generate_dataset(16; n_samples=2000, N=10)

println("Dataset generation completed.")
