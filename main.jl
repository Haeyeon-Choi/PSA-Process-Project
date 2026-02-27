using Pkg
Pkg.activate(".")

using PSASimulator
using DataFrames
using CSV
using QuasiMonteCarlo
using Random

# ----------------------------------------------------------
# Load demo data: to get simulation/isotherm parameters
# ----------------------------------------------------------
include(joinpath(dirname(dirname(pathof(PSASimulator))),
                 "demo", "demo_data.jl"))

# ==========================================================
# Dataset generation function
# ==========================================================

function generate_dataset(mat_index; n_samples=50, N=5)

    # -------------------------------
    # Material
    # -------------------------------
    material_property = vec(SIMULATION_PARAMETERS[mat_index, :])
    isoPar = vec(ISOTHERM_PARAMETERS[mat_index, :])
    material = (material_property, isoPar)

    # -------------------------------
    # Bounds (narrow, feasible-friendly)
    # --> 🚨 need to modify this part later on
    # -------------------------------
    lb = [2.0e5, 0.5, 100.0, 0.15, 0.10]
    ub = [5.0e5, 1.8, 600.0, 0.40, 0.35]

    Random.seed!(1234)

    # LHS sampling (dim × n_samples)
    X = QuasiMonteCarlo.sample(n_samples, lb, ub, LatinHypercubeSample())

    # -------------------------------
    # Fixed variables
    # --> may need to modify this part
    # -------------------------------
    L  = 1.0
    PI = 1e4
    # Pl = 1e4

    df = DataFrame(
        P0 = Float64[],
        ndot = Float64[],
        tads = Float64[],
        alpha = Float64[],
        beta = Float64[],
        productivity = Float64[],
        purity = Float64[],
        recovery = Float64[]
    )

    # -------------------------------
    # Simulation Loop
    # -------------------------------
    for i in 1:n_samples

        sample_vec = X[:, i]

        P0    = sample_vec[1]
        ndot  = sample_vec[2]
        tads  = sample_vec[3]
        alpha = sample_vec[4]
        beta  = sample_vec[5]

        vars = [L, P0, ndot, tads, alpha, beta, PI, Pl]

        productivity = NaN
        purity = NaN
        recovery = NaN

        try
            # EconomicEvaluation - productivity
            res = PSASimulator.psacycle(
                vars,
                material;
                N=N,
                it_disp=false,
                run_type=:EconomicEvaluation
            )

            productivity    = -res.objectives[1]
            energy              =

            # ProcessEvaluation - purity & recovery
            res_proc = PSASimulator.psacycle(
                vars,
                material;
                N=N,
                it_disp=false,
                run_type=:ProcessEvaluation
            )

            purity = -res_proc.objectives[1]
            recovery = -res_proc.objectives[2]

            println("✓ Sample $i / $n_samples::") 
            println("prod:$productivity, pur:$purity, rec:$recovery")

        catch e
            println("✗ Failed sample $i")
        end

        push!(df, (P0, ndot, tads, alpha, beta,
                   productivity, purity, recovery))
    end

    return df
end

# ==========================================================
# Run generation
# ==========================================================

df = generate_dataset(16; n_samples=1, N=5)

CSV.write("data/dataset_material16.csv", df)

println("Dataset generation completed.")