using Pkg
Pkg.activate(".")

using PSASimulator

include(joinpath(dirname(dirname(pathof(PSASimulator))), "demo", "demo_data.jl"))

function parse_args(args)
    if length(args) != 11
        error("Usage: julia scripts/evaluate_psa_point.jl <mat_index> <N> <L> <P0> <ndot> <tads> <alpha> <beta> <PI> <Pl> <it_disp>")
    end

    mat_index = parse(Int, args[1])
    N = parse(Int, args[2])
    L = parse(Float64, args[3])
    P0 = parse(Float64, args[4])
    ndot = parse(Float64, args[5])
    tads = parse(Float64, args[6])
    alpha = parse(Float64, args[7])
    beta = parse(Float64, args[8])
    PI = parse(Float64, args[9])
    Pl = parse(Float64, args[10])
    it_disp = lowercase(args[11]) == "true"

    return mat_index, N, [L, P0, ndot, tads, alpha, beta, PI, Pl], it_disp
end

function main()
    mat_index, N, vars, it_disp = parse_args(ARGS)

    material_property = vec(SIMULATION_PARAMETERS[mat_index, :])
    isoPar = vec(ISOTHERM_PARAMETERS[mat_index, :])
    material = (material_property, isoPar)

    res = PSASimulator.psacycle(vars, material; N=N, it_disp=it_disp, run_type=:EconomicEvaluation)

    productivity = -res.objectives[1]
    energy = res.objectives[2]
    purity = res.traj[:purity]
    recovery = res.traj[:recovery]

    println("$(productivity),$(energy),$(purity),$(recovery)")
end

main()
