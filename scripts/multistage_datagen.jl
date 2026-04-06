using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using PSASimulator, QuasiMonteCarlo, Random

include(joinpath(dirname(dirname(pathof(PSASimulator))), "demo", "demo_data.jl"))

mat_index = parse(Int, ARGS[1])
n_stages = parse(Int, ARGS[2])
n_samples = 2000
N = 10

println("Starting multi-stage ($n_stages) data generation for material $mat_index")

filepath = joinpath(@__DIR__, "..", "data", "dataset_material$(mat_index)_$(n_stages)stage.csv")

material_property = vec(SIMULATION_PARAMETERS[mat_index, :])
isoPar = vec(ISOTHERM_PARAMETERS[mat_index, :])
material = (material_property, isoPar)

lb_indep = [1.0e5, 0.5,  60.0, 0.10, 0.10, 2.0e3]
ub_indep = [4.0e5, 2.0, 400.0, 0.40, 0.40, 1.0e4]

Random.seed!(1234)
X_unit = QuasiMonteCarlo.sample(n_samples, 7, SobolSample())
X = similar(X_unit)
for j in 1:6
    X[j, :] = lb_indep[j] .+ X_unit[j, :] .* (ub_indep[j] - lb_indep[j])
end
X[7, :] = X_unit[7, :]
perm = sortperm(X[1, :]); X = X[:, perm]; L = 1.0

open(filepath, "w") do io
    # Header: inputs + per-stage outputs + final outputs
    header = "P0,ndot,tads,alpha,beta,PI,Pl"
    for s in 1:n_stages
        header *= ",purity_s$(s),recovery_s$(s),productivity_s$(s),energy_s$(s)"
    end
    header *= ",final_purity,final_recovery,overall_recovery"
    println(io, header)

    for i in 1:n_samples
        P0=X[1,i]; ndot=X[2,i]; tads=X[3,i]; alpha=X[4,i]; beta=X[5,i]; Pl=X[6,i]
        PI_frac=X[7,i]; PI_ub=min(2.0e5, 0.5*P0); PI=Pl+PI_frac*(PI_ub-Pl)
        vars=[L, P0, ndot, tads, alpha, beta, PI, Pl]

        y0_feed = 0.15
        overall_rec = 1.0
        stage_results = []
        failed = false

        for s in 1:n_stages
            try
                res = PSASimulator.psacycle(vars, material; N=N, it_disp=false, run_type=:EconomicEvaluation, y0=y0_feed)
                prod = -res.objectives[1]
                energy = res.objectives[2]
                purity = res.traj[:purity]
                recovery = res.traj[:recovery]
                push!(stage_results, (purity, recovery, prod, energy))
                overall_rec *= recovery
                y0_feed = purity
            catch e
                failed = true
                println("Failed sample $i stage $s: $e")
                break
            end
        end

        if !failed && length(stage_results) == n_stages
            line = "$P0,$ndot,$tads,$alpha,$beta,$PI,$Pl"
            for (pur, rec, prod, en) in stage_results
                line *= ",$pur,$rec,$prod,$en"
            end
            final_pur = stage_results[end][1]
            final_rec = stage_results[end][2]
            line *= ",$final_pur,$final_rec,$overall_rec"
            println(io, line)

            if i % 10 == 0
                flush(io)
                println("Saved $i / $n_samples samples (final purity=$(round(final_pur, digits=4)))")
            end
        end
    end
end
println("Dataset generation completed for material $mat_index ($n_stages stages).")
