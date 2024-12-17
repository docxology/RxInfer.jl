module MetaAnalysisSimulation

using Distributed
using SharedArrays
using Statistics
using RxInfer
using RxInfer: getmodel, getreturnval, getvarref, getvariable
using RxInfer.ReactiveMP: getrecent, messageout
using HypergeometricFunctions
using LinearAlgebra

# Import required modules
using ..MountainCar
using ..MetaAnalysisUtils

export run_simulation_batch, SimulationBatch

"""
    SimulationBatch

Structure to hold a batch of simulation parameters and results.
"""
struct SimulationBatch
    force_values::Vector{Float64}
    friction_values::Vector{Float64}
    results::Vector{Dict{String,Any}}
end

"""
    run_simulation_batch(force_range, friction_range, config)

Run a batch of simulations with different parameter combinations.
"""
function run_simulation_batch(force_range, friction_range, config)
    n_force = length(force_range)
    n_friction = length(friction_range)
    results = Vector{Dict{String,Any}}(undef, n_force * n_friction)
    
    @sync @distributed for idx in 1:length(results)
        i = div(idx-1, n_friction) + 1
        j = mod1(idx, n_friction)
        
        force = force_range[i]
        friction = friction_range[j]
        
        # Run simulation with current parameters
        metrics = MetaAnalysisUtils.run_simulation(force, friction, config)
        
        # Store results
        results[idx] = Dict{String,Any}(
            "force" => force,
            "friction" => friction,
            "metrics" => metrics
        )
    end
    
    return SimulationBatch(collect(force_range), collect(friction_range), results)
end

end # module 