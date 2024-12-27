"""
    meta_analysis_simulation.jl

Module for running batches of Mountain Car simulations.
"""
module MetaAnalysisSimulation

using RxInfer
using DataFrames
using Statistics
using Printf
using Dates

# Import MountainCar module
using Main.MountainCar

export SimulationBatch, run_simulation_batch, run_single_simulation

"""
    SimulationBatch

Structure containing parameters for a batch of simulations.
"""
struct SimulationBatch
    force_range::AbstractRange{Float64}
    friction_range::AbstractRange{Float64}
    n_episodes::Int
    max_steps::Int
    planning_horizon::Int
    initial_state::Dict{String,Float64}
    target_state::Dict{String,Float64}
end

"""
    SimulationBatch(; kwargs...)

Constructor for SimulationBatch with keyword arguments.
Handles type conversions for the state dictionaries.
"""
function SimulationBatch(;
    force_range::AbstractRange,
    friction_range::AbstractRange,
    n_episodes::Integer,
    max_steps::Integer,
    planning_horizon::Integer,
    initial_state::Dict{String,Any},
    target_state::Dict{String,Any}
)
    # Convert ranges to Float64 if needed
    force_range_f64 = convert(AbstractRange{Float64}, force_range)
    friction_range_f64 = convert(AbstractRange{Float64}, friction_range)
    
    # Convert state dictionaries to Float64
    initial_state_f64 = Dict{String,Float64}(k => Float64(v) for (k, v) in initial_state)
    target_state_f64 = Dict{String,Float64}(k => Float64(v) for (k, v) in target_state)
    
    return SimulationBatch(
        force_range_f64,
        friction_range_f64,
        Int(n_episodes),
        Int(max_steps),
        Int(planning_horizon),
        initial_state_f64,
        target_state_f64
    )
end

"""
    run_single_simulation(force, friction, agent_type, batch)

Run a single simulation with given parameters.
"""
function run_single_simulation(force::Float64, friction::Float64, agent_type::String, batch::SimulationBatch)
    # Create physics environment
    Fa, Ff, Fg, height = MountainCar.create_physics(
        engine_force_limit=force,
        friction_coefficient=friction
    )

    # Initial and target states
    initial_position = batch.initial_state["position"]
    initial_velocity = batch.initial_state["velocity"]
    x_target = [batch.target_state["position"], batch.target_state["velocity"]]

    # Create world
    execute, observe = MountainCar.create_world(
        Fg=Fg, Ff=Ff, Fa=Fa,
        initial_position=initial_position,
        initial_velocity=initial_velocity
    )

    # Initialize metrics
    positions = Float64[]
    velocities = Float64[]
    actions = Float64[]
    energies = Float64[]
    target_time = batch.max_steps + 1

    # Run simulation based on agent type
    if agent_type == "naive"
        # Naive agent (always push right)
        for t in 1:batch.max_steps
            state = observe()
            push!(positions, state[1])
            push!(velocities, state[2])
            
            # Always push right with maximum force
            action = 1.0
            push!(actions, action)
            execute(action)
            
            # Calculate energy
            ke = 0.5 * state[2]^2
            pe = 9.81 * height(state[1])
            push!(energies, ke + pe)
            
            # Check if target reached
            if abs(state[1] - x_target[1]) < 0.01 && abs(state[2] - x_target[2]) < 0.05
                target_time = t
                break
            end
        end
    else
        # Active Inference agent
        compute, act, slide, future = MountainCar.create_agent(
            T=batch.planning_horizon,
            Fa=Fa, Fg=Fg, Ff=Ff,
            engine_force_limit=force,
            x_target=x_target,
            initial_position=initial_position,
            initial_velocity=initial_velocity
        )

        # Initial observation and computation
        current_state = observe()
        push!(positions, current_state[1])
        push!(velocities, current_state[2])
        compute(0.0, current_state)

        for t in 1:batch.max_steps
            # Get action and execute
            action = act()
            push!(actions, action)
            execute(action)

            # Observe and update
            current_state = observe()
            push!(positions, current_state[1])
            push!(velocities, current_state[2])
            
            # Calculate energy
            ke = 0.5 * current_state[2]^2
            pe = 9.81 * height(current_state[1])
            push!(energies, ke + pe)

            # Check if target reached
            if abs(current_state[1] - x_target[1]) < 0.01 && abs(current_state[2] - x_target[2]) < 0.05
                target_time = t
                break
            end

            # Update agent's beliefs
            compute(action, current_state)
            slide()
        end
    end

    # Calculate metrics
    success = target_time <= batch.max_steps
    completion_time = success ? target_time : batch.max_steps
    max_position = maximum(positions)
    avg_velocity = mean(velocities)
    total_energy = sum(energies)
    avg_energy = mean(energies)
    control_effort = sum(abs.(actions))

    # Return metrics as a named tuple
    return (
        force=force,
        friction=friction,
        agent_type=agent_type,
        success=success,
        completion_time=completion_time,
        max_position=max_position,
        avg_velocity=avg_velocity,
        total_energy=total_energy,
        avg_energy=avg_energy,
        control_effort=control_effort,
        final_position=positions[end],
        final_velocity=velocities[end],
        efficiency=success ? 1.0 / (completion_time * total_energy) : 0.0
    )
end

"""
    run_simulation_batch(batch)

Run a batch of simulations with different parameters.
"""
function run_simulation_batch(batch::SimulationBatch)
    results = []
    total_sims = length(batch.force_range) * length(batch.friction_range) * 2 * batch.n_episodes
    completed_sims = 0
    successful_sims = 0

    # Progress tracking
    print("Running simulations... 0%")
    flush(stdout)

    for force in batch.force_range
        for friction in batch.friction_range
            for agent_type in ["naive", "active_inference"]
                for episode in 1:batch.n_episodes
                    # Run simulation
                    result = run_single_simulation(force, friction, agent_type, batch)
                    push!(results, result)

                    # Update counters
                    completed_sims += 1
                    successful_sims += result.success ? 1 : 0

                    # Update progress
                    progress = round(Int, 100 * completed_sims / total_sims)
                    print("\rRunning simulations... $progress%")
                    flush(stdout)
                end
            end
        end
    end

    println("\nCompleted $completed_sims simulations with $(successful_sims) successes ($(round(100 * successful_sims / completed_sims, digits=2))% success rate)")

    return results
end

end # module 