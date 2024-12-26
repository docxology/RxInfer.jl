"""
    meta_analysis_simulation.jl

Module for running batches of simulations with different physics parameters.
"""

module MetaAnalysisSimulation

using RxInfer
using DataFrames
using Statistics
using ProgressMeter: Progress, next!

# Import MountainCar module
using ..MountainCar

export SimulationBatch, run_single_simulation, run_simulation_batch

"""
    SimulationBatch

Structure containing parameters for a batch of simulations.
"""
struct SimulationBatch
    force_range::AbstractRange
    friction_range::AbstractRange
    n_episodes::Int
    max_steps::Int
    planning_horizon::Int
    initial_state::Dict{String, Float64}
    target_state::Dict{String, Float64}
end

# Constructor with keyword arguments
function SimulationBatch(;
    force_range::AbstractRange,
    friction_range::AbstractRange,
    n_episodes::Int,
    max_steps::Int,
    planning_horizon::Int,
    initial_state::Dict{String, Any},
    target_state::Dict{String, Any}
)
    # Convert state dictionaries to Float64
    initial_state_float = Dict{String, Float64}(k => Float64(v) for (k, v) in initial_state)
    target_state_float = Dict{String, Float64}(k => Float64(v) for (k, v) in target_state)
    
    return SimulationBatch(
        force_range,
        friction_range,
        n_episodes,
        max_steps,
        planning_horizon,
        initial_state_float,
        target_state_float
    )
end

"""
    run_single_simulation(force::Float64, friction::Float64, agent_type::String, batch::SimulationBatch)

Run a single simulation with given parameters and return metrics.
"""
function run_single_simulation(force::Float64, friction::Float64, agent_type::String, batch::SimulationBatch)
    # Create physics environment with specified parameters
    Fa, Ff, Fg, height = MountainCar.create_physics(
        engine_force_limit=force,
        friction_coefficient=friction
    )
    
    # Create world
    execute, observe = MountainCar.create_world(
        Fg=Fg, Ff=Ff, Fa=Fa,
        initial_position=batch.initial_state["position"],
        initial_velocity=batch.initial_state["velocity"]
    )
    
    # Create agent
    if agent_type == "naive"
        # Naive agent always pushes right
        compute = (upsilon_t::Float64, y_hat_t::Vector{Float64}) -> nothing
        act = () -> 1.0
        slide = () -> nothing
        future = () -> zeros(batch.planning_horizon)
        agent = (compute, act, slide, future)
    else
        # Active inference agent
        agent = MountainCar.create_agent(
            T=batch.planning_horizon,
            Fg=Fg, Fa=Fa, Ff=Ff,
            engine_force_limit=force,
            x_target=[batch.target_state["position"], batch.target_state["velocity"]],
            initial_position=batch.initial_state["position"],
            initial_velocity=batch.initial_state["velocity"]
        )
    end
    
    # Run simulation
    positions = Float64[]
    velocities = Float64[]
    actions = Float64[]
    energies = Float64[]
    
    # Initial observation
    state = observe()
    push!(positions, state[1])
    push!(velocities, state[2])
    
    compute, act, slide, future = agent
    success = false
    target_time = batch.max_steps
    
    for t in 1:batch.max_steps
        # Get action and execute
        action = act()
        push!(actions, action)
        execute(action)
        
        # Observe and update
        state = observe()
        push!(positions, state[1])
        push!(velocities, state[2])
        
        # Calculate energy
        ke, pe, te = MountainCar.calculate_energy([state[1]], [state[2]], height)
        push!(energies, te[1])
        
        # Check if target reached
        if abs(state[1] - batch.target_state["position"]) < 0.01 && 
           abs(state[2] - batch.target_state["velocity"]) < 0.05
            success = true
            target_time = t
            break
        end
        
        # Update agent's beliefs
        compute(action, state)
        slide()
    end
    
    # Calculate metrics
    metrics = (
        success=success,
        target_time=target_time,
        total_energy=mean(energies),
        efficiency=success ? target_time / mean(energies) : 0.0,
        stability=std(velocities),
        oscillations=count(diff(sign.(velocities)) .!= 0),
        control_effort=sum(abs.(actions)),
        max_position=maximum(positions),
        avg_velocity=mean(abs.(velocities))
    )
    
    # Return results with physics parameters
    return (
        force=force,
        friction=friction,
        agent_type=agent_type,
        success=metrics.success,
        target_time=metrics.target_time,
        total_energy=metrics.total_energy,
        efficiency=metrics.efficiency,
        stability=metrics.stability,
        oscillations=metrics.oscillations,
        control_effort=metrics.control_effort,
        max_position=metrics.max_position,
        avg_velocity=metrics.avg_velocity
    )
end

"""
    run_simulation_batch(batch::SimulationBatch)

Run a batch of simulations with different parameter combinations and return results as DataFrame.
"""
function run_simulation_batch(batch::SimulationBatch)
    # Calculate total number of simulations
    n_force = length(batch.force_range)
    n_friction = length(batch.friction_range)
    n_agents = 2  # naive and active
    total_sims = n_force * n_friction * n_agents * batch.n_episodes
    
    # Initialize progress meter
    prog = Progress(total_sims, desc="Running simulations...")
    
    # Initialize results array
    results = []
    
    # Run simulations for each parameter combination
    for force in batch.force_range
        for friction in batch.friction_range
            for agent_type in ["naive", "active"]
                for _ in 1:batch.n_episodes
                    # Run single simulation
                    metrics = run_single_simulation(force, friction, agent_type, batch)
                    push!(results, metrics)
                    
                    # Update progress
                    next!(prog)
                end
            end
        end
    end
    
    # Convert results to DataFrame
    return DataFrame(results)
end

end # module MetaAnalysisSimulation 