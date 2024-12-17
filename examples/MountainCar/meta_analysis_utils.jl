module MetaAnalysisUtils

using Statistics
using LinearAlgebra
using RxInfer
using RxInfer: getmodel, getreturnval, getvarref, getvariable
using RxInfer.ReactiveMP: getrecent, messageout
using HypergeometricFunctions

# Import MountainCar module
using ..MountainCar

export SimulationMetrics, run_simulation, calculate_metrics

"""
    SimulationMetrics

Structure to hold metrics from a single simulation run.
"""
struct SimulationMetrics
    max_position::Float64
    avg_velocity::Float64
    success::Bool
    target_time::Float64
    total_energy::Float64
    oscillations::Float64
    control_effort::Float64
    stability::Float64
    efficiency::Float64
    trajectory::Dict{String,Vector{Float64}}
end

"""
    calculate_oscillations(positions)

Calculate the number of oscillations in a position trajectory.
"""
function calculate_oscillations(positions)
    # Count direction changes
    direction_changes = 0
    prev_direction = sign(positions[2] - positions[1])
    
    for i in 2:(length(positions)-1)
        curr_direction = sign(positions[i+1] - positions[i])
        if curr_direction != prev_direction && curr_direction != 0
            direction_changes += 1
            prev_direction = curr_direction
        end
    end
    
    return direction_changes / 2  # Each full oscillation is two direction changes
end

"""
    calculate_control_effort(actions)

Calculate the total control effort from a sequence of actions.
"""
function calculate_control_effort(actions)
    return sum(abs.(actions))
end

"""
    calculate_stability(positions, velocities)

Calculate trajectory stability based on position and velocity variance.
"""
function calculate_stability(positions, velocities)
    pos_var = var(positions)
    vel_var = var(velocities)
    return 1.0 / (1.0 + sqrt(pos_var + vel_var))
end

"""
    calculate_efficiency(success, target_time, total_energy)

Calculate overall efficiency based on success, time to target, and energy use.
"""
function calculate_efficiency(success, target_time, total_energy)
    if !success
        return 0.0
    end
    return 1.0 / (target_time * total_energy)
end

"""
    run_simulation(force, friction, config)

Run a single simulation with given parameters and return metrics.
"""
function run_simulation(force, friction, config)
    # Extract simulation parameters from config
    timesteps = config["simulation"]["timesteps"]
    planning_horizon = config["simulation"]["planning_horizon"]
    initial_position = config["initial_state"]["position"]
    initial_velocity = config["initial_state"]["velocity"]
    target_position = config["target_state"]["position"]
    target_velocity = config["target_state"]["velocity"]
    
    # Create physics environment
    Fa, Ff, Fg, height = MountainCar.create_physics(
        engine_force_limit=force,
        friction_coefficient=friction
    )
    
    # Create world and agent
    (execute, observe) = MountainCar.create_world(
        Fg=Fg, Ff=Ff, Fa=Fa,
        initial_position=initial_position,
        initial_velocity=initial_velocity
    )
    
    x_target = [target_position, target_velocity]
    (compute, act, slide, future) = MountainCar.create_agent(
        T=planning_horizon,
        Fa=Fa, Fg=Fg, Ff=Ff,
        engine_force_limit=force,
        x_target=x_target,
        initial_position=initial_position,
        initial_velocity=initial_velocity
    )
    
    # Run simulation
    positions = Float64[]
    velocities = Float64[]
    actions = Float64[]
    predictions = Vector{Float64}[]
    target_time = Inf
    
    for t = 1:timesteps
        state = observe()
        push!(positions, state[1])
        push!(velocities, state[2])
        
        compute(0.0, state)
        action = clamp(act(), -1.0, 1.0)
        push!(actions, action)
        execute(action)
        slide()
        
        push!(predictions, future())
        
        # Check if target reached
        if abs(state[1] - target_position) < 0.01 && 
           abs(state[2] - target_velocity) < 0.05 && 
           target_time == Inf
            target_time = t
        end
    end
    
    # Calculate metrics
    ke, pe, te = MountainCar.calculate_energy(positions, velocities, height)
    success = !isinf(target_time)
    oscillations = calculate_oscillations(positions)
    control_effort = calculate_control_effort(actions)
    stability = calculate_stability(positions, velocities)
    efficiency = calculate_efficiency(success, target_time, mean(te))
    
    # Create trajectory dictionary
    trajectory = Dict{String,Vector{Float64}}(
        "positions" => positions,
        "velocities" => velocities,
        "actions" => actions,
        "kinetic_energy" => ke,
        "potential_energy" => pe,
        "total_energy" => te
    )
    
    return SimulationMetrics(
        maximum(positions),
        mean(velocities),
        success,
        target_time,
        mean(te),
        oscillations,
        control_effort,
        stability,
        efficiency,
        trajectory
    )
end

end # module 