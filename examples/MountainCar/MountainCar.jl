# Ensure we're in the right environment
import Pkg
Pkg.activate(@__DIR__)

module MountainCar

using RxInfer
using ReactiveMP
using ReactiveMP: getrecent, messageout
using RxInfer: getmodel, getreturnval, getvarref, getvariable
using HypergeometricFunctions
using LinearAlgebra
using Statistics
using Plots
using Plots: @animate, gif, plot, scatter!, annotate!, text

export create_physics, create_world, create_agent, calculate_energy, create_and_save_animation

# Define the model at the top level
@model function mountain_car_model(m_u, V_u, m_x, V_x, m_s_t_min, V_s_t_min, T, Fg, Fa, Ff, engine_force_limit)
    # Hidden states
    x = randomvar(T+1)  # Position
    v = randomvar(T+1)  # Velocity
    a = randomvar(T)    # Action
    
    # Initial state
    x[1] ~ Normal(m_s_t_min[1], 0.01)
    v[1] ~ Normal(m_s_t_min[2], 0.01)
    
    # State transitions
    for t in 1:T
        # Total force
        F = Fa(a[t]) + Ff(v[t]) + Fg(x[t])
        
        # Next velocity
        v[t+1] ~ Normal(v[t] + 0.001 * F, 0.01)
        
        # Next position
        x[t+1] ~ Normal(x[t] + 0.001 * v[t], 0.01)
    end
    
    # Prior preferences
    for t in 1:T
        # Position preference
        x[t] ~ Normal(m_x[t][1], 1/sqrt(γ_x))
        
        # Velocity preference
        v[t] ~ Normal(m_x[t][2], 1/sqrt(γ_v))
        
        # Action preference
        a[t] ~ Normal(m_u[t][1], 1/sqrt(γ_a))
    end
end

"""
    create_physics(; engine_force_limit=0.04, friction_coefficient=0.1)

Create physics functions for the Mountain Car environment.
"""
function create_physics(; engine_force_limit=0.04, friction_coefficient=0.1)
    # Mountain height function and its derivative
    height(x) = sin(3x)
    dheight(x) = 3cos(3x)
    
    # Force functions
    Fa(a) = engine_force_limit * a  # Engine force
    Ff(v) = -friction_coefficient * v  # Friction force
    Fg(x) = -9.81 * dheight(x)  # Gravitational force
    
    return Fa, Ff, Fg, height
end

"""
    create_world(; Fg, Ff, Fa, initial_position=-0.5, initial_velocity=0.0)

Create a simulation world with the given physics and initial conditions.
"""
function create_world(; Fg, Ff, Fa, initial_position=-0.5, initial_velocity=0.0)
    # State variables
    x = initial_position  # Position
    v = initial_velocity  # Velocity
    
    # Execute action
    function execute(a)
        # Update velocity
        F = Fa(a) + Ff(v) + Fg(x)  # Total force
        v += 0.001 * F  # Euler integration
        
        # Update position
        x += 0.001 * v  # Euler integration
        
        # Enforce position bounds
        if x < -1.2
            x = -1.2
            v = 0.0
        elseif x > 0.6
            x = 0.6
            v = 0.0
        end
    end
    
    # Observe state
    observe() = [x, v]
    
    return execute, observe
end

"""
    create_agent(; T, Fa, Fg, Ff, engine_force_limit, x_target, initial_position, initial_velocity)

Create an active inference agent for the Mountain Car problem.
"""
function create_agent(; T, Fa, Fg, Ff, engine_force_limit, x_target, initial_position, initial_velocity)
    # Prior preferences
    μ_x = x_target[1]  # Target position
    μ_v = x_target[2]  # Target velocity
    
    # Precision parameters
    γ_x = 100.0  # Position precision
    γ_v = 10.0   # Velocity precision
    γ_a = 1.0    # Action precision
    
    # Initialize model parameters
    m_u = [[0.0] for _ in 1:T]  # Action means
    V_u = [1.0 for _ in 1:T]    # Action variances
    m_x = [[μ_x, μ_v] for _ in 1:T]  # State means
    V_x = [1.0 for _ in 1:T]    # State variances
    m_s_t_min = [initial_position, initial_velocity]  # Initial state mean
    V_s_t_min = 0.01  # Initial state variance
    
    # Create model with keyword arguments
    model = mountain_car_model(
        m_u = m_u,
        V_u = V_u,
        m_x = m_x,
        V_x = V_x,
        m_s_t_min = m_s_t_min,
        V_s_t_min = V_s_t_min,
        T = T,
        Fg = Fg,
        Fa = Fa,
        Ff = Ff,
        engine_force_limit = engine_force_limit
    )
    
    # Initialize inference
    x = fill(initial_position, T+1)
    v = fill(initial_velocity, T+1)
    a = zeros(T)
    
    # Create message passing program
    program = ReactiveMP.messagepassingalgorithm(model)
    
    # Initialize marginals
    marginals = initialize!(program)
    
    # Compute messages
    function compute(action, state)
        # Update current state
        x[1] = state[1]
        v[1] = state[2]
        
        # Update action
        if !isnothing(action)
            a[1] = action
        end
        
        # Update marginals
        update!(program, marginals, (x=x, v=v, a=a))
    end
    
    # Get action
    function act()
        μ_a = mean(marginals[:a][1])
        return clamp(μ_a, -1.0, 1.0)
    end
    
    # Slide window
    function slide()
        x[1:end-1] = x[2:end]
        v[1:end-1] = v[2:end]
        a[1:end-1] = a[2:end]
        a[end] = 0.0
    end
    
    # Get future predictions
    future() = x[2:end]
    
    return compute, act, slide, future
end

"""
    calculate_energy(positions, velocities, height)

Calculate kinetic, potential, and total energy for a trajectory.
"""
function calculate_energy(positions, velocities, height)
    # Calculate energies
    ke = 0.5 .* velocities.^2  # Kinetic energy
    pe = 9.81 .* height.(positions)  # Potential energy
    te = ke .+ pe  # Total energy
    
    return ke, pe, te
end

"""
    create_and_save_animation(positions, velocities, actions, predictions, height, output_path)

Create and save an animation of the mountain car trajectory.
"""
function create_and_save_animation(positions, velocities, actions, predictions, height, output_path)
    println("\n📊 Creating animation...")
    
    anim = Plots.@animate for t in 1:length(positions)
        # Plot mountain
        x_range = range(-1.2, 0.6, length=100)
        Plots.plot(x_range, height.(x_range), 
            label="Mountain",
            color=:black,
            linewidth=2,
            xlabel="Position",
            ylabel="Height",
            title="Mountain Car (t=$t)",
            legend=:topright
        )

        # Plot car
        Plots.scatter!([positions[t]], [height(positions[t])],
            label="Car",
            color=:red,
            markersize=6
        )

        # Plot predictions if available
        if t <= length(predictions)
            pred_heights = height.(predictions[t])
            Plots.scatter!(predictions[t], pred_heights,
                label="Predictions",
                color=:blue,
                alpha=0.5,
                markersize=3
            )
        end

        # Add state information
        Plots.annotate!(
            -1.1,
            1.1,
            Plots.text(
                "Position: $(round(positions[t], digits=3))\nVelocity: $(round(velocities[t], digits=3))\nAction: $(round(actions[t], digits=3))",
                :left,
                8
            )
        )
    end

    println("💾 Saving animation...")
    Plots.gif(anim, output_path, fps=30)
end

end # module MountainCar

# Main execution block
if abspath(PROGRAM_FILE) == @__FILE__
    # Import required packages
    using .MountainCar
    using Statistics

    # Create Outputs directory if it doesn't exist
    mkpath(joinpath(@__DIR__, "Outputs"))

    # Simulation parameters
    timesteps = 200
    planning_horizon = 30
    initial_position = -0.5
    initial_velocity = 0.0
    target_position = 0.5
    target_velocity = 0.0

    # Create physics environment
    Fa, Ff, Fg, height = MountainCar.create_physics(
        engine_force_limit=0.04,
        friction_coefficient=0.1
    )

    # Create world
    execute, observe = MountainCar.create_world(
        Fg=Fg, Ff=Ff, Fa=Fa,
        initial_position=initial_position,
        initial_velocity=initial_velocity
    )

    # Create agent
    compute, act, slide, future = MountainCar.create_agent(
        T=planning_horizon,
        Fa=Fa, Fg=Fg, Ff=Ff,
        engine_force_limit=0.04,
        x_target=[target_position, target_velocity],
        initial_position=initial_position,
        initial_velocity=initial_velocity
    )

    # Run simulation
    positions = Float64[]
    velocities = Float64[]
    actions = Float64[]
    predictions = Vector{Float64}[]

    println("🚗 Starting Mountain Car simulation...")
    println("Initial state: position = $initial_position, velocity = $initial_velocity")
    println("Target state: position = $target_position, velocity = $target_velocity")

    for t in 1:timesteps
        # Get current state
        state = observe()
        push!(positions, state[1])
        push!(velocities, state[2])

        # Compute action
        compute(nothing, state)
        action = act()
        push!(actions, action)

        # Execute action
        execute(action)
        slide()

        # Store predictions
        push!(predictions, future())

        # Print progress
        if t % 20 == 0
            println("Step $t: position = $(state[1]), velocity = $(state[2])")
        end
    end

    # Calculate energies
    ke, pe, te = MountainCar.calculate_energy(positions, velocities, height)

    # Create and save animation
    MountainCar.create_and_save_animation(
        positions, 
        velocities, 
        actions, 
        predictions, 
        height,
        joinpath(@__DIR__, "Outputs", "mountain_car.gif")
    )

    # Print final statistics
    println("\n📊 Simulation Statistics:")
    println("Final position: $(positions[end])")
    println("Final velocity: $(velocities[end])")
    println("Average kinetic energy: $(mean(ke))")
    println("Average potential energy: $(mean(pe))")
    println("Average total energy: $(mean(te))")
    println("\n✅ Simulation complete! Check the Outputs directory for the animation.")
end