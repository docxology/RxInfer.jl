# Ensure we're in the right environment
import Pkg
Pkg.activate(@__DIR__)

module MountainCar

using RxInfer
using RxInfer: getmodel, getreturnval, getvarref, getvariable
using RxInfer.ReactiveMP: getrecent, messageout
using HypergeometricFunctions: _₂F₁
using LinearAlgebra
using Statistics
using Plots
using Plots: @animate, gif, plot, scatter!, annotate!, text
using Dates

# Include visualization module
include("visualization.jl")

# Export all public functions
export create_physics, create_world, create_agent, calculate_energy, create_and_save_animation

# Helper functions
diageye(n::Int) = Matrix{Float64}(I, n, n)
const huge = 1e10
const tiny = 1e-10

# Define the model at the top level
@model function mountain_car_model(m_u, V_u, m_x, V_x, m_s_t_min, V_s_t_min, T, Fg, Fa, Ff, engine_force_limit)
    # Transition function modeling transition due to gravity and friction
    g = (s_t_min::AbstractVector) -> begin 
        s_t = similar(s_t_min) # Next state
        s_t[2] = s_t_min[2] + Fg(s_t_min[1]) + Ff(s_t_min[2]) # Update velocity
        s_t[1] = s_t_min[1] + s_t[2] # Update position
        return s_t
    end
    
    # Function for modeling engine control
    h = (u::AbstractVector) -> [0.0, Fa(u[1])] 
    
    # Inverse engine force, from change in state to corresponding engine force
    h_inv = (delta_s_dot::AbstractVector) -> [atanh(clamp(delta_s_dot[2], -engine_force_limit+1e-3, engine_force_limit-1e-3)/engine_force_limit)] 
    
    # Internal model parameters
    Gamma = 1e4*diageye(2) # Transition precision
    Theta = 1e-4*diageye(2) # Observation variance

    s_t_min ~ MvNormal(mean = m_s_t_min, cov = V_s_t_min)
    s_k_min = s_t_min

    local s
    
    for k in 1:T
        u[k] ~ MvNormal(mean = m_u[k], cov = V_u[k])
        u_h_k[k] ~ h(u[k]) where { meta = DeltaMeta(method = Linearization(), inverse = h_inv) }
        s_g_k[k] ~ g(s_k_min) where { meta = DeltaMeta(method = Linearization()) }
        u_s_sum[k] ~ s_g_k[k] + u_h_k[k]
        s[k] ~ MvNormal(mean = u_s_sum[k], precision = Gamma)
        x[k] ~ MvNormal(mean = s[k], cov = Theta)
        x[k] ~ MvNormal(mean = m_x[k], cov = V_x[k]) # goal
        s_k_min = s[k]
    end
    
    return (s, )
end

"""
    create_physics(; engine_force_limit=0.04, friction_coefficient=0.1)

Create physics functions for the Mountain Car environment.
"""
function create_physics(; engine_force_limit = 0.04, friction_coefficient = 0.1)
    # Engine force as function of action
    Fa = (a::Real) -> engine_force_limit * tanh(a) 

    # Friction force as function of velocity
    Ff = (y_dot::Real) -> -friction_coefficient * y_dot 
    
    # Gravitational force (horizontal component) as function of position
    Fg = (y::Real) -> begin
        if y < 0
            0.05*(-2*y - 1)
        else
            0.05*(-(1 + 5*y^2)^(-0.5) - (y^2)*(1 + 5*y^2)^(-3/2) - (y^4)/16)
        end
    end
    
    # The height of the landscape as a function of the horizontal coordinate
    height = (x::Float64) -> begin
        if x < 0
            h = x^2 + x
        else
            h = x * _₂F₁(0.5, 0.5, 1.5, -5*x^2) + x^3 * _₂F₁(1.5, 1.5, 2.5, -5*x^2) / 3 + x^5 / 80
        end
        return 0.05*h
    end

    return (Fa, Ff, Fg, height)
end

"""
    create_world(; Fg, Ff, Fa, initial_position=-0.5, initial_velocity=0.0)

Create a simulation world with the given physics and initial conditions.
"""
function create_world(; Fg, Ff, Fa, initial_position = -0.5, initial_velocity = 0.0)
    y_t_min = initial_position
    y_dot_t_min = initial_velocity
    
    y_t = y_t_min
    y_dot_t = y_dot_t_min
    
    execute = (a_t::Float64) -> begin
        # Compute next state
        y_dot_t = y_dot_t_min + Fg(y_t_min) + Ff(y_dot_t_min) + Fa(a_t)
        y_t = y_t_min + y_dot_t
    
        # Reset state for next step
        y_t_min = y_t
        y_dot_t_min = y_dot_t
    end
    
    observe = () -> begin 
        return [y_t, y_dot_t]
    end
        
    return (execute, observe)
end

"""
    create_agent(; T = 20, Fg, Fa, Ff, engine_force_limit, x_target, initial_position, initial_velocity)

Create an active inference agent for the Mountain Car problem.
"""
function create_agent(; T = 20, Fg, Fa, Ff, engine_force_limit, x_target, initial_position, initial_velocity)
    Epsilon = fill(huge, 1, 1)                # Control prior variance
    m_u = Vector{Float64}[ [ 0.0] for k=1:T ] # Set control priors
    V_u = Matrix{Float64}[ Epsilon for k=1:T ]

    Sigma    = 1e-4*diageye(2) # Goal prior variance
    m_x      = [zeros(2) for k=1:T]
    V_x      = [huge*diageye(2) for k=1:T]
    V_x[end] = Sigma # Set prior to reach goal at t=T

    # Set initial brain state prior
    m_s_t_min = [initial_position, initial_velocity] 
    V_s_t_min = tiny * diageye(2)
    
    # Set current inference results
    result = nothing

    # The `compute` function is the heart of the agent
    # It calls the `RxInfer.infer` function to perform Bayesian inference by message passing
    compute = (upsilon_t::Float64, y_hat_t::Vector{Float64}) -> begin
        m_u[1] = [ upsilon_t ] # Register action with the generative model
        V_u[1] = fill(tiny, 1, 1) # Clamp control prior to performed action

        m_x[1] = y_hat_t # Register observation with the generative model
        V_x[1] = tiny*diageye(2) # Clamp goal prior to observation

        data = Dict(:m_u       => m_u, 
                   :V_u       => V_u, 
                   :m_x       => m_x, 
                   :V_x       => V_x,
                   :m_s_t_min => m_s_t_min,
                   :V_s_t_min => V_s_t_min)
        
        model  = mountain_car_model(T = T, Fg = Fg, Fa = Fa, Ff = Ff, engine_force_limit = engine_force_limit) 
        result = infer(model = model, data = data)
    end
    
    # The `act` function returns the inferred best possible action
    act = () -> begin
        if result !== nothing
            return mode(result.posteriors[:u][2])[1]
        else
            return 0.0 # Without inference result we return some 'random' action
        end
    end
    
    # The `future` function returns the inferred future states
    future = () -> begin 
        if result !== nothing 
            return getindex.(mode.(result.posteriors[:s]), 1)
        else
            return zeros(T)
        end
    end

    # The `slide` function modifies the `(m_s_t_min, V_s_t_min)` for the next step
    # and shifts (or slides) the array of future goals `(m_x, V_x)` and inferred actions `(m_u, V_u)`
    slide = () -> begin
        if result !== nothing
            model  = RxInfer.getmodel(result.model)
            (s, )  = RxInfer.getreturnval(model)
            varref = RxInfer.getvarref(model, s) 
            var    = RxInfer.getvariable(varref)
            
            slide_msg_idx = 3 # This index is model dependent
            (m_s_t_min, V_s_t_min) = mean_cov(getrecent(messageout(var[2], slide_msg_idx)))

            m_u = circshift(m_u, -1)
            m_u[end] = [0.0]
            V_u = circshift(V_u, -1)
            V_u[end] = Epsilon

            m_x = circshift(m_x, -1)
            m_x[end] = x_target
            V_x = circshift(V_x, -1)
            V_x[end] = Sigma
        end
    end

    return (compute, act, slide, future)    
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
    
    # Ensure all arrays have the same length for animation
    n_frames = min(length(positions), length(velocities), length(actions))
    
    anim = Plots.@animate for t in 1:n_frames
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

        # Plot predictions if available and valid
        if t <= length(predictions) && !isempty(predictions[t])
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
    using Dates
    using .MountainCar.Visualization

    # Create Outputs directory if it doesn't exist
    mkpath(joinpath(@__DIR__, "Outputs"))

    # Simulation parameters
    const timesteps = 100
    const planning_horizon = 50
    const initial_position = -0.5
    const initial_velocity = 0.0
    const x_target = [0.5, 0.0]  # Target state [position, velocity]

    println("\n🚗 Starting Mountain Car simulation...")
    println("Initial state: position = $initial_position, velocity = $initial_velocity")
    println("Target state: position = $(x_target[1]), velocity = $(x_target[2])")

    # Create physics environment (shared between both approaches)
    Fa, Ff, Fg, height = MountainCar.create_physics(
        engine_force_limit = 0.04,
        friction_coefficient = 0.1
    )

    # Run naive approach (always push right)
    println("\n📈 Running naive approach (always push right)...")
    naive_positions = Float64[]
    naive_velocities = Float64[]
    naive_actions = Float64[]
    local naive_target_time = Inf  # Use local to fix soft scope warning

    # Create world for naive approach
    execute_naive, observe_naive = MountainCar.create_world(
        Fg = Fg, Ff = Ff, Fa = Fa,
        initial_position = initial_position,
        initial_velocity = initial_velocity
    )

    # Run naive simulation
    let state = nothing  # Use let block to create new scope
        for t in 1:timesteps
            state = observe_naive()
            push!(naive_positions, state[1])
            push!(naive_velocities, state[2])
            
            # Always push right with maximum force
            local action = 1.0
            push!(naive_actions, action)
            execute_naive(action)
            
            # Check if target reached
            if abs(state[1] - x_target[1]) < 0.01 && 
               abs(state[2] - x_target[2]) < 0.05 && 
               naive_target_time == Inf
                naive_target_time = t
            end

            # Print progress
            if t % 20 == 0
                println("Step $t: position = $(state[1]), velocity = $(state[2])")
            end
        end
    end

    # Calculate naive energies
    naive_ke, naive_pe, naive_te = MountainCar.calculate_energy(naive_positions, naive_velocities, height)

    # Run Active Inference approach
    println("\n📈 Running Active Inference approach...")
    ai_positions = Float64[]
    ai_velocities = Float64[]
    ai_actions = Float64[]
    ai_predictions = Vector{Float64}[]
    local ai_target_time = Inf  # Use local to fix soft scope warning

    # Create world and agent for Active Inference
    execute_ai, observe_ai = MountainCar.create_world(
        Fg = Fg, Ff = Ff, Fa = Fa,
        initial_position = initial_position,
        initial_velocity = initial_velocity
    )

    compute, act, slide, future = MountainCar.create_agent(
        T = planning_horizon,
        Fa = Fa, Fg = Fg, Ff = Ff,
        engine_force_limit = 0.04,
        x_target = x_target,
        initial_position = initial_position,
        initial_velocity = initial_velocity
    )

    # Run Active Inference simulation in a let block to create new scope
    let current_state = observe_ai()  # Initial observation
        push!(ai_positions, current_state[1])
        push!(ai_velocities, current_state[2])
        compute(0.0, current_state)  # Initial computation

        for t in 1:timesteps
            # Get action and execute
            local action = act()
            push!(ai_actions, action)
            execute_ai(action)

            # Observe and update
            current_state = observe_ai()
            push!(ai_positions, current_state[1])
            push!(ai_velocities, current_state[2])
            
            # Store predictions
            push!(ai_predictions, future())

            # Check if target reached
            if abs(current_state[1] - x_target[1]) < 0.01 && 
               abs(current_state[2] - x_target[2]) < 0.05 && 
               ai_target_time == Inf
                ai_target_time = t
            end

            # Update agent's beliefs
            compute(action, current_state)
            slide()

            # Print progress
            if t % 20 == 0
                println("Step $t: position = $(current_state[1]), velocity = $(current_state[2])")
            end
        end
    end

    # Calculate Active Inference energies
    ai_ke, ai_pe, ai_te = MountainCar.calculate_energy(ai_positions, ai_velocities, height)

    # Create comparison visualizations
    println("\n📊 Creating visualizations...")
    
    # Package data for visualization
    naive_data = (naive_positions, naive_velocities, naive_actions, naive_te)
    ai_data = (ai_positions, ai_velocities, ai_actions, ai_te, ai_predictions)
    
    # Create output directory for analysis
    analysis_dir = mkpath(joinpath(@__DIR__, "Outputs", "analysis_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"))
    
    # Create comparison plots
    create_comparison_animation(naive_data, ai_data, height, joinpath(analysis_dir, "comparison.gif"))
    plot_comparison(naive_data, ai_data, height, analysis_dir, 
                   naive_target_time=naive_target_time, 
                   ai_target_time=ai_target_time)
    save_analysis_plots(naive_data, ai_data, analysis_dir)

    # Print final statistics
    println("\n📊 Simulation Statistics:")
    println("\nNaive Approach:")
    println("  Final position: $(naive_positions[end])")
    println("  Final velocity: $(naive_velocities[end])")
    println("  Target reached: $(naive_target_time < Inf ? "Yes, at step $(naive_target_time)" : "No")")
    println("  Average kinetic energy: $(mean(naive_ke))")
    println("  Average potential energy: $(mean(naive_pe))")
    println("  Average total energy: $(mean(naive_te))")

    println("\nActive Inference:")
    println("  Final position: $(ai_positions[end])")
    println("  Final velocity: $(ai_velocities[end])")
    println("  Target reached: $(ai_target_time < Inf ? "Yes, at step $(ai_target_time)" : "No")")
    println("  Average kinetic energy: $(mean(ai_ke))")
    println("  Average potential energy: $(mean(ai_pe))")
    println("  Average total energy: $(mean(ai_te))")

    println("\n✅ Simulation complete! Check the Outputs/analysis directory for visualizations.")
end