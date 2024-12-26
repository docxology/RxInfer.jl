"""
    MountainCar.jl

Mountain Car environment implementation with configurable physics parameters.
"""

using RxInfer
using RxInfer: getmodel, getreturnval, getvarref, getvariable
using RxInfer.ReactiveMP: getrecent, messageout
using HypergeometricFunctions
using LinearAlgebra

"""
    MountainCarEnv

Mountain Car environment with configurable physics parameters.
"""
mutable struct MountainCarEnv
    # Physics parameters
    force::Float64  # Engine force limit
    friction::Float64  # Friction coefficient
    gravity::Float64  # Gravity constant
    
    # State variables
    position::Float64
    velocity::Float64
    target_position::Float64
    
    # Constructor
    function MountainCarEnv(;
        force::Float64=0.001,
        friction::Float64=0.0,
        gravity::Float64=0.0025,
        initial_position::Float64=-0.5,
        initial_velocity::Float64=0.0,
        target_position::Float64=0.6
    )
        new(force, friction, gravity, initial_position, initial_velocity, target_position)
    end
end

"""
    reset!(env::MountainCarEnv)

Reset the environment to its initial state.
"""
function reset!(env::MountainCarEnv)
    env.position = -0.5
    env.velocity = 0.0
    return nothing
end

"""
    height(position::Float64)

Calculate height at a given position using the mountain shape function.
"""
function height(position::Float64)
    return cos(3.0 * position)
end

"""
    step!(env::MountainCarEnv, action::Float64)

Update the environment state given an action.
Returns (reward, done).
"""
function step!(env::MountainCarEnv, action::Float64)
    # Clamp action to [-1, 1]
    action = clamp(action, -1.0, 1.0)
    
    # Calculate forces
    engine_force = action * env.force
    gravity_force = env.gravity * sin(3.0 * env.position)
    friction_force = -env.friction * env.velocity
    
    # Update velocity
    env.velocity += engine_force + gravity_force + friction_force
    env.velocity = clamp(env.velocity, -0.07, 0.07)
    
    # Update position
    env.position += env.velocity
    env.position = clamp(env.position, -1.2, 0.6)
    
    # Check if velocity is zero at leftmost position
    if env.position ≈ -1.2 && env.velocity < 0
        env.velocity = 0.0
    end
    
    # Calculate reward and check if done
    reward = -1.0  # Constant penalty to encourage reaching target quickly
    done = abs(env.position - env.target_position) < 0.01
    
    return reward, done
end

"""
    NaiveAgent

Simple agent that uses a heuristic strategy.
"""
mutable struct NaiveAgent
    env::MountainCarEnv
end

"""
    get_action(agent::NaiveAgent, env::MountainCarEnv)

Get action from naive agent using heuristic strategy.
"""
function get_action(agent::NaiveAgent, env::MountainCarEnv)
    if abs(env.velocity) < 1e-5
        # When nearly stationary, push right to start moving
        return 1.0
    else
        # Otherwise, push in direction of motion to maintain momentum
        return sign(env.velocity)
    end
end

"""
    ActiveInferenceAgent

Agent that uses active inference for control.
"""
mutable struct ActiveInferenceAgent
    env::MountainCarEnv
    planning_horizon::Int
    beliefs::Dict{Symbol,Any}
    
    function ActiveInferenceAgent(env::MountainCarEnv; planning_horizon::Int=15)
        # Initialize beliefs
        beliefs = Dict{Symbol,Any}()
        
        # Create generative model
        model = create_generative_model(env, planning_horizon)
        beliefs[:model] = model
        
        # Initialize message passing
        beliefs[:messages] = initialize_messages(model)
        
        new(env, planning_horizon, beliefs)
    end
end

"""
    create_generative_model(env::MountainCarEnv, T::Int)

Create the generative model for active inference.
"""
function create_generative_model(env::MountainCarEnv, T::Int)
    # Model parameters
    dt = 1.0
    σ_x = 0.01  # State noise
    σ_v = 0.01  # Velocity noise
    σ_a = 0.1   # Action noise
    
    # Create model
    @model function mountain_car()
        # Prior preferences for final state
        μ_x = constvar(env.target_position)
        μ_v = constvar(0.0)
        
        # Initial state
        x_prev = randomvar(1)
        v_prev = randomvar(1)
        
        # Dynamics over time
        for t in 1:T
            # Action selection (to be inferred)
            a_t ~ NormalMeanVariance(0.0, σ_a^2)
            
            # State transition
            engine_force = a_t * env.force
            gravity_force = env.gravity * sin(3.0 * x_prev)
            friction_force = -env.friction * v_prev
            
            # Velocity update
            v_t ~ NormalMeanVariance(v_prev + engine_force + gravity_force + friction_force, σ_v^2)
            v_t = clamp(v_t, -0.07, 0.07)
            
            # Position update
            x_t ~ NormalMeanVariance(x_prev + v_t * dt, σ_x^2)
            x_t = clamp(x_t, -1.2, 0.6)
            
            # Update previous state
            x_prev = x_t
            v_prev = v_t
            
            # Add observations
            y_x_t ~ NormalMeanVariance(x_t, σ_x^2)
            y_v_t ~ NormalMeanVariance(v_t, σ_v^2)
        end
        
        # Target state preference
        x_T ~ NormalMeanVariance(μ_x, σ_x^2)
        v_T ~ NormalMeanVariance(μ_v, σ_v^2)
    end
    
    return mountain_car
end

"""
    initialize_messages(model)

Initialize message passing for the active inference agent.
"""
function initialize_messages(model)
    # Initialize messages dictionary
    messages = Dict{Symbol,Any}()
    
    # Create message passing schedule
    messages[:schedule] = [:forward_pass, :backward_pass]
    
    # Initialize beliefs over states and actions
    messages[:state_beliefs] = []
    messages[:action_beliefs] = []
    
    return messages
end

"""
    get_action(agent::ActiveInferenceAgent, env::MountainCarEnv)

Get action from active inference agent using belief updates.
"""
function get_action(agent::ActiveInferenceAgent, env::MountainCarEnv)
    # Update beliefs based on current state
    update_beliefs!(agent, env)
    
    # Get action from current beliefs
    action_belief = agent.beliefs[:messages][:action_beliefs][1]
    action = mean(action_belief)
    
    return clamp(action, -1.0, 1.0)
end

"""
    update_beliefs!(agent::ActiveInferenceAgent, env::MountainCarEnv)

Update agent's beliefs based on current environment state.
"""
function update_beliefs!(agent::ActiveInferenceAgent, env::MountainCarEnv)
    # Get current state
    x = env.position
    v = env.velocity
    
    # Update model with current observations
    model = agent.beliefs[:model]
    messages = agent.beliefs[:messages]
    
    # Perform belief propagation
    for step in messages[:schedule]
        if step == :forward_pass
            # Forward pass: update state beliefs
            update_state_beliefs!(messages, x, v)
        else
            # Backward pass: update action beliefs
            update_action_beliefs!(messages)
        end
    end
end

"""
    update_state_beliefs!(messages::Dict{Symbol,Any}, x::Float64, v::Float64)

Update beliefs about states based on observations.
"""
function update_state_beliefs!(messages::Dict{Symbol,Any}, x::Float64, v::Float64)
    # Get current beliefs
    state_beliefs = messages[:state_beliefs]
    
    # Create observation message
    obs_msg = Message(Multivariate, GaussianMeanVariance, [x, v], [1e-4 0.0; 0.0 1e-4])
    
    # Update state beliefs with observation
    if isempty(state_beliefs)
        # Initialize beliefs with observation
        push!(state_beliefs, obs_msg)
    else
        # Update existing beliefs
        state_beliefs[1] = obs_msg
    end
end

"""
    update_action_beliefs!(messages::Dict{Symbol,Any})

Update beliefs about actions based on state beliefs and preferences.
"""
function update_action_beliefs!(messages::Dict{Symbol,Any})
    # Get current beliefs
    state_beliefs = messages[:state_beliefs]
    action_beliefs = messages[:action_beliefs]
    
    if isempty(state_beliefs)
        # No state beliefs yet, initialize with zero action
        if isempty(action_beliefs)
            push!(action_beliefs, Message(Univariate, GaussianMeanVariance, 0.0, 1.0))
        end
        return
    end
    
    # Get current state belief
    state = mean(state_beliefs[1])
    
    # Simple control law: push towards target
    desired_action = if abs(state[2]) < 1e-5  # If velocity is near zero
        sign(0.6 - state[1])  # Push towards target
    else
        sign(state[2])  # Maintain momentum
    end
    
    # Create action message with some uncertainty
    action_msg = Message(Univariate, GaussianMeanVariance, desired_action, 0.1)
    
    # Update action beliefs
    if isempty(action_beliefs)
        push!(action_beliefs, action_msg)
    else
        action_beliefs[1] = action_msg
    end
end