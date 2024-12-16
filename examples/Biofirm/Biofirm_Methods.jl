using HypergeometricFunctions: _₂F₁
using RxInfer
using RxInfer: getmodel, getreturnval, getvarref, getvariable
using RxInfer.ReactiveMP: getrecent, messageout

# Constants for numerical stability
const HUGE = 1e10
const TINY = 1e-10

"""
    create_physics(; engine_force_limit = 0.04, friction_coefficient = 0.1)

Create the physics functions for the mountain car environment.
Returns tuple of (Fa, Ff, Fg, height) where:
- Fa: Engine force function
- Ff: Friction force function
- Fg: Gravitational force function
- height: Landscape height function

References:
- Friston, K. et al. (2017). Active Inference: A Process Theory
- DOI: 10.1162/NECO_a_00912
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

    # The height of the landscape as function of horizontal coordinate
    height = (x::Float64) -> begin
        if x < 0
            h = x^2 + x
        else 
            h = x * _₂F₁(0.5,0.5,1.5, -5*x^2) + x^3 * _₂F₁(1.5, 1.5, 2.5, -5*x^2) / 3 + x^5 / 80
        end
        return 0.05*h
    end

    return (Fa, Ff, Fg, height)
end

"""
    create_world(; Fg, Ff, Fa, initial_position = -0.5, initial_velocity = 0.0)

Create a simulation environment for the mountain car.
Returns tuple of (execute, observe) functions where:
- execute: Function to execute an action
- observe: Function to get current state
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

@model function mountain_car(m_u, V_u, m_x, V_x, m_s_t_min, V_s_t_min, T, Fg, Fa, Ff, engine_force_limit)
    # Transition function modeling transition due to gravity and friction
    g = (s_t_min::AbstractVector) -> begin
        s_t = similar(s_t_min)
        s_t[2] = s_t_min[2] + Fg(s_t_min[1]) + Ff(s_t_min[2])
        s_t[1] = s_t_min[1] + s_t[2]
        return s_t
    end
    
    # Function for modeling engine control
    h = (u::AbstractVector) -> [0.0, Fa(u[1])]
    
    # Inverse engine force
    h_inv = (delta_s_dot::AbstractVector) -> [atanh(clamp(delta_s_dot[2], -engine_force_limit+1e-3, engine_force_limit-1e-3)/engine_force_limit)]
    
    # Internal model parameters
    Gamma = 1e4*diageye(2)  # Precision of state transitions
    Theta = 1e-4*diageye(2) # Precision of observations

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
        x[k] ~ MvNormal(mean = m_x[k], cov = V_x[k])
        s_k_min = s[k]
    end
    
    return (s, )
end

"""
    create_agent(;T = 20, Fg, Fa, Ff, engine_force_limit, x_target, initial_position, initial_velocity)

Create an Active Inference agent for the mountain car problem.

Parameters:
- T: Planning horizon
- Fg, Fa, Ff: Physics functions
- engine_force_limit: Maximum engine force
- x_target: Target state [position, velocity]
- initial_position, initial_velocity: Initial state

Returns:
- compute: Function to update beliefs
- act: Function to select action
- slide: Function to shift beliefs forward
- future: Function to predict future states
"""
function create_agent(;T = 20, Fg, Fa, Ff, engine_force_limit, x_target, initial_position, initial_velocity)
    # Initialize beliefs
    Epsilon = fill(HUGE, 1, 1)
    m_u = Vector{Float64}[ [ 0.0] for k=1:T ]
    V_u = Matrix{Float64}[ Epsilon for k=1:T ]

    Sigma = 1e-4*diageye(2)
    m_x = [zeros(2) for k=1:T]
    V_x = [HUGE*diageye(2) for k=1:T]
    V_x[end] = Sigma

    m_s_t_min = [initial_position, initial_velocity]
    V_s_t_min = TINY * diageye(2)
    
    result = nothing

    # Define agent functions
    compute = (upsilon_t::Float64, y_hat_t::Vector{Float64}) -> begin
        m_u[1] = [ upsilon_t ]
        V_u[1] = fill(TINY, 1, 1)

        m_x[1] = y_hat_t
        V_x[1] = TINY*diageye(2)

        data = Dict(:m_u => m_u,
                   :V_u => V_u,
                   :m_x => m_x,
                   :V_x => V_x,
                   :m_s_t_min => m_s_t_min,
                   :V_s_t_min => V_s_t_min)
        
        model = mountain_car(T = T, Fg = Fg, Fa = Fa, Ff = Ff, engine_force_limit = engine_force_limit)
        result = infer(model = model, data = data)
    end
    
    act = () -> begin
        if result !== nothing
            return mode(result.posteriors[:u][2])[1]
        else
            return 0.0
        end
    end
    
    future = () -> begin
        if result !== nothing
            return getindex.(mode.(result.posteriors[:s]), 1)
        else
            return zeros(T)
        end
    end

    slide = () -> begin
        model = getmodel(result.model)
        (s, ) = getreturnval(model)
        varref = getvarref(model, s)
        var = getvariable(varref)
        
        slide_msg_idx = 3
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

    return (compute, act, slide, future)
end

export create_physics, create_world, create_agent