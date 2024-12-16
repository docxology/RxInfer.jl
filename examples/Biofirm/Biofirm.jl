# Active Inference Mountain Car Simulation
# 
# This simulation demonstrates Active Inference in action using a mountain car example.
# The car must escape a valley by building up momentum through oscillatory movements.
#
# Based on the paper: "Active Inference: A Process Theory" by Friston et al. (2017)
# Reference: https://doi.org/10.1162/NECO_a_00912
#
# The simulation compares two approaches:
# 1. Naive approach: Apply constant rightward force
# 2. Active Inference: Use beliefs about future states to plan actions

# Import required packages
import Pkg; Pkg.activate(".."); Pkg.instantiate()
using RxInfer, Plots
using Printf
include("Biofirm_Methods.jl")

# Ensure output directory exists
mkpath("Outputs")

# Set up initial conditions
println("\n🚗 Setting up Mountain Car simulation...")
engine_force_limit = 0.04    # Maximum force the engine can apply
friction_coefficient = 0.1   # Coefficient of friction
initial_position = -0.5     # Starting position in the valley
initial_velocity = 0.0      # Starting velocity
x_target = [0.5, 0.0]      # Target state [position, velocity]

# Create physics environment
@printf("⚙️  Initializing physics (engine_force_limit=%.3f, friction=%.3f)...\n", 
        engine_force_limit, friction_coefficient)
Fa, Ff, Fg, height = create_physics(
    engine_force_limit = engine_force_limit,
    friction_coefficient = friction_coefficient
)

@printf("📍 Initial state: position=%.2f, velocity=%.2f\n", initial_position, initial_velocity)
@printf("🎯 Target state: position=%.2f, velocity=%.2f\n", x_target[1], x_target[2])

# Visualize mountain landscape
println("\n📊 Generating environment visualization...")
valley_x = range(-2, 2, length=400)
valley_y = [ height(xs) for xs in valley_x ]
p1 = plot(valley_x, valley_y, 
    title = "Mountain Car Environment", 
    label = "Landscape", 
    color = "black",
    xlabel = "Position",
    ylabel = "Height",
    legend = :topright
)
scatter!([ initial_position ], [ height(initial_position) ], 
    label="Initial Position", 
    color="blue",
    markersize=6
)
scatter!([x_target[1]], [height(x_target[1])], 
    label="Target", 
    color="green",
    markersize=6
)
savefig(p1, "Outputs/01_environment.png")
println("✅ Saved environment plot to Outputs/01_environment.png")

# Try naive approach first
println("\n🔄 Running naive approach simulation...")
N_naive = 100 # Total simulation time
pi_naive = 100.0 * ones(N_naive) # Naive policy: constant rightward force

(execute_naive, observe_naive) = create_world(
    Fg = Fg, Ff = Ff, Fa = Fa,
    initial_position = initial_position,
    initial_velocity = initial_velocity
)

y_naive = Vector{Vector{Float64}}(undef, N_naive)
for t = 1:N_naive
    execute_naive(pi_naive[t])
    y_naive[t] = observe_naive()
    if t % 20 == 0
        @printf("⏳ Naive simulation progress: %d%%\n", Int(round(100*t/N_naive)))
    end
end

# Run Active Inference simulation
println("\n🧠 Running Active Inference simulation...")
(execute_ai, observe_ai) = create_world(
    Fg = Fg, Ff = Ff, Fa = Fa,
    initial_position = initial_position,
    initial_velocity = initial_velocity
)

# Active Inference parameters
T_ai = 50   # Planning horizon (how far ahead to plan)
N_ai = 100  # Total simulation time
@printf("📝 Configuration: Planning horizon=%d steps, Total time=%d steps\n", T_ai, N_ai)

(compute_ai, act_ai, slide_ai, future_ai) = create_agent(
    T = T_ai,
    Fa = Fa, 
    Fg = Fg,
    Ff = Ff,
    engine_force_limit = engine_force_limit,
    x_target = x_target,
    initial_position = initial_position,
    initial_velocity = initial_velocity
)

# Run simulation and collect results
agent_a = Vector{Float64}(undef, N_ai)    # Actions taken
agent_f = Vector{Vector{Float64}}(undef, N_ai)  # Future predictions
agent_x = Vector{Vector{Float64}}(undef, N_ai)  # Actual states

for t=1:N_ai
    agent_a[t] = act_ai()
    agent_f[t] = future_ai()
    execute_ai(agent_a[t])
    agent_x[t] = observe_ai()
    compute_ai(agent_a[t], agent_x[t])
    slide_ai()
    if t % 20 == 0
        @printf("⏳ Active Inference simulation progress: %d%%\n", Int(round(100*t/N_ai)))
        @printf("   Current position: %.3f, velocity: %.3f\n", 
                agent_x[t][1], agent_x[t][2])
    end
end

# Visualize results
println("\n📊 Generating result visualizations...")

# Plot trajectories in phase space
p2 = plot(getindex.(y_naive, 1), getindex.(y_naive, 2),
    title = "Phase Space Trajectories",
    label = "Naive approach",
    xlabel = "Position",
    ylabel = "Velocity",
    color = "red",
    legend = :topright
)
plot!(getindex.(agent_x, 1), getindex.(agent_x, 2),
    label = "Active Inference",
    color = "blue"
)
scatter!([initial_position], [initial_velocity], 
    label = "Start",
    color = "green",
    markersize = 6
)
scatter!([x_target[1]], [x_target[2]], 
    label = "Target",
    color = "orange",
    markersize = 6
)
savefig(p2, "Outputs/02_phase_space.png")
println("✅ Saved phase space plot to Outputs/02_phase_space.png")

# Plot position over time
p3 = plot(getindex.(y_naive, 1),
    title = "Position over Time",
    label = "Naive approach",
    xlabel = "Time step",
    ylabel = "Position",
    color = "red",
    legend = :bottomright
)
plot!(getindex.(agent_x, 1),
    label = "Active Inference",
    color = "blue"
)
hline!([x_target[1]], 
    label = "Target position",
    color = "green",
    linestyle = :dash
)
savefig(p3, "Outputs/03_position_time.png")
println("✅ Saved position-time plot to Outputs/03_position_time.png")

# Plot actions over time
p4 = plot(agent_a,
    title = "Control Actions over Time",
    label = "Active Inference",
    xlabel = "Time step",
    ylabel = "Action (force)",
    color = "blue",
    legend = :topright
)
hline!([engine_force_limit, -engine_force_limit], 
    label = ["Force limit" nothing],
    color = "red",
    linestyle = :dash
)
savefig(p4, "Outputs/04_actions.png")
println("✅ Saved actions plot to Outputs/04_actions.png")

println("\n✨ Simulation complete! Results saved in Outputs/ directory")
println("📈 Generated visualizations:")
println("   - 01_environment.png: Mountain car environment")
println("   - 02_phase_space.png: Phase space trajectories")
println("   - 03_position_time.png: Position over time comparison")
println("   - 04_actions.png: Control actions over time")

# Print final statistics
println("\n📊 Simulation Statistics:")
@printf("Naive approach final position: %.3f\n", y_naive[end][1])
@printf("Active Inference final position: %.3f\n", agent_x[end][1])
@printf("Distance to target: %.3f\n", abs(agent_x[end][1] - x_target[1]))
