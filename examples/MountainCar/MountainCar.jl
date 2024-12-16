# Active Inference Mountain Car Simulation
# 
# This simulation demonstrates Active Inference in action using a mountain car example.
# The car must escape a valley by building up momentum through oscillatory movements.
#
# Based on:
# - van de Laar, T. W., & de Vries, B. (2019). Simulating active inference processes by message passing.
#   Frontiers in Robotics and AI, 6, 20.
# - Ueltzhöffer, K. (2018). Deep active inference.
#   Biological Cybernetics, 112(6), 547-573.

# Ensure we're in the right environment
import Pkg
if !isfile(joinpath(@__DIR__, "Project.toml"))
    error("Project.toml not found. Please run Setup.jl first")
end
Pkg.activate(@__DIR__)

# Import required packages
using RxInfer
using Plots
using Printf
using Statistics
using Distributions
using TOML

# Include local files
include("MountainCar_Methods.jl")
include("visualization.jl")

# Load configuration
println("\n📝 Loading configuration...")
config = TOML.parsefile(joinpath(@__DIR__, "config.toml"))

# Ensure output directory exists
mkpath("Outputs")

# Set up initial conditions
println("\n🚗 Setting up Mountain Car simulation...")
engine_force_limit = config["physics"]["engine_force_limit"]
friction_coefficient = config["physics"]["friction_coefficient"]
initial_position = config["initial_state"]["position"]
initial_velocity = config["initial_state"]["velocity"]
x_target = [config["target_state"]["position"], config["target_state"]["velocity"]]

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
p = plot_landscape(height, 
    xlims=config["visualization"]["landscape_xlims"],
    ylims=config["visualization"]["landscape_ylims"]
)
scatter!([initial_position], [height(initial_position)], 
    label="Initial Position", 
    color="blue",
    markersize=6
)
scatter!([x_target[1]], [height(x_target[1])], 
    label="Target", 
    color="green",
    markersize=6
)
savefig(p, "Outputs/01_environment.png")
println("✅ Saved environment plot to Outputs/01_environment.png")

# Try naive approach first
println("\n🔄 Running naive approach simulation...")
N_naive = config["simulation"]["naive_timesteps"]
times = 1:N_naive

(execute_naive, observe_naive) = create_world(
    Fg = Fg, Ff = Ff, Fa = Fa,
    initial_position = initial_position,
    initial_velocity = initial_velocity
)

naive_positions = Float64[]
naive_velocities = Float64[]
naive_actions = fill(engine_force_limit, N_naive) # Constant rightward force

for t = 1:N_naive
    if t % 20 == 0
        @printf("⏳ Naive simulation progress: %d%%\n", Int(round(100*t/N_naive)))
    end
    
    execute_naive(naive_actions[t])
    state = observe_naive()
    push!(naive_positions, state[1])
    push!(naive_velocities, state[2])
end

# Save naive approach visualizations
println("\n📈 Saving naive approach results...")
p_naive = plot_simulation_summary(naive_positions, naive_velocities, height, times, actions=naive_actions)
savefig(p_naive, "Outputs/02_naive_summary.png")
println("✅ Saved naive approach summary to Outputs/02_naive_summary.png")

println("\n🎬 Creating naive approach animation...")
create_simulation_animation(naive_positions, naive_velocities, height, fps=config["visualization"]["fps"])
mv("simulation.gif", "Outputs/03_naive_animation.gif", force=true)
println("✅ Saved naive approach animation to Outputs/03_naive_animation.gif")

# Run Active Inference simulation
println("\n🧠 Running Active Inference simulation...")
(execute_ai, observe_ai) = create_world(
    Fg = Fg, Ff = Ff, Fa = Fa,
    initial_position = initial_position,
    initial_velocity = initial_velocity
)

# Active Inference parameters
T_ai = config["simulation"]["planning_horizon"]
N_ai = config["simulation"]["active_inference_timesteps"]
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
ai_positions = Float64[]
ai_velocities = Float64[]
ai_actions = Float64[]
ai_predicted_positions = Vector{Float64}[]

for t=1:N_ai
    # Get current state
    state = observe_ai()
    push!(ai_positions, state[1])
    push!(ai_velocities, state[2])
    
    # Store predicted future positions
    push!(ai_predicted_positions, future_ai())
    
    # Compute and execute action
    compute_ai(0.0, state)
    action = act_ai()
    push!(ai_actions, action)
    execute_ai(action)
    slide_ai()
    
    if t % 20 == 0
        @printf("⏳ Active Inference simulation progress: %d%%\n", Int(round(100*t/N_ai)))
        @printf("   Current position: %.3f, velocity: %.3f\n", state[1], state[2])
    end
end

# Save Active Inference visualizations
println("\n📈 Saving Active Inference results...")
p_ai = plot_simulation_summary(ai_positions, ai_velocities, height, times, actions=ai_actions)
savefig(p_ai, "Outputs/04_ai_summary.png")
println("✅ Saved Active Inference summary to Outputs/04_ai_summary.png")

println("\n🎬 Creating Active Inference animation...")
create_simulation_animation(ai_positions, ai_velocities, height, fps=config["visualization"]["fps"])
mv("simulation.gif", "Outputs/05_ai_animation.gif", force=true)
println("✅ Saved Active Inference animation to Outputs/05_ai_animation.gif")

# Create predictions visualization
println("\n🔮 Generating predictions visualization...")
p_pred = plot(layout=grid(2,1), size=(800, 600))

# Position predictions
plot!(p_pred[1], times, ai_positions, 
    label="Actual", 
    color=:black, 
    linewidth=2,
    title="Position Predictions",
    xlabel="Time Step",
    ylabel="Position")

for t in 1:10:N_ai
    # Calculate valid prediction range
    pred_range = t:min(t+T_ai-1, N_ai)
    # Only plot if we have valid predictions
    if length(pred_range) > 1
        plot!(p_pred[1], pred_range, ai_predicted_positions[t][1:length(pred_range)], 
              label=t==1 ? "Predictions" : nothing,
              color=:blue, 
              alpha=0.2)
    end
end

# Velocity trajectory
plot!(p_pred[2], times, ai_velocities, 
    label="Actual", 
    color=:black, 
    linewidth=2,
    title="Velocity Trajectory",
    xlabel="Time Step",
    ylabel="Velocity")

savefig(p_pred, "Outputs/06_predictions.png")
println("✅ Saved predictions visualization to Outputs/06_predictions.png")

println("\n✨ Simulation complete! Results saved in Outputs/ directory")
println("📈 Generated visualizations:")
println("   - 01_environment.png: Mountain car environment")
println("   - 02_naive_summary.png: Naive approach summary")
println("   - 03_naive_animation.gif: Naive approach animation")
println("   - 04_ai_summary.png: Active Inference summary")
println("   - 05_ai_animation.gif: Active Inference animation")
println("   - 06_predictions.png: Predictions visualization")

# Print final statistics
println("\n📊 Simulation Statistics:")
@printf("Naive approach final position: %.3f\n", naive_positions[end])
@printf("Active Inference final position: %.3f\n", ai_positions[end])
@printf("Distance to target: %.3f\n", abs(ai_positions[end] - x_target[1]))

# After running both simulations, create the comparison animation
println("\n🎬 Creating comparison animation...")
create_comparison_animation(
    naive_positions, naive_velocities,
    ai_positions, ai_velocities,
    height, fps=config["visualization"]["fps"]
)
mv("comparison.gif", "Outputs/07_comparison.gif", force=true)
println("✅ Saved comparison animation to Outputs/07_comparison.gif")

# Update the individual animations with trace colors
println("\n🎬 Creating naive approach animation...")
create_simulation_animation(
    naive_positions, naive_velocities, height,
    fps=config["visualization"]["fps"], trace_color=:red, title_prefix="Naive Approach"
)
mv("simulation.gif", "Outputs/03_naive_animation.gif", force=true)
println("✅ Saved naive approach animation to Outputs/03_naive_animation.gif")

println("\n🎬 Creating Active Inference animation...")
create_simulation_animation(
    ai_positions, ai_velocities, height,
    fps=config["visualization"]["fps"], trace_color=:blue, title_prefix="Active Inference"
)
mv("simulation.gif", "Outputs/05_ai_animation.gif", force=true)
println("✅ Saved Active Inference animation to Outputs/05_ai_animation.gif")

# Update the final output message
println("\n✨ Simulation complete! Results saved in Outputs/ directory")
println("📈 Generated visualizations:")
println("   - 01_environment.png: Mountain car environment")
println("   - 02_naive_summary.png: Naive approach summary")
println("   - 03_naive_animation.gif: Naive approach animation")
println("   - 04_ai_summary.png: Active Inference summary")
println("   - 05_ai_animation.gif: Active Inference animation")
println("   - 06_predictions.png: Predictions visualization")
println("   - 07_comparison.gif: Side-by-side comparison animation")
