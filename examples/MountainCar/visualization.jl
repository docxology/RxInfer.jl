module Visualization

using Plots
using Measures
using Printf
using Statistics
using StatsBase
using DataFrames

export create_animation, plot_comparison, save_analysis_plots, create_comparison_animation
export analyze_trajectory, create_heatmap_analysis, create_differential_analysis
export create_parameter_sweep_visualization

# Constants for visualization
const DEFAULT_Y_RANGE = (-0.1, 0.2)
const DEFAULT_X_RANGE = (-1.5, 1.5)

"""
    create_landscape(height_func, x_range=DEFAULT_X_RANGE)

Create a visualization of the mountain car landscape.
"""
function create_landscape(height_func; x_range=DEFAULT_X_RANGE, points=200)
    x = range(x_range[1], x_range[2], length=points)
    y = map(height_func, x)
    return x, y
end

"""
    calculate_entropy(values; bins=30)

Calculate entropy of a distribution using normalized histogram counts.
"""
function calculate_entropy(values; bins=30)
    hist = fit(Histogram, values, nbins=bins)
    probs = hist.weights / sum(hist.weights)
    # Filter out zero probabilities to avoid NaN in log
    valid_probs = filter(p -> p > 0, probs)
    return -sum(p * log(p) for p in valid_probs)
end

"""
    analyze_trajectory(positions, velocities, actions, energy; name="")

Analyze a trajectory and return key metrics.
"""
function analyze_trajectory(positions, velocities, actions, energy; name="")
    metrics = Dict{String, Any}(
        "name" => name,
        "max_position" => maximum(positions),
        "min_position" => minimum(positions),
        "max_velocity" => maximum(velocities),
        "min_velocity" => minimum(velocities),
        "mean_velocity" => mean(velocities),
        "velocity_std" => std(velocities),
        "mean_action" => mean(actions),
        "action_std" => std(actions),
        "total_energy" => sum(abs.(energy)),
        "mean_energy" => mean(energy),
        "energy_std" => std(energy),
        "position_entropy" => calculate_entropy(positions),
        "velocity_entropy" => calculate_entropy(velocities),
        "action_entropy" => calculate_entropy(actions)
    )
    return metrics
end

"""
    create_heatmap_analysis(positions, velocities, actions; bins=30)

Create heatmap analysis of the state-action space.
"""
function create_heatmap_analysis(positions, velocities, actions; bins=30)
    # Ensure all arrays have the same length
    n = min(length(positions), length(velocities), length(actions))
    positions = positions[1:n]
    velocities = velocities[1:n]
    actions = actions[1:n]
    
    # Create state-action heatmap with edges
    pos_edges = range(minimum(positions), maximum(positions), length=bins+1)
    vel_edges = range(minimum(velocities), maximum(velocities), length=bins+1)
    
    state_action_map = zeros(bins, bins)
    visit_count = zeros(Int, bins, bins)
    
    for i in 1:n
        pos_bin = searchsortedfirst(pos_edges, positions[i]) - 1
        vel_bin = searchsortedfirst(vel_edges, velocities[i]) - 1
        
        # Ensure bin indices are within bounds
        pos_bin = clamp(pos_bin, 1, bins)
        vel_bin = clamp(vel_bin, 1, bins)
        
        state_action_map[pos_bin, vel_bin] += actions[i]
        visit_count[pos_bin, vel_bin] += 1
    end
    
    # Average actions in each bin
    for i in 1:bins, j in 1:bins
        if visit_count[i,j] > 0
            state_action_map[i,j] /= visit_count[i,j]
        end
    end
    
    # Create bin centers for plotting
    pos_centers = [(pos_edges[i] + pos_edges[i+1])/2 for i in 1:bins]
    vel_centers = [(vel_edges[i] + vel_edges[i+1])/2 for i in 1:bins]
    
    return pos_centers, vel_centers, state_action_map
end

"""
    create_differential_analysis(positions, velocities, actions, energy)

Create differential analysis of the trajectory.
"""
function create_differential_analysis(positions, velocities, actions, energy)
    # Calculate differentials
    pos_diff = diff(positions)
    vel_diff = diff(velocities)
    action_diff = diff(actions)
    energy_diff = diff(energy)
    
    # Calculate statistics
    metrics = Dict(
        "position_derivatives" => pos_diff,
        "velocity_derivatives" => vel_diff,
        "action_derivatives" => action_diff,
        "energy_derivatives" => energy_diff,
        "position_acceleration" => diff(pos_diff),
        "velocity_acceleration" => diff(vel_diff),
        "mean_position_change" => mean(abs.(pos_diff)),
        "mean_velocity_change" => mean(abs.(vel_diff)),
        "mean_action_change" => mean(abs.(action_diff)),
        "mean_energy_change" => mean(abs.(energy_diff))
    )
    
    return metrics
end

"""
    create_parameter_sweep_visualization(sweep_results, parameter_names; save_dir)

Create visualizations for parameter sweep results.
"""
function create_parameter_sweep_visualization(sweep_results, parameter_names; save_dir)
    # Convert results to DataFrame for easier analysis
    df = DataFrame(sweep_results)
    
    # Create parameter interaction plots
    for i in 1:length(parameter_names)
        for j in i+1:length(parameter_names)
            p = scatter(df[:, parameter_names[i]], df[:, parameter_names[j]],
                       xlabel=parameter_names[i],
                       ylabel=parameter_names[j],
                       title="Parameter Interaction",
                       color=:viridis,
                       alpha=0.6,
                       size=(800, 600),
                       margin=5mm)
            savefig(p, joinpath(save_dir, "param_interaction_$(parameter_names[i])_$(parameter_names[j]).png"))
        end
    end
    
    # Create performance correlation heatmap
    correlation_matrix = cor(Matrix(df[:, parameter_names]))
    p_corr = heatmap(parameter_names, parameter_names, correlation_matrix,
                     title="Parameter Correlations",
                     color=:viridis,
                     size=(800, 800),
                     margin=5mm)
    savefig(p_corr, joinpath(save_dir, "parameter_correlations.png"))
    
    return df
end

"""
    create_animation(positions, velocities, height_func, filename; kwargs...)

Create an animation of the mountain car trajectory.
"""
function create_animation(positions, velocities, height_func, filename;
                         title="Mountain Car Trajectory",
                         x_range=DEFAULT_X_RANGE,
                         y_range=DEFAULT_Y_RANGE,
                         show_predictions=false,
                         predictions=nothing,
                         planning_horizon=20)
    # Create landscape
    landscape_x, landscape_y = create_landscape(height_func, x_range=x_range)
    
    # Create animation
    anim = @animate for i in 1:length(positions)
        # Plot landscape
        plot(landscape_x, landscape_y,
             label="Landscape",
             color=:black,
             linewidth=2,
             xlabel="Position",
             ylabel="Height",
             title="$title (t=$i)",
             size=(800, 600),
             margin=5mm,
             legend=:topright)
        
        # Add predictions if requested
        if show_predictions && predictions !== nothing && i <= length(predictions)
            pred = predictions[i]
            # Plot full prediction trajectory
            if !isempty(pred)
                # Convert predicted positions to heights
                pred_heights = map(height_func, pred[1:min(length(pred), planning_horizon)])
                pred_times = i:min(i+length(pred)-1, i+planning_horizon-1)
                
                # Plot predicted trajectory
                plot!(pred[1:min(length(pred), planning_horizon)], pred_heights,
                      label="Predicted Trajectory",
                      color=:cyan,
                      alpha=0.3,
                      linewidth=2,
                      linestyle=:dash)
                
                # Plot predicted positions as points
                scatter!(pred[1:min(length(pred), planning_horizon)], pred_heights,
                        label=nothing,
                        color=:cyan,
                        alpha=0.3,
                        markersize=4)
            end
        end
        
        # Plot car position
        scatter!([positions[i]], [height_func(positions[i])],
                label="Car",
                markersize=8,
                markershape=:circle,
                color=:red)
        
        # Add velocity vector
        quiver!([positions[i]], [height_func(positions[i])],
                quiver=([velocities[i]], [0]),
                color=:blue,
                label="Velocity")
        
        # Set fixed axis limits
        xlims!(x_range)
        ylims!(y_range)
    end
    
    gif(anim, filename, fps=30)
end

"""
    create_comparison_animation(naive_data, ai_data, height_func, filename)

Create an animation comparing naive control and active inference approaches.
"""
function create_comparison_animation(naive_data, ai_data, height_func, filename;
                                   x_range=DEFAULT_X_RANGE,
                                   y_range=DEFAULT_Y_RANGE,
                                   planning_horizon=20)
    # Unpack data
    (naive_pos, naive_vel, _, _) = naive_data
    (ai_pos, ai_vel, _, _, ai_predictions) = ai_data
    
    # Create landscape
    landscape_x, landscape_y = create_landscape(height_func, x_range=x_range)
    
    # Determine maximum timesteps
    max_steps = max(length(naive_pos), length(ai_pos))
    
    # Create animation
    anim = @animate for i in 1:max_steps
        # Plot landscape
        plot(landscape_x, landscape_y,
             label="Landscape",
             color=:black,
             linewidth=2,
             xlabel="Position",
             ylabel="Height",
             title="Comparison (t=$i)",
             size=(800, 600),
             margin=5mm,
             legend=:topright)
        
        # Plot predictions if available (plot first for proper layering)
        if i <= length(ai_predictions)
            pred = ai_predictions[i]
            if !isempty(pred)
                # Convert predicted positions to heights
                pred_heights = map(height_func, pred[1:min(length(pred), planning_horizon)])
                
                # Plot predicted trajectory
                plot!(pred[1:min(length(pred), planning_horizon)], pred_heights,
                      label="AI Predictions",
                      color=:cyan,
                      alpha=0.3,
                      linewidth=2,
                      linestyle=:dash)
                
                # Plot predicted positions as points
                scatter!(pred[1:min(length(pred), planning_horizon)], pred_heights,
                        label=nothing,
                        color=:cyan,
                        alpha=0.3,
                        markersize=4)
            end
        end
        
        # Plot naive control car if within bounds
        if i <= length(naive_pos)
            scatter!([naive_pos[i]], [height_func(naive_pos[i])],
                    label="Naive Control",
                    markersize=8,
                    markershape=:circle,
                    color=:red)
            
            quiver!([naive_pos[i]], [height_func(naive_pos[i])],
                    quiver=([naive_vel[i]], [0]),
                    color=:red,
                    alpha=0.7,
                    label=nothing)
        end
        
        # Plot active inference car if within bounds
        if i <= length(ai_pos)
            scatter!([ai_pos[i]], [height_func(ai_pos[i])],
                    label="Active Inference",
                    markersize=8,
                    markershape=:circle,
                    color=:blue)
            
            quiver!([ai_pos[i]], [height_func(ai_pos[i])],
                    quiver=([ai_vel[i]], [0]),
                    color=:blue,
                    alpha=0.7,
                    label=nothing)
        end
        
        # Set fixed axis limits
        xlims!(x_range)
        ylims!(y_range)
    end
    
    gif(anim, filename, fps=30)
end

"""
    format_metrics(metrics)

Format metrics dictionary for output, separating numeric and non-numeric values.
"""
function format_metrics(metrics)
    # Separate metrics by type
    numeric_metrics = Dict{String,Float64}()
    other_metrics = Dict{String,Any}()
    
    for (k, v) in metrics
        if typeof(v) <: AbstractFloat || typeof(v) <: Integer
            numeric_metrics[k] = float(v)
        else
            other_metrics[k] = v
        end
    end
    
    return numeric_metrics, other_metrics
end

"""
    write_metrics(io, metrics, header)

Write metrics to IO with proper formatting.
"""
function write_metrics(io, metrics, header)
    println(io, header)
    
    # Split and format metrics
    numeric_metrics, other_metrics = format_metrics(metrics)
    
    # Print non-numeric metrics first
    for (k, v) in sort(collect(other_metrics))
        println(io, "  $k: $v")
    end
    
    # Print numeric metrics with consistent formatting
    for (k, v) in sort(collect(numeric_metrics))
        println(io, "  $k: ", @sprintf("%.6f", v))
    end
end

"""
    plot_comparison(naive_data, ai_data, height_func, save_dir; naive_target_time=Inf, ai_target_time=Inf)

Create enhanced comparison plots between naive and active inference approaches.
"""
function plot_comparison(naive_data, ai_data, height_func, save_dir; naive_target_time=Inf, ai_target_time=Inf)
    # Unpack data
    (naive_pos, naive_vel, naive_actions, naive_energy) = naive_data
    (ai_pos, ai_vel, ai_actions, ai_energy, ai_predictions) = ai_data
    
    # Phase space plot
    p1 = plot(naive_pos, naive_vel,
              label="Naive Control",
              xlabel="Position",
              ylabel="Velocity",
              title="Phase Space Comparison",
              linewidth=2,
              legend=:topright,
              size=(800, 600),
              margin=5mm)
    plot!(p1, ai_pos, ai_vel,
          label="Active Inference",
          linewidth=2)
    savefig(p1, joinpath(save_dir, "phase_space_comparison.png"))
    
    # Energy comparison
    p2 = plot(1:length(naive_energy), naive_energy,
              label="Naive Control",
              xlabel="Time Step",
              ylabel="Total Energy",
              title="Energy Comparison",
              linewidth=2,
              legend=:topright,
              size=(800, 600),
              margin=5mm)
    plot!(p2, 1:length(ai_energy), ai_energy,
          label="Active Inference",
          linewidth=2)
    savefig(p2, joinpath(save_dir, "energy_comparison.png"))
    
    # Action comparison
    p3 = plot(1:length(naive_actions), naive_actions,
              label="Naive Control",
              xlabel="Time Step",
              ylabel="Action",
              title="Control Actions Comparison",
              linewidth=2,
              legend=:topright,
              size=(800, 600),
              margin=5mm)
    plot!(p3, 1:length(ai_actions), ai_actions,
          label="Active Inference",
          linewidth=2)
    savefig(p3, joinpath(save_dir, "actions_comparison.png"))
    
    # Prediction visualization (only for AI)
    if !isempty(ai_predictions)
        p4 = plot(ai_pos,
                 label="Actual Trajectory",
                 xlabel="Time Step",
                 ylabel="Position",
                 title="Active Inference Predictions",
                 linewidth=2,
                 legend=:topright,
                 size=(800, 600),
                 margin=5mm)
        
        # Plot predictions at regular intervals
        pred_interval = max(1, div(length(ai_predictions), 10))
        for i in 1:pred_interval:length(ai_predictions)
            pred = ai_predictions[i]
            pred_times = i:min(i+length(pred)-1, length(ai_pos))
            if !isempty(pred_times)
                plot!(p4, pred_times, pred[1:length(pred_times)],
                      label=i == 1 ? "Predictions" : nothing,
                      color=:gray,
                      alpha=0.3,
                      linewidth=1)
            end
        end
        savefig(p4, joinpath(save_dir, "predictions.png"))
    end
    
    # Add new analyses
    naive_metrics = analyze_trajectory(naive_pos, naive_vel, naive_actions, naive_energy, name="Naive")
    ai_metrics = analyze_trajectory(ai_pos, ai_vel, ai_actions, ai_energy, name="Active Inference")
    
    # Create state-action heatmaps with improved visualization
    naive_pos_centers, naive_vel_centers, naive_heatmap = create_heatmap_analysis(naive_pos, naive_vel, naive_actions)
    p_naive_heat = heatmap(naive_pos_centers, naive_vel_centers, naive_heatmap',
                          title="Naive Control State-Action Map",
                          xlabel="Position",
                          ylabel="Velocity",
                          color=:viridis,
                          colorbar_title="Action",
                          size=(800, 600),
                          margin=5mm)
    # Add trajectory overlay
    plot!(p_naive_heat, naive_pos, naive_vel,
          color=:red,
          alpha=0.5,
          linewidth=2,
          label="Trajectory")
    savefig(p_naive_heat, joinpath(save_dir, "naive_state_action_heatmap.png"))
    
    ai_pos_centers, ai_vel_centers, ai_heatmap = create_heatmap_analysis(ai_pos, ai_vel, ai_actions)
    p_ai_heat = heatmap(ai_pos_centers, ai_vel_centers, ai_heatmap',
                        title="Active Inference State-Action Map",
                        xlabel="Position",
                        ylabel="Velocity",
                        color=:viridis,
                        colorbar_title="Action",
                        size=(800, 600),
                        margin=5mm)
    # Add trajectory overlay
    plot!(p_ai_heat, ai_pos, ai_vel,
          color=:blue,
          alpha=0.5,
          linewidth=2,
          label="Trajectory")
    savefig(p_ai_heat, joinpath(save_dir, "ai_state_action_heatmap.png"))
    
    # Create differential analysis plots with improved visualization
    naive_diff = create_differential_analysis(naive_pos, naive_vel, naive_actions, naive_energy)
    ai_diff = create_differential_analysis(ai_pos, ai_vel, ai_actions, ai_energy)
    
    # Plot phase space with velocity vectors
    p_phase = plot(naive_pos, naive_vel,
                  label="Naive Control",
                  xlabel="Position",
                  ylabel="Velocity",
                  title="Phase Space with Velocity Vectors",
                  linewidth=2,
                  legend=:topright,
                  size=(800, 600),
                  margin=5mm)
    plot!(p_phase, ai_pos, ai_vel,
          label="Active Inference",
          linewidth=2)
    # Add velocity vectors at regular intervals
    interval = max(1, div(length(naive_pos), 20))
    for i in 1:interval:length(naive_pos)
        quiver!([naive_pos[i]], [naive_vel[i]],
                quiver=([naive_diff["position_derivatives"][min(i, end)]], [naive_diff["velocity_derivatives"][min(i, end)]]),
                color=:red,
                alpha=0.3)
    end
    for i in 1:interval:length(ai_pos)
        quiver!([ai_pos[i]], [ai_vel[i]],
                quiver=([ai_diff["position_derivatives"][min(i, end)]], [ai_diff["velocity_derivatives"][min(i, end)]]),
                color=:blue,
                alpha=0.3)
    end
    savefig(p_phase, joinpath(save_dir, "phase_space_vectors.png"))
    
    # Save detailed metrics to file
    open(joinpath(save_dir, "trajectory_metrics.txt"), "w") do io
        println(io, "=== Detailed Trajectory Analysis ===\n")
        
        # Write metrics for each approach
        write_metrics(io, naive_metrics, "Naive Control Metrics:")
        println(io)
        write_metrics(io, naive_diff, "Naive Control Differential Analysis:")
        println(io)
        write_metrics(io, ai_metrics, "Active Inference Metrics:")
        println(io)
        write_metrics(io, ai_diff, "Active Inference Differential Analysis:")
        
        # Add comparison summary
        println(io, "\n=== Comparative Analysis ===")
        
        # Calculate comparison ratios
        energy_ratio = ai_metrics["total_energy"] / naive_metrics["total_energy"]
        velocity_range_ratio = (ai_metrics["max_velocity"] - ai_metrics["min_velocity"]) /
                             (naive_metrics["max_velocity"] - naive_metrics["min_velocity"])
        position_range_ratio = (ai_metrics["max_position"] - ai_metrics["min_position"]) /
                             (naive_metrics["max_position"] - naive_metrics["min_position"])
        
        # Print comparison metrics
        println(io, "Energy Efficiency Ratio (AI/Naive): ", @sprintf("%.3f", energy_ratio))
        println(io, "Velocity Range Ratio (AI/Naive): ", @sprintf("%.3f", velocity_range_ratio))
        println(io, "Position Range Ratio (AI/Naive): ", @sprintf("%.3f", position_range_ratio))
        
        # Add performance summary
        println(io, "\n=== Performance Summary ===")
        println(io, "Time to Target:")
        println(io, "  Naive Control: ", isinf(naive_target_time) ? "Failed" : 
                @sprintf("%.1f timesteps", naive_target_time))
        println(io, "  Active Inference: ", isinf(ai_target_time) ? "Failed" : 
                @sprintf("%.1f timesteps", ai_target_time))
        
        println(io, "\nEnergy Usage:")
        println(io, "  Naive Control: ", @sprintf("%.6f", naive_metrics["total_energy"]))
        println(io, "  Active Inference: ", @sprintf("%.6f", ai_metrics["total_energy"]))
        
        println(io, "\nControl Efficiency:")
        println(io, "  Naive Control mean action: ", @sprintf("%.6f", naive_metrics["mean_action"]))
        println(io, "  Active Inference mean action: ", @sprintf("%.6f", ai_metrics["mean_action"]))
        
        # Add timing analysis
        if !isinf(ai_target_time)
            println(io, "\nTiming Analysis:")
            println(io, "  Active Inference reached target in: ", @sprintf("%.1f timesteps", ai_target_time))
            println(io, "  Energy used until target: ", 
                    @sprintf("%.6f", sum(abs.(ai_energy[1:Int(ai_target_time)]))))
            println(io, "  Average action until target: ",
                    @sprintf("%.6f", mean(abs.(ai_actions[1:Int(ai_target_time)]))))
        end
    end
end

"""
    save_analysis_plots(naive_data, ai_data, save_dir)

Generate and save detailed analysis plots.
"""
function save_analysis_plots(naive_data, ai_data, save_dir)
    # Unpack data
    (naive_pos, naive_vel, naive_actions, naive_energy) = naive_data
    (ai_pos, ai_vel, ai_actions, ai_energy, _) = ai_data
    
    # Velocity distribution
    p1 = histogram(naive_vel,
                  label="Naive Control",
                  xlabel="Velocity",
                  ylabel="Frequency",
                  title="Velocity Distribution",
                  alpha=0.5,
                  size=(800, 600),
                  margin=5mm)
    histogram!(p1, ai_vel,
               label="Active Inference",
               alpha=0.5)
    savefig(p1, joinpath(save_dir, "velocity_distribution.png"))
    
    # Action distribution
    p2 = histogram(naive_actions,
                  label="Naive Control",
                  xlabel="Action",
                  ylabel="Frequency",
                  title="Action Distribution",
                  alpha=0.5,
                  size=(800, 600),
                  margin=5mm)
    histogram!(p2, ai_actions,
               label="Active Inference",
               alpha=0.5)
    savefig(p2, joinpath(save_dir, "action_distribution.png"))
    
    # Energy efficiency
    p3 = plot(cumsum(abs.(naive_energy)),
              label="Naive Control",
              xlabel="Time Step",
              ylabel="Cumulative Energy",
              title="Energy Efficiency",
              linewidth=2,
              legend=:topright,
              size=(800, 600),
              margin=5mm)
    plot!(p3, cumsum(abs.(ai_energy)),
          label="Active Inference",
          linewidth=2)
    savefig(p3, joinpath(save_dir, "energy_efficiency.png"))
end

end # module