using Plots
using Statistics
using Distributions

"""
    plot_landscape(height_fn; xlims=(-1.0, 1.0), ylims=(-0.05, 0.15))

Plot the mountain car landscape with optional position markers.
"""
function plot_landscape(height_fn; xlims=(-1.0, 1.0), ylims=(-0.05, 0.15))
    x = range(xlims[1], xlims[2], length=200)
    y = height_fn.(x)
    
    plot(x, y, 
         label="Landscape",
         color=:black,
         linewidth=2,
         xlabel="Position",
         ylabel="Height",
         title="Mountain Car Environment",
         ylims=ylims)
end

"""
    plot_state_distribution(positions, velocities; bins=30)

Create a heatmap of state visitation distribution.
"""
function plot_state_distribution(positions, velocities; bins=30)
    h = histogram2d(positions, velocities,
                   bins=bins,
                   xlabel="Position",
                   ylabel="Velocity",
                   title="State Visitation Distribution",
                   color=:viridis,
                   colorbar_title="Frequency")
    return h
end

"""
    plot_trajectory_statistics(positions, velocities, times)

Plot trajectory statistics including mean and standard deviation bands.
"""
function plot_trajectory_statistics(positions, velocities, times)
    p1 = plot(times, mean(positions, dims=2),
              ribbon=std(positions, dims=2),
              label="Position",
              xlabel="Time Step",
              ylabel="Position",
              title="Position Statistics")
              
    p2 = plot(times, mean(velocities, dims=2),
              ribbon=std(velocities, dims=2),
              label="Velocity",
              xlabel="Time Step",
              ylabel="Velocity",
              title="Velocity Statistics")
              
    return plot(p1, p2, layout=(2,1))
end

"""
    create_simulation_animation(positions, velocities, height_fn; fps=30)

Create an animated GIF of the mountain car simulation.
"""
function create_simulation_animation(positions, velocities, height_fn; fps=30, trace_color=:blue, title_prefix="Simulation")
    anim = @animate for i in 1:length(positions)
        # Plot landscape
        p = plot_landscape(height_fn)
        
        # Add historical trace
        if i > 1
            plot!(positions[1:i], height_fn.(positions[1:i]),
                  label="Path",
                  color=trace_color,
                  alpha=0.3,
                  linewidth=2)
        end
        
        # Add car position
        scatter!([positions[i]], [height_fn(positions[i])],
                label="Car",
                color=trace_color,
                markersize=6)
        
        # Add velocity vector
        quiver!([positions[i]], [height_fn(positions[i])],
                quiver=([velocities[i]*0.1], [velocities[i]*0.1*derivative(height_fn, positions[i])]),
                color=trace_color,
                label="Velocity")
                
        title!("$(title_prefix) (t=$(i))")
    end
    
    return gif(anim, "simulation.gif", fps=fps)
end

"""
    derivative(f, x; h=1e-5)

Compute numerical derivative of function f at point x.
"""
function derivative(f, x; h=1e-5)
    return (f(x + h) - f(x - h)) / (2h)
end

"""
    plot_simulation_summary(positions, velocities, height_fn, times)

Create a comprehensive summary plot of the simulation including:
- Landscape with final position
- State distribution
- Statistical trajectories
- Control actions if provided
"""
function plot_simulation_summary(positions, velocities, height_fn, times; actions=nothing)
    # Create a 3-panel plot layout
    p = plot(layout=grid(3,1), size=(800, 900))
    
    # Position vs Time
    plot!(p[1], times, positions, 
        label="Position", 
        color=:blue,
        title="Position vs Time",
        xlabel="Time Step",
        ylabel="Position")
    
    # Velocity vs Time  
    plot!(p[2], times, velocities,
        label="Velocity",
        color=:red, 
        title="Velocity vs Time",
        xlabel="Time Step",
        ylabel="Velocity")
    
    # Actions vs Time (if provided)
    if !isnothing(actions)
        plot!(p[3], times, actions,
            label="Actions",
            color=:purple,
            title="Actions vs Time", 
            xlabel="Time Step",
            ylabel="Force")
    end
    
    return p
end

"""
    create_comparison_animation(naive_positions, naive_velocities, ai_positions, ai_velocities, height_fn; fps=30)

Create an animated GIF of the comparison between naive and AI approaches.
"""
function create_comparison_animation(naive_positions, naive_velocities, ai_positions, ai_velocities, height_fn; fps=30)
    anim = @animate for i in 1:length(naive_positions)
        # Plot landscape
        p = plot_landscape(height_fn)
        
        # Add naive approach trace
        if i > 1
            plot!(naive_positions[1:i], height_fn.(naive_positions[1:i]),
                  label="Naive Path",
                  color=:red,
                  alpha=0.3,
                  linewidth=2)
            
            plot!(ai_positions[1:i], height_fn.(ai_positions[1:i]),
                  label="AI Path",
                  color=:blue,
                  alpha=0.3,
                  linewidth=2)
        end
        
        # Add naive car position
        scatter!([naive_positions[i]], [height_fn(naive_positions[i])],
                label="Naive Car",
                color=:red,
                markersize=6)
                
        # Add AI car position
        scatter!([ai_positions[i]], [height_fn(ai_positions[i])],
                label="AI Car",
                color=:blue,
                markersize=6)
        
        title!("Comparison: Naive vs Active Inference (t=$(i))")
    end
    
    return gif(anim, "comparison.gif", fps=fps)
end

export plot_landscape, plot_state_distribution, plot_trajectory_statistics,
       create_simulation_animation, plot_simulation_summary, create_comparison_animation 