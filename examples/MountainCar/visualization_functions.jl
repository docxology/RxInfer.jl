"""
    visualization_functions.jl

Contains all visualization functions for the mountain car meta-analysis.
"""

using UnicodePlots
using Statistics
using Printf
using DataFrames

export plot_success_rate_comparison, plot_performance_metrics, plot_energy_comparison,
       plot_control_comparison, plot_parameter_sweep_analysis, plot_trajectory_analysis

"""
    plot_success_rate_comparison(results::DataFrame, output_path::String)

Create heatmaps comparing success rates between naive and active inference agents
across different force and friction values.
"""
function plot_success_rate_comparison(results::DataFrame, output_path::String)
    # Create plots for each agent type
    for agent in ["naive", "active"]
        # Filter results for current agent
        agent_results = filter(r -> r.agent_type == agent, results)
        
        # Calculate average success rate for each force-friction combination
        success_rates = combine(
            groupby(agent_results, [:force, :friction]),
            :success => mean => :success_rate
        )
        
        # Create heatmap using Unicode plots
        forces = sort(unique(success_rates.force))
        frictions = sort(unique(success_rates.friction))
        
        success_matrix = zeros(length(forces), length(frictions))
        for (i, f) in enumerate(forces)
            for (j, μ) in enumerate(frictions)
                row = filter(r -> r.force == f && r.friction == μ, success_rates)
                if !isempty(row)
                    success_matrix[i, j] = row.success_rate[1]
                end
            end
        end
        
        # Create and save heatmap
        p = heatmap(
            success_matrix,
            title="$(titlecase(agent)) Agent Success Rate",
            xlabel="Force",
            ylabel="Friction",
            xfact=minimum(forces),  # Scale factor for x-axis
            yfact=minimum(frictions),  # Scale factor for y-axis
            width=50,  # Adjust width for better visualization
            height=15,  # Adjust height for better visualization
            colormap=:viridis
        )
        
        # Save plot and data
        open(joinpath(dirname(output_path), "success_rate_$(agent).txt"), "w") do io
            show(io, p)
            println(io, "\n\nSuccess Rate Analysis for $(titlecase(agent)) Agent")
            println(io, "=" ^ 50)
            
            # Print parameter ranges with better formatting
            println(io, "\nForce values (x-axis):")
            println(io, "-" ^ 20)
            for (i, f) in enumerate(forces)
                println(io, "  Index $(@sprintf("%2d", i)): $(@sprintf("%.4f", f))")
            end
            
            println(io, "\nFriction values (y-axis):")
            println(io, "-" ^ 20)
            for (i, f) in enumerate(frictions)
                println(io, "  Index $(@sprintf("%2d", i)): $(@sprintf("%.4f", f))")
            end
            
            # Print success rate matrix with better formatting
            println(io, "\nSuccess Rate Matrix (rows=force, cols=friction):")
            println(io, "-" ^ 50)
            
            # Print header with friction indices
            print(io, "F\\μ |")
            for j in 1:length(frictions)
                @printf(io, " %4d |", j)
            end
            println(io)
            println(io, "-" ^ (7 + 7 * length(frictions)))
            
            # Print matrix with force indices and values
            for i in 1:size(success_matrix, 1)
                @printf(io, " %3d |", i)
                for j in 1:size(success_matrix, 2)
                    @printf(io, " %4.1f |", 100 * success_matrix[i,j])
                end
                println(io)
            end
            
            # Print summary statistics
            println(io, "\nSummary Statistics:")
            println(io, "-" ^ 20)
            println(io, "Overall success rate: $(@sprintf("%.1f%%", 100 * mean(success_matrix)))")
            println(io, "Best success rate: $(@sprintf("%.1f%%", 100 * maximum(success_matrix)))")
            println(io, "Worst success rate: $(@sprintf("%.1f%%", 100 * minimum(success_matrix)))")
            
            # Find best parameter combination
            best_i, best_j = argmax(success_matrix).I
            println(io, "\nBest parameter combination:")
            println(io, "  Force: $(@sprintf("%.4f", forces[best_i])) (index $best_i)")
            println(io, "  Friction: $(@sprintf("%.4f", frictions[best_j])) (index $best_j)")
            println(io, "  Success rate: $(@sprintf("%.1f%%", 100 * success_matrix[best_i, best_j]))")
            
            # Add parameter space insights
            println(io, "\nParameter Space Analysis:")
            println(io, "-" ^ 20)
            
            # Force analysis
            force_means = vec(mean(success_matrix, dims=2))
            best_force_idx = argmax(force_means)
            println(io, "Force effectiveness (averaged over all friction values):")
            println(io, "  Best force: $(@sprintf("%.4f", forces[best_force_idx]))")
            println(io, "  Average success: $(@sprintf("%.1f%%", 100 * force_means[best_force_idx]))")
            
            # Friction analysis
            friction_means = vec(mean(success_matrix, dims=1))
            best_friction_idx = argmax(friction_means)
            println(io, "\nFriction effectiveness (averaged over all force values):")
            println(io, "  Best friction: $(@sprintf("%.4f", frictions[best_friction_idx]))")
            println(io, "  Average success: $(@sprintf("%.1f%%", 100 * friction_means[best_friction_idx]))")
        end
    end
end

"""
    plot_performance_metrics(results::DataFrame, output_path::String)

Create plots comparing key performance metrics between agent types.
"""
function plot_performance_metrics(results::DataFrame, output_path::String)
    # Calculate summary statistics for each agent type
    metrics = combine(
        groupby(results, :agent_type),
        :success => mean => :success_rate,
        :target_time => (x -> mean(filter(isfinite, x))) => :avg_time,
        :total_energy => mean => :avg_energy,
        :efficiency => mean => :avg_efficiency,
        :stability => mean => :avg_stability,
        :oscillations => mean => :avg_oscillations,
        :control_effort => mean => :avg_control_effort
    )
    
    # Create and save plots
    open(joinpath(dirname(output_path), "performance_metrics.txt"), "w") do io
        println(io, "=== Performance Metrics Analysis ===\n")
        
        # Create plots for each metric
        metrics_to_plot = [
            (:success_rate, "Success Rate", true),
            (:avg_time, "Average Time to Target", false),
            (:avg_energy, "Average Energy Usage", false),
            (:efficiency, "Efficiency", true),
            (:stability, "Stability", true),
            (:oscillations, "Oscillations", false),
            (:control_effort, "Control Effort", false)
        ]
        
        for (metric, title, higher_is_better) in metrics_to_plot
            values = metrics[:, metric]
            p = lineplot(1:nrow(metrics), values,
                title=title,
                xlabel="Agent Type",
                ylabel=title,
                labels=metrics.agent_type
            )
            show(io, p)
            println(io, "\n")
            
            # Print comparison
            println(io, "\n$title Comparison:")
            for (agent, value) in zip(metrics.agent_type, values)
                println(io, "  $agent: $(@sprintf("%.3f", value))")
            end
            
            # Calculate and print improvement
            if length(values) == 2
                diff_pct = abs(values[2] - values[1]) / abs(values[1]) * 100
                better = if higher_is_better
                    values[2] > values[1] ? "Active" : "Naive"
                else
                    values[2] < values[1] ? "Active" : "Naive"
                end
                println(io, "  Improvement: $(@sprintf("%.1f%%", diff_pct)) ($better agent performs better)")
            end
            println(io)
        end
    end
end

"""
    plot_energy_comparison(results::DataFrame, output_path::String)

Create plots comparing energy usage between agent types.
"""
function plot_energy_comparison(results::DataFrame, output_path::String)
    open(joinpath(dirname(output_path), "energy_analysis.txt"), "w") do io
        println(io, "=== Energy Usage Analysis ===\n")
        
        # Overall energy comparison
        energy_by_agent = combine(
            groupby(results, :agent_type),
            :total_energy => mean => :avg_energy,
            :total_energy => std => :std_energy,
            :efficiency => mean => :avg_efficiency
        )
        
        # Create energy comparison plot
        p = lineplot(1:nrow(energy_by_agent), energy_by_agent.avg_energy,
            title="Average Energy Usage",
            xlabel="Agent Type",
            ylabel="Energy",
            labels=energy_by_agent.agent_type
        )
        show(io, p)
        println(io, "\n")
        
        # Print detailed statistics
        println(io, "\nEnergy Statistics:")
        for row in eachrow(energy_by_agent)
            println(io, "\n$(row.agent_type) Agent:")
            println(io, "  Average Energy: $(@sprintf("%.3f", row.avg_energy))")
            println(io, "  Energy Std Dev: $(@sprintf("%.3f", row.std_energy))")
            println(io, "  Average Efficiency: $(@sprintf("%.3f", row.avg_efficiency))")
        end
        
        # Energy vs Success analysis
        println(io, "\nEnergy vs Success Correlation:")
        for agent in unique(results.agent_type)
            agent_results = filter(r -> r.agent_type == agent, results)
            correlation = cor(agent_results.total_energy, Float64.(agent_results.success))
            println(io, "  $agent: $(@sprintf("%.3f", correlation))")
        end
    end
end

"""
    plot_control_comparison(results::DataFrame, output_path::String)

Create plots comparing control strategies between agent types.
"""
function plot_control_comparison(results::DataFrame, output_path::String)
    open(joinpath(dirname(output_path), "control_analysis.txt"), "w") do io
        println(io, "=== Control Strategy Analysis ===\n")
        
        # Control effort comparison
        control_by_agent = combine(
            groupby(results, :agent_type),
            :control_effort => mean => :avg_effort,
            :control_effort => std => :std_effort,
            :oscillations => mean => :avg_oscillations
        )
        
        # Create control effort plot
        p = lineplot(1:nrow(control_by_agent), control_by_agent.avg_effort,
            title="Average Control Effort",
            xlabel="Agent Type",
            ylabel="Control Effort",
            labels=control_by_agent.agent_type
        )
        show(io, p)
        println(io, "\n")
        
        # Print detailed statistics
        println(io, "\nControl Statistics:")
        for row in eachrow(control_by_agent)
            println(io, "\n$(row.agent_type) Agent:")
            println(io, "  Average Control Effort: $(@sprintf("%.3f", row.avg_effort))")
            println(io, "  Control Effort Std Dev: $(@sprintf("%.3f", row.std_effort))")
            println(io, "  Average Oscillations: $(@sprintf("%.3f", row.avg_oscillations))")
        end
    end
end

"""
    plot_parameter_sweep_analysis(results::DataFrame, output_path::String)

Create plots analyzing the effect of different parameters on performance.
"""
function plot_parameter_sweep_analysis(results::DataFrame, output_path::String)
    open(joinpath(dirname(output_path), "parameter_analysis.txt"), "w") do io
        println(io, "=== Parameter Sweep Analysis ===\n")
        
        # Analyze success rate vs parameters
        for param in [:force, :friction]
            println(io, "\nAnalysis for $(param):")
            
            for agent in unique(results.agent_type)
                agent_results = filter(r -> r.agent_type == agent, results)
                param_success = combine(
                    groupby(agent_results, param),
                    :success => mean => :success_rate
                )
                
                # Create parameter effect plot
                p = lineplot(param_success[:, param], param_success.success_rate,
                    title="$(titlecase(string(param))) Effect on Success ($agent)",
                    xlabel=string(param),
                    ylabel="Success Rate"
                )
                show(io, p)
                println(io, "\n")
                
                # Find optimal parameter value
                optimal_idx = argmax(param_success.success_rate)
                optimal_value = param_success[optimal_idx, param]
                optimal_success = param_success[optimal_idx, :success_rate]
                
                println(io, "$(agent) Agent:")
                println(io, "  Optimal $(param): $(@sprintf("%.3f", optimal_value))")
                println(io, "  Success rate at optimum: $(@sprintf("%.3f", optimal_success))")
                println(io)
            end
        end
    end
end

"""
    plot_trajectory_analysis(results::DataFrame, output_path::String)

Create plots analyzing the trajectories of successful runs.
"""
function plot_trajectory_analysis(results::DataFrame, output_path::String)
    open(joinpath(dirname(output_path), "trajectory_analysis.txt"), "w") do io
        println(io, "=== Trajectory Analysis ===\n")
        
        # Analyze successful runs
        successful = filter(r -> r.success, results)
        
        if !isempty(successful)
            # Analyze completion time distribution
            time_stats = combine(
                groupby(successful, :agent_type),
                :target_time => mean => :avg_time,
                :target_time => std => :std_time,
                :target_time => minimum => :min_time,
                :target_time => maximum => :max_time
            )
            
            # Create completion time plot
            p = lineplot(1:nrow(time_stats), time_stats.avg_time,
                title="Average Completion Time (Successful Runs)",
                xlabel="Agent Type",
                ylabel="Time Steps",
                labels=time_stats.agent_type
            )
            show(io, p)
            println(io, "\n")
            
            # Print detailed statistics
            println(io, "\nCompletion Time Statistics (Successful Runs):")
            for row in eachrow(time_stats)
                println(io, "\n$(row.agent_type) Agent:")
                println(io, "  Average Time: $(@sprintf("%.3f", row.avg_time))")
                println(io, "  Time Std Dev: $(@sprintf("%.3f", row.std_time))")
                println(io, "  Minimum Time: $(@sprintf("%.3f", row.min_time))")
                println(io, "  Maximum Time: $(@sprintf("%.3f", row.max_time))")
            end
        else
            println(io, "No successful runs found in the results.")
        end
    end
end 