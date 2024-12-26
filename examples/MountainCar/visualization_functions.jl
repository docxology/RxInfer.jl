"""
    visualization_functions.jl

Contains all visualization functions for the mountain car meta-analysis.
"""

using UnicodePlots
using Statistics
using Printf
using DataFrames
using Dates

export plot_success_rate_comparison, plot_performance_metrics, plot_energy_comparison,
       plot_control_comparison, plot_parameter_sweep_analysis, plot_trajectory_analysis

"""
Helper functions for safe DataFrame operations and statistics
"""

function get_available_metrics(df::DataFrame)
    metrics = Symbol[]
    for metric in [:success, :target_time, :total_energy, :efficiency, :stability, :oscillations, :control_effort]
        if metric in propertynames(df)
            push!(metrics, metric)
        end
    end
    return metrics
end

function safe_mean(v::AbstractVector)
    try
        if isempty(v)
            return 0.0
        end
        return mean(v)
    catch
        return 0.0
    end
end

function safe_combine(df::DataFrame, groupcols::Vector{Symbol}, aggpairs::Vector{Pair})
    try
        return combine(groupby(df, groupcols), aggpairs...)
    catch
        @warn "Error in safe_combine operation"
        return DataFrame()
    end
end

# Helper function to safely compute metrics
function safe_mean(x)
    try
        isempty(x) ? 0.0 : mean(filter(!isnan, x))
    catch
        0.0
    end
end

function safe_std(x)
    try
        isempty(x) ? 0.0 : std(filter(!isnan, x))
    catch
        0.0
    end
end

function safe_maximum(x)
    try
        isempty(x) ? 0.0 : maximum(filter(!isnan, x))
    catch
        0.0
    end
end

function safe_minimum(x)
    try
        isempty(x) ? 0.0 : minimum(filter(!isnan, x))
    catch
        0.0
    end
end

function safe_cor(x, y)
    try
        cor(Float64.(x), Float64.(y))
    catch
        0.0
    end
end

"""
    plot_success_rate_comparison(results::DataFrame, output_path::String)

Create heatmaps comparing success rates between naive and active inference agents.
"""
function plot_success_rate_comparison(results::DataFrame, output_path::String)
    try
        # Check if we have the necessary columns
        required_cols = [:agent_type, :force, :friction, :success]
        if !all(col in propertynames(results) for col in required_cols)
            @warn "Missing required columns for success rate comparison"
            return
        end
        
        # Get unique agent types
        agent_types = unique(results.agent_type)
        if length(agent_types) != 2
            @warn "Expected exactly 2 agent types for comparison, got $(length(agent_types))"
            return
        end
        
        # Get parameter ranges
        force_range = sort(unique(results.force))
        friction_range = sort(unique(results.friction))
        
        # Initialize success rate matrices
        success_matrices = Dict{String, Matrix{Float64}}()
        for agent in agent_types
            success_matrix = zeros(length(force_range), length(friction_range))
            agent_results = filter(r -> r.agent_type == agent, results)
            
            # Calculate success rates for each parameter combination
            for (i, force) in enumerate(force_range)
                for (j, friction) in enumerate(friction_range)
                    subset = filter(r -> r.force ≈ force && r.friction ≈ friction, agent_results)
                    success_matrix[i, j] = safe_mean(subset.success)
                end
            end
            success_matrices[agent] = success_matrix
        end
        
        # Create output directory if it doesn't exist
        mkpath(dirname(output_path))
        
        # Generate comparison plots and save to file
        open(joinpath(dirname(output_path), "success_rate_comparison.txt"), "w") do io
            println(io, "=== Success Rate Comparison Analysis ===\n")
            println(io, "Analysis timestamp: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
            println(io, "=" ^ 50, "\n")
            
            # Plot individual heatmaps
            for agent in agent_types
                println(io, "\n$(titlecase(agent)) Agent Success Rate Heatmap")
                println(io, "-" ^ (length(agent) + 27), "\n")
                
                try
                    # Create heatmap using Unicode plots
                    p = heatmap(
                        collect(1:length(force_range)),
                        collect(1:length(friction_range)),
                        success_matrices[agent]',
                        title="$(titlecase(agent)) Agent Success Rate",
                        xlabel="Force Index",
                        ylabel="Friction Index",
                        colormap=:viridis
                    )
                    show(io, p)
                    
                    # Print parameter mappings
                    println(io, "\n\nForce Index Mapping:")
                    for (i, force) in enumerate(force_range)
                        println(io, "  $i => $(@sprintf("%.4f", force))")
                    end
                    
                    println(io, "\nFriction Index Mapping:")
                    for (i, friction) in enumerate(friction_range)
                        println(io, "  $i => $(@sprintf("%.4f", friction))")
                    end
                    
                    # Calculate and print statistics
                    success_rates = vec(success_matrices[agent])
                    println(io, "\nSuccess Rate Statistics:")
                    println(io, "  Mean: $(@sprintf("%.2f%%", 100 * mean(success_rates)))")
                    println(io, "  Max:  $(@sprintf("%.2f%%", 100 * maximum(success_rates)))")
                    println(io, "  Min:  $(@sprintf("%.2f%%", 100 * minimum(success_rates)))")
                    println(io, "  Std:  $(@sprintf("%.2f%%", 100 * std(success_rates)))")
                    
                    # Find best parameter combination
                    max_idx = argmax(success_matrices[agent])
                    best_force = force_range[max_idx[1]]
                    best_friction = friction_range[max_idx[2]]
                    max_success = success_matrices[agent][max_idx]
                    
                    println(io, "\nBest Parameter Combination:")
                    println(io, "  Force: $(@sprintf("%.4f", best_force))")
                    println(io, "  Friction: $(@sprintf("%.4f", best_friction))")
                    println(io, "  Success Rate: $(@sprintf("%.2f%%", 100 * max_success))")
                catch e
                    @warn "Error creating heatmap for $agent: $e"
                    println(io, "\nError creating heatmap: $e")
                end
                
                println(io, "\n", "=" ^ 50)
            end
            
            # Calculate and plot success rate difference
            try
                diff_matrix = success_matrices[agent_types[2]] - success_matrices[agent_types[1]]
                println(io, "\nSuccess Rate Difference Heatmap")
                println(io, "($(titlecase(agent_types[2])) - $(titlecase(agent_types[1])))")
                println(io, "-" ^ 50, "\n")
                
                p = heatmap(
                    collect(1:length(force_range)),
                    collect(1:length(friction_range)),
                    diff_matrix',
                    title="Success Rate Difference",
                    xlabel="Force Index",
                    ylabel="Friction Index",
                    colormap=:viridis
                )
                show(io, p)
                
                # Print difference statistics
                diff_values = vec(diff_matrix)
                println(io, "\n\nDifference Statistics:")
                println(io, "  Mean: $(@sprintf("%.2f%%", 100 * mean(diff_values)))")
                println(io, "  Max:  $(@sprintf("%.2f%%", 100 * maximum(diff_values)))")
                println(io, "  Min:  $(@sprintf("%.2f%%", 100 * minimum(diff_values)))")
                println(io, "  Std:  $(@sprintf("%.2f%%", 100 * std(diff_values)))")
                
                # Find regions of largest improvement
                max_improvement_idx = argmax(diff_matrix)
                max_improvement = diff_matrix[max_improvement_idx]
                
                println(io, "\nLargest Improvement Region:")
                println(io, "  Force: $(@sprintf("%.4f", force_range[max_improvement_idx[1]]))")
                println(io, "  Friction: $(@sprintf("%.4f", friction_range[max_improvement_idx[2]]))")
                println(io, "  Improvement: $(@sprintf("%.2f%%", 100 * max_improvement))")
            catch e
                @warn "Error creating difference heatmap: $e"
                println(io, "\nError creating difference heatmap: $e")
            end
        end
    catch e
        @error "Failed to generate success rate comparison" exception=(e, catch_backtrace())
    end
end

"""
    plot_performance_metrics(results::DataFrame, output_path::String)

Create plots comparing key performance metrics between agent types.
"""
function plot_performance_metrics(results::DataFrame, output_path::String)
    try
        # Get available metrics
        available_metrics = get_available_metrics(results)
        
        if isempty(available_metrics)
            @warn "No metrics available for performance analysis"
            return
        end
        
        # Calculate summary statistics for each agent type
        metrics_dict = Dict{Symbol, Vector{Float64}}()
        agent_types = unique(results.agent_type)
        
        for metric in available_metrics
            metrics_dict[metric] = Float64[]
            for agent in agent_types
                agent_results = filter(r -> r.agent_type == agent, results)
                if metric == :target_time
                    # Special handling for target_time (filter out infinite values)
                    push!(metrics_dict[metric], safe_mean(filter(isfinite, agent_results[!, metric])))
                else
                    push!(metrics_dict[metric], safe_mean(agent_results[!, metric]))
                end
            end
        end
        
        # Create and save plots
        open(joinpath(dirname(output_path), "performance_metrics.txt"), "w") do io
            println(io, "=== Performance Metrics Analysis ===\n")
            println(io, "Analysis timestamp: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
            println(io, "=" ^ 50, "\n")
            
            # Metrics descriptions
            metrics_info = Dict(
                :success => ("Success Rate", true, "Percentage of successful runs"),
                :target_time => ("Time to Target", false, "Average steps to reach target"),
                :total_energy => ("Energy Usage", false, "Average energy consumption"),
                :efficiency => ("Efficiency", true, "Success rate per energy unit"),
                :stability => ("Stability", true, "Inverse of position variance"),
                :oscillations => ("Oscillations", false, "Number of direction changes"),
                :control_effort => ("Control Effort", false, "Total absolute control input")
            )
            
            # Plot available metrics
            for metric in available_metrics
                title, higher_is_better, description = get(metrics_info, metric, 
                    (string(metric), true, "No description available"))
                
                values = metrics_dict[metric]
                
                try
                    # Create plot with proper labels
                    p = lineplot(collect(1:length(agent_types)), values,
                        title=title,
                        xlabel="Agent Type",
                        ylabel=title,
                        name=join(agent_types, " vs ")
                    )
                    
                    # Print section header
                    println(io, "\n", title, " Analysis")
                    println(io, "-" ^ (length(title) + 9))
                    println(io, "Description: ", description, "\n")
                    
                    # Show plot
                    show(io, p)
                    println(io, "\n")
                    
                    # Print detailed statistics
                    println(io, "Detailed Statistics:")
                    for (agent, value) in zip(agent_types, values)
                        println(io, "  $agent: $(@sprintf("%.3f", value))")
                    end
                    
                    # Calculate and print improvement if we have exactly 2 agents
                    if length(values) == 2 && !iszero(values[1])
                        diff_pct = abs(values[2] - values[1]) / abs(values[1]) * 100
                        better = if higher_is_better
                            values[2] > values[1] ? "Active" : "Naive"
                        else
                            values[2] < values[1] ? "Active" : "Naive"
                        end
                        println(io, "\nComparison:")
                        println(io, "  Difference: $(@sprintf("%.1f%%", diff_pct))")
                        println(io, "  Better agent: $better")
                        println(io, "  Interpretation: $(better) agent $(higher_is_better ? "achieved higher" : "required lower") $(lowercase(title))")
                    end
                    println(io, "\n", "-" ^ 50)
                catch e
                    @warn "Error plotting metric $metric: $e"
                    println(io, "\nError plotting $title: $e\n")
                end
            end
            
            # Add overall performance summary
            try
                println(io, "\nOverall Performance Summary")
                println(io, "=" ^ 25)
                
                for agent in agent_types
                    agent_metrics = filter(r -> r.agent_type == agent, results)
                    println(io, "\n$(titlecase(agent)) Agent Overall Performance:")
                    
                    # Print available metrics
                    if :success in available_metrics
                        println(io, "  Success Rate: $(@sprintf("%.1f%%", 100 * safe_mean(agent_metrics.success)))")
                    end
                    if :target_time in available_metrics
                        println(io, "  Average Steps (successful runs): $(@sprintf("%.1f", safe_mean(filter(isfinite, agent_metrics.target_time))))")
                    end
                    if :efficiency in available_metrics
                        println(io, "  Energy Efficiency: $(@sprintf("%.3f", safe_mean(agent_metrics.efficiency)))")
                    end
                    if :stability in available_metrics
                        println(io, "  Control Stability: $(@sprintf("%.3f", safe_mean(agent_metrics.stability)))")
                    end
                    
                    # Calculate parameter effectiveness if available
                    if :force in propertynames(results) && :success in available_metrics
                        force_success = safe_combine(agent_metrics, [:force], [:success => mean => :success_rate])
                        if !isempty(force_success)
                            best_force = force_success[argmax(force_success.success_rate), :force]
                            println(io, "  Best Force: $(@sprintf("%.4f", best_force))")
                        end
                    end
                    
                    if :friction in propertynames(results) && :success in available_metrics
                        friction_success = safe_combine(agent_metrics, [:friction], [:success => mean => :success_rate])
                        if !isempty(friction_success)
                            best_friction = friction_success[argmax(friction_success.success_rate), :friction]
                            println(io, "  Best Friction: $(@sprintf("%.4f", best_friction))")
                        end
                    end
                end
            catch e
                @warn "Error generating performance summary: $e"
                println(io, "\nError generating performance summary: $e")
            end
        end
    catch e
        @error "Failed to generate performance metrics analysis" exception=(e, catch_backtrace())
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