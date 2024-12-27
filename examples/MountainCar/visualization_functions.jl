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

function safe_combine_with_default(df::DataFrame, groupcols::Vector{Symbol}, aggpairs::Vector{<:Pair}, default_value::Any=0.0)
    try
        return combine(groupby(df, groupcols), aggpairs...)
    catch e
        @warn "Error in safe_combine operation" exception=e
        # Create an empty DataFrame with the expected structure
        result = DataFrame()
        # Add group columns
        for col in groupcols
            result[!, col] = unique(df[!, col])
        end
        # Add aggregated columns with default values
        for pair in aggpairs
            col_name = if pair.second isa Pair
                pair.second.second
            else
                Symbol(string(pair.first) * "_" * string(pair.second))
            end
            result[!, col_name] .= default_value
        end
        return result
    end
end

function safe_filter(df::DataFrame, condition::Function)
    try
        return filter(condition, df)
    catch e
        @warn "Error in safe_filter operation" exception=e
        return DataFrame()
    end
end

function safe_unique(v::AbstractVector)
    try
        return sort(unique(v))
    catch e
        @warn "Error in safe_unique operation" exception=e
        return eltype(v)[]
    end
end

function safe_statistics(v::AbstractVector{T}) where T
    stats = Dict{Symbol, Union{T, Float64}}()
    try
        filtered_v = filter(!isnan, v)
        if !isempty(filtered_v)
            stats[:mean] = mean(filtered_v)
            stats[:std] = std(filtered_v)
            stats[:min] = minimum(filtered_v)
            stats[:max] = maximum(filtered_v)
            stats[:median] = median(filtered_v)
        else
            stats[:mean] = zero(T)
            stats[:std] = zero(T)
            stats[:min] = zero(T)
            stats[:max] = zero(T)
            stats[:median] = zero(T)
        end
    catch e
        @warn "Error calculating statistics" exception=e
        stats[:mean] = zero(T)
        stats[:std] = zero(T)
        stats[:min] = zero(T)
        stats[:max] = zero(T)
        stats[:median] = zero(T)
    end
    return stats
end

function safe_correlation(x::AbstractVector, y::AbstractVector)
    try
        if length(x) == length(y) && !isempty(x)
            filtered_indices = .!isnan.(x) .& .!isnan.(y)
            if any(filtered_indices)
                return cor(x[filtered_indices], y[filtered_indices])
            end
        end
        return 0.0
    catch e
        @warn "Error calculating correlation" exception=e
        return 0.0
    end
end

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
        filtered_v = filter(!isnan, v)
        if isempty(filtered_v)
            return 0.0
        end
        return mean(filtered_v)
    catch
        return 0.0
    end
end

function safe_std(v::AbstractVector)
    try
        filtered_v = filter(!isnan, v)
        if isempty(filtered_v)
            return 0.0
        end
        return std(filtered_v)
    catch
        return 0.0
    end
end

function safe_extrema(v::AbstractVector)
    try
        filtered_v = filter(!isnan, v)
        if isempty(filtered_v)
            return (0.0, 0.0)
        end
        return extrema(filtered_v)
    catch
        return (0.0, 0.0)
    end
end

"""
    create_safe_heatmap(x, y, z; title="", xlabel="", ylabel="")

Helper function to safely create a heatmap with proper keyword arguments.
"""
function create_safe_heatmap(x, y, z; title="", xlabel="", ylabel="")
    try
        # Create heatmap with just the matrix
        p = heatmap(z; title=title, xlabel=xlabel, ylabel=ylabel, 
                   xfact=minimum(x), yfact=minimum(y),
                   width=50, height=15)
        return p
    catch e
        @warn "Failed to create heatmap" exception=e
        return nothing
    end
end

"""
    create_safe_lineplot(x, y; title="", xlabel="", ylabel="", name="")

Helper function to safely create a lineplot with proper keyword arguments.
"""
function create_safe_lineplot(x, y; title="", xlabel="", ylabel="", name="")
    try
        # Create lineplot with basic arguments first
        p = lineplot(x, y; title=title, xlabel=xlabel, ylabel=ylabel)
        return p
    catch e
        @warn "Failed to create lineplot" exception=e
        return nothing
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
        agent_types = safe_unique(results.agent_type)
        if length(agent_types) != 2
            @warn "Expected exactly 2 agent types for comparison, got $(length(agent_types))"
            return
        end
        
        # Get parameter ranges
        force_range = safe_unique(results.force)
        friction_range = safe_unique(results.friction)
        
        # Initialize success rate matrices
        success_matrices = Dict{String, Matrix{Float64}}()
        for agent in agent_types
            success_matrix = zeros(length(force_range), length(friction_range))
            agent_results = safe_filter(results, r -> r.agent_type == agent)
            
            # Calculate success rates for each parameter combination
            for (i, force) in enumerate(force_range)
                for (j, friction) in enumerate(friction_range)
                    subset = safe_filter(agent_results, r -> r.force ≈ force && r.friction ≈ friction)
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
                    p = create_safe_heatmap(
                        collect(1:length(force_range)),
                        collect(1:length(friction_range)),
                        success_matrices[agent]';
                        title="$(titlecase(agent)) Agent Success Rate",
                        xlabel="Force Index",
                        ylabel="Friction Index"
                    )
                    
                    if !isnothing(p)
                        show(io, p)
                    else
                        println(io, "\nUnable to create heatmap visualization")
                    end
                    
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
                    stats = safe_statistics(success_rates)
                    println(io, "\nSuccess Rate Statistics:")
                    println(io, "  Mean: $(@sprintf("%.2f%%", 100 * stats[:mean]))")
                    println(io, "  Max:  $(@sprintf("%.2f%%", 100 * stats[:max]))")
                    println(io, "  Min:  $(@sprintf("%.2f%%", 100 * stats[:min]))")
                    println(io, "  Std:  $(@sprintf("%.2f%%", 100 * stats[:std]))")
                    
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
                
                p = create_safe_heatmap(
                    collect(1:length(force_range)),
                    collect(1:length(friction_range)),
                    diff_matrix';
                    title="Success Rate Difference",
                    xlabel="Force Index",
                    ylabel="Friction Index"
                )
                
                if !isnothing(p)
                    show(io, p)
                    
                    # Print difference statistics
                    diff_values = vec(diff_matrix)
                    stats = safe_statistics(diff_values)
                    println(io, "\n\nDifference Statistics:")
                    println(io, "  Mean: $(@sprintf("%.2f%%", 100 * stats[:mean]))")
                    println(io, "  Max:  $(@sprintf("%.2f%%", 100 * stats[:max]))")
                    println(io, "  Min:  $(@sprintf("%.2f%%", 100 * stats[:min]))")
                    println(io, "  Std:  $(@sprintf("%.2f%%", 100 * stats[:std]))")
                    
                    # Find regions of largest improvement
                    max_improvement_idx = argmax(diff_matrix)
                    max_improvement = diff_matrix[max_improvement_idx]
                    
                    println(io, "\nLargest Improvement Region:")
                    println(io, "  Force: $(@sprintf("%.4f", force_range[max_improvement_idx[1]]))")
                    println(io, "  Friction: $(@sprintf("%.4f", friction_range[max_improvement_idx[2]]))")
                    println(io, "  Improvement: $(@sprintf("%.2f%%", 100 * max_improvement))")
                else
                    println(io, "\nUnable to create difference heatmap")
                end
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
        agent_types = safe_unique(results.agent_type)
        
        for metric in available_metrics
            metrics_dict[metric] = Float64[]
            for agent in agent_types
                agent_results = safe_filter(results, r -> r.agent_type == agent)
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
                    p = create_safe_lineplot(
                        collect(1:length(agent_types)),
                        values;
                        title=title,
                        xlabel="Agent Type",
                        ylabel=title,
                        name=join(agent_types, " vs ")
                    )
                    
                    # Print section header
                    println(io, "\n", title, " Analysis")
                    println(io, "-" ^ (length(title) + 9))
                    println(io, "Description: ", description, "\n")
                    
                    if !isnothing(p)
                        show(io, p)
                        println(io, "\n")
                    else
                        println(io, "\nUnable to create performance plot")
                    end
                    
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
                    agent_results = safe_filter(results, r -> r.agent_type == agent)
                    println(io, "\n$(titlecase(agent)) Agent Overall Performance:")
                    
                    # Print available metrics
                    if :success in available_metrics
                        println(io, "  Success Rate: $(@sprintf("%.1f%%", 100 * safe_mean(agent_results.success)))")
                    end
                    if :target_time in available_metrics
                        println(io, "  Average Steps (successful runs): $(@sprintf("%.1f", safe_mean(filter(isfinite, agent_results.target_time))))")
                    end
                    if :efficiency in available_metrics
                        println(io, "  Energy Efficiency: $(@sprintf("%.3f", safe_mean(agent_results.efficiency)))")
                    end
                    if :stability in available_metrics
                        println(io, "  Control Stability: $(@sprintf("%.3f", safe_mean(agent_results.stability)))")
                    end
                    
                    # Calculate parameter effectiveness if available
                    if :force in propertynames(results) && :success in available_metrics
                        force_success = safe_combine_with_default(
                            agent_results,
                            [:force],
                            [:success => mean => :success_rate]
                        )
                        if !isempty(force_success)
                            best_force = force_success[argmax(force_success.success_rate), :force]
                            println(io, "  Best Force: $(@sprintf("%.4f", best_force))")
                        end
                    end
                    
                    if :friction in propertynames(results) && :success in available_metrics
                        friction_success = safe_combine_with_default(
                            agent_results,
                            [:friction],
                            [:success => mean => :success_rate]
                        )
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
    try
        # Check if we have the necessary columns
        required_cols = [:agent_type, :total_energy, :efficiency]
        if !all(col in propertynames(results) for col in required_cols)
            @warn "Missing required columns for energy comparison"
            return
        end
        
        open(joinpath(dirname(output_path), "energy_analysis.txt"), "w") do io
            println(io, "=== Energy Usage Analysis ===\n")
            println(io, "Analysis timestamp: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
            println(io, "=" ^ 50, "\n")
            
            # Overall energy comparison
            energy_by_agent = safe_combine_with_default(
                results,
                [:agent_type],
                [
                    :total_energy => mean => :avg_energy,
                    :total_energy => std => :std_energy,
                    :efficiency => mean => :avg_efficiency
                ]
            )
            
            # Create energy comparison plot
            p = create_safe_lineplot(
                1:nrow(energy_by_agent),
                energy_by_agent.avg_energy;
                title="Average Energy Usage",
                xlabel="Agent Type",
                ylabel="Energy",
                name=join(energy_by_agent.agent_type, " vs ")
            )
            
            if !isnothing(p)
                show(io, p)
                println(io, "\n")
            else
                println(io, "\nUnable to create energy comparison plot")
            end
            
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
            for agent in safe_unique(results.agent_type)
                agent_results = safe_filter(results, r -> r.agent_type == agent)
                correlation = safe_correlation(agent_results.total_energy, Float64.(agent_results.success))
                println(io, "  $agent: $(@sprintf("%.3f", correlation))")
            end
            
            # Additional energy analysis
            println(io, "\nDetailed Energy Analysis:")
            for agent in safe_unique(results.agent_type)
                agent_results = safe_filter(results, r -> r.agent_type == agent)
                energy_stats = safe_statistics(agent_results.total_energy)
                
                println(io, "\n$(titlecase(agent)) Agent Energy Distribution:")
                println(io, "  Mean Energy: $(@sprintf("%.3f", energy_stats[:mean]))")
                println(io, "  Median Energy: $(@sprintf("%.3f", energy_stats[:median]))")
                println(io, "  Min Energy: $(@sprintf("%.3f", energy_stats[:min]))")
                println(io, "  Max Energy: $(@sprintf("%.3f", energy_stats[:max]))")
                println(io, "  Energy Std Dev: $(@sprintf("%.3f", energy_stats[:std]))")
                
                # Energy efficiency analysis
                efficiency_stats = safe_statistics(agent_results.efficiency)
                println(io, "\n$(titlecase(agent)) Agent Efficiency Metrics:")
                println(io, "  Mean Efficiency: $(@sprintf("%.3f", efficiency_stats[:mean]))")
                println(io, "  Median Efficiency: $(@sprintf("%.3f", efficiency_stats[:median]))")
                println(io, "  Min Efficiency: $(@sprintf("%.3f", efficiency_stats[:min]))")
                println(io, "  Max Efficiency: $(@sprintf("%.3f", efficiency_stats[:max]))")
                println(io, "  Efficiency Std Dev: $(@sprintf("%.3f", efficiency_stats[:std]))")
            end
            
            # Compare successful vs unsuccessful runs
            println(io, "\nEnergy Analysis by Success Status:")
            for agent in safe_unique(results.agent_type)
                agent_results = safe_filter(results, r -> r.agent_type == agent)
                successful = safe_filter(agent_results, r -> r.success)
                unsuccessful = safe_filter(agent_results, r -> !r.success)
                
                println(io, "\n$(titlecase(agent)) Agent:")
                if !isempty(successful)
                    successful_stats = safe_statistics(successful.total_energy)
                    println(io, "  Successful Runs:")
                    println(io, "    Mean Energy: $(@sprintf("%.3f", successful_stats[:mean]))")
                    println(io, "    Energy Std Dev: $(@sprintf("%.3f", successful_stats[:std]))")
                end
                
                if !isempty(unsuccessful)
                    unsuccessful_stats = safe_statistics(unsuccessful.total_energy)
                    println(io, "  Unsuccessful Runs:")
                    println(io, "    Mean Energy: $(@sprintf("%.3f", unsuccessful_stats[:mean]))")
                    println(io, "    Energy Std Dev: $(@sprintf("%.3f", unsuccessful_stats[:std]))")
                end
            end
        end
    catch e
        @error "Failed to generate energy comparison" exception=(e, catch_backtrace())
    end
end

"""
    plot_control_comparison(results::DataFrame, output_path::String)

Create plots comparing control strategies between agent types.
"""
function plot_control_comparison(results::DataFrame, output_path::String)
    try
        # Check if we have the necessary columns
        required_cols = [:agent_type, :control_effort, :oscillations]
        if !all(col in propertynames(results) for col in required_cols)
            @warn "Missing required columns for control comparison"
            return
        end
        
        open(joinpath(dirname(output_path), "control_analysis.txt"), "w") do io
            println(io, "=== Control Strategy Analysis ===\n")
            println(io, "Analysis timestamp: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
            println(io, "=" ^ 50, "\n")
            
            # Control effort comparison
            control_by_agent = safe_combine_with_default(
                results,
                [:agent_type],
                [
                    :control_effort => mean => :avg_effort,
                    :control_effort => std => :std_effort,
                    :oscillations => mean => :avg_oscillations
                ]
            )
            
            # Create control effort plot
            p = create_safe_lineplot(
                1:nrow(control_by_agent),
                control_by_agent.avg_effort;
                title="Average Control Effort",
                xlabel="Agent Type",
                ylabel="Control Effort",
                name=join(control_by_agent.agent_type, " vs ")
            )
            
            if !isnothing(p)
                show(io, p)
                println(io, "\n")
            else
                println(io, "\nUnable to create control effort plot")
            end
            
            # Print detailed statistics
            println(io, "\nControl Statistics:")
            for row in eachrow(control_by_agent)
                println(io, "\n$(row.agent_type) Agent:")
                println(io, "  Average Control Effort: $(@sprintf("%.3f", row.avg_effort))")
                println(io, "  Control Effort Std Dev: $(@sprintf("%.3f", row.std_effort))")
                println(io, "  Average Oscillations: $(@sprintf("%.3f", row.avg_oscillations))")
            end
            
            # Detailed control analysis by agent
            println(io, "\nDetailed Control Analysis:")
            for agent in safe_unique(results.agent_type)
                agent_results = safe_filter(results, r -> r.agent_type == agent)
                
                # Control effort statistics
                effort_stats = safe_statistics(agent_results.control_effort)
                println(io, "\n$(titlecase(agent)) Agent Control Effort Distribution:")
                println(io, "  Mean Effort: $(@sprintf("%.3f", effort_stats[:mean]))")
                println(io, "  Median Effort: $(@sprintf("%.3f", effort_stats[:median]))")
                println(io, "  Min Effort: $(@sprintf("%.3f", effort_stats[:min]))")
                println(io, "  Max Effort: $(@sprintf("%.3f", effort_stats[:max]))")
                println(io, "  Effort Std Dev: $(@sprintf("%.3f", effort_stats[:std]))")
                
                # Oscillation statistics
                oscillation_stats = safe_statistics(agent_results.oscillations)
                println(io, "\n$(titlecase(agent)) Agent Oscillation Metrics:")
                println(io, "  Mean Oscillations: $(@sprintf("%.3f", oscillation_stats[:mean]))")
                println(io, "  Median Oscillations: $(@sprintf("%.3f", oscillation_stats[:median]))")
                println(io, "  Min Oscillations: $(@sprintf("%.3f", oscillation_stats[:min]))")
                println(io, "  Max Oscillations: $(@sprintf("%.3f", oscillation_stats[:max]))")
                println(io, "  Oscillations Std Dev: $(@sprintf("%.3f", oscillation_stats[:std]))")
            end
            
            # Compare control metrics between successful and unsuccessful runs
            println(io, "\nControl Analysis by Success Status:")
            for agent in safe_unique(results.agent_type)
                agent_results = safe_filter(results, r -> r.agent_type == agent)
                successful = safe_filter(agent_results, r -> r.success)
                unsuccessful = safe_filter(agent_results, r -> !r.success)
                
                println(io, "\n$(titlecase(agent)) Agent:")
                if !isempty(successful)
                    successful_effort = safe_statistics(successful.control_effort)
                    successful_osc = safe_statistics(successful.oscillations)
                    println(io, "  Successful Runs:")
                    println(io, "    Mean Control Effort: $(@sprintf("%.3f", successful_effort[:mean]))")
                    println(io, "    Control Effort Std Dev: $(@sprintf("%.3f", successful_effort[:std]))")
                    println(io, "    Mean Oscillations: $(@sprintf("%.3f", successful_osc[:mean]))")
                end
                
                if !isempty(unsuccessful)
                    unsuccessful_effort = safe_statistics(unsuccessful.control_effort)
                    unsuccessful_osc = safe_statistics(unsuccessful.oscillations)
                    println(io, "  Unsuccessful Runs:")
                    println(io, "    Mean Control Effort: $(@sprintf("%.3f", unsuccessful_effort[:mean]))")
                    println(io, "    Control Effort Std Dev: $(@sprintf("%.3f", unsuccessful_effort[:std]))")
                    println(io, "    Mean Oscillations: $(@sprintf("%.3f", unsuccessful_osc[:mean]))")
                end
            end
            
            # Correlation analysis
            println(io, "\nControl Metric Correlations:")
            for agent in safe_unique(results.agent_type)
                agent_results = safe_filter(results, r -> r.agent_type == agent)
                println(io, "\n$(titlecase(agent)) Agent:")
                
                # Correlation between control effort and success
                effort_success_cor = safe_correlation(agent_results.control_effort, Float64.(agent_results.success))
                println(io, "  Control Effort vs Success: $(@sprintf("%.3f", effort_success_cor))")
                
                # Correlation between oscillations and success
                osc_success_cor = safe_correlation(agent_results.oscillations, Float64.(agent_results.success))
                println(io, "  Oscillations vs Success: $(@sprintf("%.3f", osc_success_cor))")
                
                # Correlation between control effort and oscillations
                effort_osc_cor = safe_correlation(agent_results.control_effort, Float64.(agent_results.oscillations))
                println(io, "  Control Effort vs Oscillations: $(@sprintf("%.3f", effort_osc_cor))")
            end
        end
    catch e
        @error "Failed to generate control comparison" exception=(e, catch_backtrace())
    end
end

"""
    plot_parameter_sweep_analysis(results::DataFrame, output_path::String)

Create plots analyzing the effect of different parameters on performance.
"""
function plot_parameter_sweep_analysis(results::DataFrame, output_path::String)
    try
        # Check if we have the necessary columns
        required_cols = [:agent_type, :force, :friction, :success]
        if !all(col in propertynames(results) for col in required_cols)
            @warn "Missing required columns for parameter sweep analysis"
            return
        end
        
        open(joinpath(dirname(output_path), "parameter_analysis.txt"), "w") do io
            println(io, "=== Parameter Sweep Analysis ===\n")
            println(io, "Analysis timestamp: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
            println(io, "=" ^ 50, "\n")
            
            # Analyze success rate vs parameters
            for param in [:force, :friction]
                println(io, "\nAnalysis for $(param):")
                
                for agent in safe_unique(results.agent_type)
                    agent_results = safe_filter(results, r -> r.agent_type == agent)
                    param_success = safe_combine_with_default(
                        agent_results,
                        [param],
                        [:success => mean => :success_rate]
                    )
                    
                    # Create parameter effect plot
                    p = create_safe_lineplot(
                        param_success[:, param],
                        param_success.success_rate;
                        title="$(titlecase(string(param))) Effect on Success ($agent)",
                        xlabel=string(param),
                        ylabel="Success Rate"
                    )
                    
                    if !isnothing(p)
                        show(io, p)
                        println(io, "\n")
                    else
                        println(io, "\nUnable to create parameter effect plot")
                    end
                    
                    # Calculate and print statistics
                    println(io, "$(agent) Agent:")
                    stats = safe_statistics(param_success.success_rate)
                    println(io, "  Success Rate Statistics:")
                    println(io, "    Mean: $(@sprintf("%.3f", stats[:mean]))")
                    println(io, "    Median: $(@sprintf("%.3f", stats[:median]))")
                    println(io, "    Min: $(@sprintf("%.3f", stats[:min]))")
                    println(io, "    Max: $(@sprintf("%.3f", stats[:max]))")
                    println(io, "    Std Dev: $(@sprintf("%.3f", stats[:std]))")
                    
                    # Find optimal parameter value
                    if !isempty(param_success)
                        optimal_idx = argmax(param_success.success_rate)
                        optimal_value = param_success[optimal_idx, param]
                        optimal_success = param_success[optimal_idx, :success_rate]
                        
                        println(io, "  Optimal Configuration:")
                        println(io, "    Best $(param): $(@sprintf("%.3f", optimal_value))")
                        println(io, "    Success rate at optimum: $(@sprintf("%.3f", optimal_success))")
                    end
                    println(io)
                end
                
                # Cross-parameter analysis
                println(io, "\nCross-Parameter Analysis for $(param):")
                for agent in safe_unique(results.agent_type)
                    agent_results = safe_filter(results, r -> r.agent_type == agent)
                    
                    # Analyze success rate by parameter value and success status
                    println(io, "\n$(titlecase(agent)) Agent Parameter Effectiveness:")
                    
                    param_values = safe_unique(agent_results[:, param])
                    for value in param_values
                        value_results = safe_filter(agent_results, r -> r[param] ≈ value)
                        
                        # Calculate statistics for this parameter value
                        success_rate = safe_mean(value_results.success)
                        avg_time = safe_mean(filter(isfinite, value_results.target_time))
                        avg_energy = safe_mean(value_results.total_energy)
                        avg_control = safe_mean(value_results.control_effort)
                        
                        println(io, "\n  $(param) = $(@sprintf("%.3f", value)):")
                        println(io, "    Success Rate: $(@sprintf("%.3f", success_rate))")
                        println(io, "    Average Time: $(@sprintf("%.3f", avg_time))")
                        println(io, "    Average Energy: $(@sprintf("%.3f", avg_energy))")
                        println(io, "    Average Control Effort: $(@sprintf("%.3f", avg_control))")
                    end
                end
            end
            
            # Parameter interaction analysis
            println(io, "\nParameter Interaction Analysis:")
            for agent in safe_unique(results.agent_type)
                agent_results = safe_filter(results, r -> r.agent_type == agent)
                
                println(io, "\n$(titlecase(agent)) Agent:")
                
                # Calculate correlation between parameters and metrics
                metrics = [:success, :target_time, :total_energy, :control_effort]
                parameters = [:force, :friction]
                
                println(io, "\nParameter-Metric Correlations:")
                for param in parameters
                    for metric in metrics
                        if metric in propertynames(agent_results)
                            metric_values = metric == :target_time ? 
                                filter(isfinite, agent_results[:, metric]) :
                                agent_results[:, metric]
                            
                            correlation = safe_correlation(
                                agent_results[1:length(metric_values), param],
                                Float64.(metric_values)
                            )
                            println(io, "  $(titlecase(string(param))) vs $(titlecase(string(metric))): $(@sprintf("%.3f", correlation))")
                        end
                    end
                end
                
                # Parameter interaction effect on success
                println(io, "\nParameter Interaction Effect on Success:")
                interaction = safe_combine_with_default(
                    agent_results,
                    [:force, :friction],
                    [:success => mean => :success_rate]
                )
                
                if !isempty(interaction)
                    best_idx = argmax(interaction.success_rate)
                    println(io, "  Best Combined Parameters:")
                    println(io, "    Force: $(@sprintf("%.3f", interaction[best_idx, :force]))")
                    println(io, "    Friction: $(@sprintf("%.3f", interaction[best_idx, :friction]))")
                    println(io, "    Success Rate: $(@sprintf("%.3f", interaction[best_idx, :success_rate]))")
                end
            end
        end
    catch e
        @error "Failed to generate parameter sweep analysis" exception=(e, catch_backtrace())
    end
end

"""
    plot_trajectory_analysis(results::DataFrame, output_path::String)

Create plots analyzing the trajectories of successful runs.
"""
function plot_trajectory_analysis(results::DataFrame, output_path::String)
    try
        # Check if we have the necessary columns
        required_cols = [:agent_type, :success, :target_time, :max_position, :avg_velocity]
        if !all(col in propertynames(results) for col in required_cols)
            @warn "Missing required columns for trajectory analysis"
            return
        end
        
        open(joinpath(dirname(output_path), "trajectory_analysis.txt"), "w") do io
            println(io, "=== Trajectory Analysis ===\n")
            println(io, "Analysis timestamp: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
            println(io, "=" ^ 50, "\n")
            
            # Analyze successful runs
            successful = safe_filter(results, r -> r.success)
            
            if !isempty(successful)
                # Analyze completion time distribution
                time_stats = safe_combine_with_default(
                    successful,
                    [:agent_type],
                    [
                        :target_time => mean => :avg_time,
                        :target_time => std => :std_time,
                        :target_time => minimum => :min_time,
                        :target_time => maximum => :max_time
                    ]
                )
                
                # Create completion time plot
                p = create_safe_lineplot(
                    1:nrow(time_stats),
                    time_stats.avg_time;
                    title="Average Completion Time (Successful Runs)",
                    xlabel="Agent Type",
                    ylabel="Time Steps",
                    name=join(time_stats.agent_type, " vs ")
                )
                
                if !isnothing(p)
                    show(io, p)
                    println(io, "\n")
                else
                    println(io, "\nUnable to create completion time plot")
                end
                
                # Print detailed statistics
                println(io, "\nCompletion Time Statistics (Successful Runs):")
                for row in eachrow(time_stats)
                    println(io, "\n$(row.agent_type) Agent:")
                    println(io, "  Average Time: $(@sprintf("%.3f", row.avg_time))")
                    println(io, "  Time Std Dev: $(@sprintf("%.3f", row.std_time))")
                    println(io, "  Minimum Time: $(@sprintf("%.3f", row.min_time))")
                    println(io, "  Maximum Time: $(@sprintf("%.3f", row.max_time))")
                end
                
                # Analyze trajectory characteristics
                println(io, "\nTrajectory Characteristics Analysis:")
                for agent in safe_unique(results.agent_type)
                    agent_successful = safe_filter(successful, r -> r.agent_type == agent)
                    
                    if !isempty(agent_successful)
                        println(io, "\n$(titlecase(agent)) Agent Trajectory Metrics:")
                        
                        # Maximum position statistics
                        max_pos_stats = safe_statistics(agent_successful.max_position)
                        println(io, "\nMaximum Position Statistics:")
                        println(io, "  Mean: $(@sprintf("%.3f", max_pos_stats[:mean]))")
                        println(io, "  Median: $(@sprintf("%.3f", max_pos_stats[:median]))")
                        println(io, "  Min: $(@sprintf("%.3f", max_pos_stats[:min]))")
                        println(io, "  Max: $(@sprintf("%.3f", max_pos_stats[:max]))")
                        println(io, "  Std Dev: $(@sprintf("%.3f", max_pos_stats[:std]))")
                        
                        # Average velocity statistics
                        avg_vel_stats = safe_statistics(agent_successful.avg_velocity)
                        println(io, "\nAverage Velocity Statistics:")
                        println(io, "  Mean: $(@sprintf("%.3f", avg_vel_stats[:mean]))")
                        println(io, "  Median: $(@sprintf("%.3f", avg_vel_stats[:median]))")
                        println(io, "  Min: $(@sprintf("%.3f", avg_vel_stats[:min]))")
                        println(io, "  Max: $(@sprintf("%.3f", avg_vel_stats[:max]))")
                        println(io, "  Std Dev: $(@sprintf("%.3f", avg_vel_stats[:std]))")
                        
                        # Correlations between metrics
                        println(io, "\nMetric Correlations:")
                        time_pos_cor = safe_correlation(agent_successful.target_time, agent_successful.max_position)
                        time_vel_cor = safe_correlation(agent_successful.target_time, agent_successful.avg_velocity)
                        pos_vel_cor = safe_correlation(agent_successful.max_position, agent_successful.avg_velocity)
                        
                        println(io, "  Time vs Max Position: $(@sprintf("%.3f", time_pos_cor))")
                        println(io, "  Time vs Avg Velocity: $(@sprintf("%.3f", time_vel_cor))")
                        println(io, "  Max Position vs Avg Velocity: $(@sprintf("%.3f", pos_vel_cor))")
                    end
                end
                
                # Compare trajectory characteristics between agents
                if length(safe_unique(successful.agent_type)) > 1
                    println(io, "\nAgent Comparison (Successful Runs):")
                    
                    # Compare completion times
                    println(io, "\nCompletion Time Comparison:")
                    for (i, agent1) in enumerate(safe_unique(successful.agent_type))
                        for agent2 in safe_unique(successful.agent_type)[i+1:end]
                            agent1_times = safe_filter(successful, r -> r.agent_type == agent1).target_time
                            agent2_times = safe_filter(successful, r -> r.agent_type == agent2).target_time
                            
                            if !isempty(agent1_times) && !isempty(agent2_times)
                                time_diff = safe_mean(agent2_times) - safe_mean(agent1_times)
                                time_diff_pct = abs(time_diff) / safe_mean(agent1_times) * 100
                                
                                println(io, "\n  $(titlecase(agent2)) vs $(titlecase(agent1)):")
                                println(io, "    Absolute Difference: $(@sprintf("%.3f", time_diff)) steps")
                                println(io, "    Relative Difference: $(@sprintf("%.1f%%", time_diff_pct))")
                                println(io, "    Better Agent: $(time_diff < 0 ? titlecase(agent2) : titlecase(agent1))")
                            end
                        end
                    end
                end
            else
                println(io, "No successful runs found in the results.")
            end
        end
    catch e
        @error "Failed to generate trajectory analysis" exception=(e, catch_backtrace())
    end
end 