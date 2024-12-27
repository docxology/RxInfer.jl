"""
    visualization_functions.jl

Contains visualization functions for the Mountain Car meta-analysis.
"""
module MetaAnalysisVisualization

import UnicodePlots
import Plots
import StatsPlots  # Required for boxplots
using Statistics
using Printf
using DataFrames
using Dates
using StatsBase
using HypothesisTests

export plot_success_rate_comparison, plot_performance_metrics, plot_energy_comparison,
       plot_control_comparison, plot_parameter_sweep_analysis, plot_trajectory_analysis,
       create_comparative_heatmaps, perform_anova_analysis, generate_summary_report

# Helper function to create a heatmap with UnicodePlots
function create_heatmap(data::Matrix, title::String)
    return UnicodePlots.heatmap(data, 
        title=title,
        colormap=:viridis,
        width=40,
        height=15
    )
end

# Helper function to create a line plot
function create_lineplot(x::AbstractVector, y::AbstractVector, title::String)
    return UnicodePlots.lineplot(x, y,
        title=title,
        width=40,
        height=15
    )
end

# Helper function to create a histogram
function create_histogram(data::AbstractVector, title::String)
    return UnicodePlots.histogram(data,
        title=title,
        width=40,
        height=15
    )
end

# Helper function to ensure directory exists
function ensure_directory(path::String)
    if !isdir(path)
        mkpath(path)
    end
end

# Helper function to save plot safely
function save_plot_safely(plot, path::String, metric::String)
    try
        ensure_directory(dirname(path))
        Plots.savefig(plot, path)
    catch e
        @warn "Failed to save plot" metric=metric path=path exception=e
    end
end

# Helper function to calculate success rate
function calculate_success_rate(df::DataFrame, group_cols::Vector{Symbol})
    success_df = combine(groupby(df, group_cols)) do group_df
        (
            success_rate = mean(group_df.success) * 100,
            completion_time = mean(skipmissing(group_df.completion_time)),
            total_energy = mean(skipmissing(group_df.total_energy)),
            control_effort = mean(skipmissing(group_df.control_effort))
        )
    end
    return success_df
end

"""
    generate_summary_report(results_df::DataFrame, output_dir::String)

Generate a comprehensive summary report of the meta-analysis results.
"""
function generate_summary_report(results_df::DataFrame, output_dir::String)
    try
        report_path = joinpath(output_dir, "summary_report.txt")
        open(report_path, "w") do io
            println(io, "Meta-Analysis Summary Report")
            println(io, "========================\n")
            
            # Overall statistics
            println(io, "Overall Statistics:")
            println(io, "-----------------")
            total_sims = nrow(results_df)
            total_successes = count(results_df.success)
            println(io, "Total Simulations: $total_sims")
            println(io, "Successful Simulations: $total_successes")
            println(io, "Overall Success Rate: $(round(100 * total_successes / total_sims, digits=2))%\n")
            
            # Statistics by agent type
            println(io, "Performance by Agent Type:")
            println(io, "-----------------------")
            by_agent = combine(groupby(results_df, :agent_type)) do group_df
                (
                    total = nrow(group_df),
                    successes = count(group_df.success),
                    success_rate = 100 * mean(group_df.success),
                    avg_completion_time = mean(skipmissing(group_df.completion_time)),
                    avg_energy = mean(skipmissing(group_df.total_energy)),
                    avg_control = mean(skipmissing(group_df.control_effort))
                )
            end
            
            for row in eachrow(by_agent)
                println(io, "\n$(titlecase(row.agent_type)):")
                println(io, "  Simulations: $(row.total)")
                println(io, "  Successes: $(row.successes)")
                println(io, "  Success Rate: $(round(row.success_rate, digits=2))%")
                println(io, "  Average Completion Time: $(round(row.avg_completion_time, digits=2)) steps")
                println(io, "  Average Energy Usage: $(round(row.avg_energy, digits=2))")
                println(io, "  Average Control Effort: $(round(row.avg_control, digits=2))")
            end
        end
    catch e
        @error "Failed to generate summary report" exception=e
    end
end

"""
    plot_success_rate_comparison(results_df::DataFrame, output_path::String)

Create success rate comparison plots.
"""
function plot_success_rate_comparison(results_df::DataFrame, output_path::String)
    try
        # Calculate success rates for each agent type and parameter combination
        success_df = calculate_success_rate(results_df, [:agent_type, :force, :friction])
        
        # Create output file
        open("$(output_path).txt", "w") do io
            println(io, "Success Rate Comparison")
            println(io, "=====================\n")
            
            # Overall success rates by agent type
            overall_rates = combine(groupby(results_df, :agent_type)) do group_df
                (success_rate = mean(group_df.success) * 100,)
            end
            
            println(io, "Overall Success Rates:")
            for row in eachrow(overall_rates)
                println(io, "  $(row.agent_type): $(round(row.success_rate, digits=2))%")
            end
            
            # Create heatmaps for each agent type
            for agent_type in unique(success_df.agent_type)
                agent_data = filter(row -> row.agent_type == agent_type, success_df)
                
                # Create matrix for heatmap
                force_vals = sort(unique(agent_data.force))
                friction_vals = sort(unique(agent_data.friction))
                success_matrix = zeros(length(force_vals), length(friction_vals))
                
                for (i, force) in enumerate(force_vals)
                    for (j, friction) in enumerate(friction_vals)
                        data = filter(row -> row.force ≈ force && row.friction ≈ friction, agent_data)
                        if !isempty(data)
                            success_matrix[i, j] = first(data.success_rate)
                        end
                    end
                end
                
                # Create heatmap
                println(io, "\nSuccess Rate Heatmap for $(titlecase(agent_type)):")
                hm = create_heatmap(success_matrix, "Success Rate (%)")
                println(io, hm)
            end
        end
    catch e
        @warn "Failed to create success rate comparison" exception=e
    end
end

"""
    plot_performance_metrics(results_df::DataFrame, output_path::String)

Create performance metrics comparison plots.
"""
function plot_performance_metrics(results_df::DataFrame, output_path::String)
    try
        # Calculate metrics by agent type
        metrics_df = combine(groupby(results_df, :agent_type)) do group_df
            (
                avg_completion_time = mean(skipmissing(group_df.completion_time)),
                std_completion_time = std(skipmissing(group_df.completion_time)),
                avg_energy = mean(skipmissing(group_df.total_energy)),
                std_energy = std(skipmissing(group_df.total_energy)),
                avg_control = mean(skipmissing(group_df.control_effort)),
                std_control = std(skipmissing(group_df.control_effort))
            )
        end
        
        open("$(output_path).txt", "w") do io
            println(io, "Performance Metrics Comparison")
            println(io, "===========================\n")
            
            # Print metrics table
            println(io, "Metrics by Agent Type:")
            println(io, "-----------------")
            for row in eachrow(metrics_df)
                println(io, "\n$(titlecase(row.agent_type)):")
                println(io, "  Completion Time: $(round(row.avg_completion_time, digits=2)) ± $(round(row.std_completion_time, digits=2)) steps")
                println(io, "  Total Energy: $(round(row.avg_energy, digits=2)) ± $(round(row.std_energy, digits=2))")
                println(io, "  Control Effort: $(round(row.avg_control, digits=2)) ± $(round(row.std_control, digits=2))")
            end
            
            # Create time series plots
            println(io, "\nTime Series Analysis:")
            for metric in [:completion_time, :total_energy, :control_effort]
                println(io, "\n$(titlecase(String(metric))):")
                for agent_type in unique(results_df.agent_type)
                    data = filter(row -> row.agent_type == agent_type, results_df)
                    p = create_lineplot(
                        1:nrow(data),
                        data[!, metric],
                        "$(titlecase(agent_type)) - $(titlecase(String(metric)))"
                    )
                    println(io, p)
                end
            end
        end
    catch e
        @warn "Failed to create performance metrics plots" exception=e
    end
end

"""
    plot_energy_comparison(results_df::DataFrame, output_path::String)

Create energy usage comparison plots.
"""
function plot_energy_comparison(results_df::DataFrame, output_path::String)
    try
        # Calculate energy metrics by agent type and parameter combination
        energy_df = combine(groupby(results_df, [:agent_type, :force, :friction])) do group_df
            (
                avg_energy = mean(skipmissing(group_df.total_energy)),
                std_energy = std(skipmissing(group_df.total_energy)),
                efficiency = mean(skipmissing(group_df.efficiency))
            )
        end
        
        open("$(output_path).txt", "w") do io
            println(io, "Energy Usage Analysis")
            println(io, "===================\n")
            
            # Overall energy statistics
            println(io, "Overall Energy Statistics:")
            for agent_type in unique(results_df.agent_type)
                data = filter(row -> row.agent_type == agent_type, results_df)
                println(io, "\n$(titlecase(agent_type)):")
                println(io, "  Average Energy: $(round(mean(skipmissing(data.total_energy)), digits=2))")
                println(io, "  Energy Std Dev: $(round(std(skipmissing(data.total_energy)), digits=2))")
                println(io, "  Average Efficiency: $(round(mean(skipmissing(data.efficiency)), digits=4))")
            end
            
            # Create energy distribution plots
            println(io, "\nEnergy Distribution by Agent Type:")
            for agent_type in unique(results_df.agent_type)
                data = filter(row -> row.agent_type == agent_type, results_df)
                p = create_histogram(
                    collect(skipmissing(data.total_energy)),
                    "$(titlecase(agent_type)) Energy Distribution"
                )
                println(io, p)
            end
        end
    catch e
        @warn "Failed to create energy comparison plots" exception=e
    end
end

"""
    plot_control_comparison(results_df::DataFrame, output_path::String)

Create control strategy comparison plots.
"""
function plot_control_comparison(results_df::DataFrame, output_path::String)
    try
        open("$(output_path).txt", "w") do io
            println(io, "Control Strategy Analysis")
            println(io, "=======================\n")
            
            # Overall control statistics
            println(io, "Overall Control Statistics:")
            for agent_type in unique(results_df.agent_type)
                data = filter(row -> row.agent_type == agent_type, results_df)
                println(io, "\n$(titlecase(agent_type)):")
                println(io, "  Average Control Effort: $(round(mean(skipmissing(data.control_effort)), digits=2))")
                println(io, "  Control Effort Std Dev: $(round(std(skipmissing(data.control_effort)), digits=2))")
            end
            
            # Create control effort distribution plots
            println(io, "\nControl Effort Distribution by Agent Type:")
            for agent_type in unique(results_df.agent_type)
                data = filter(row -> row.agent_type == agent_type, results_df)
                p = create_histogram(
                    collect(skipmissing(data.control_effort)),
                    "$(titlecase(agent_type)) Control Effort"
                )
                println(io, p)
            end
        end
    catch e
        @warn "Failed to create control comparison plots" exception=e
    end
end

"""
    plot_parameter_sweep_analysis(results_df::DataFrame, output_path::String)

Create parameter sweep analysis plots.
"""
function plot_parameter_sweep_analysis(results_df::DataFrame, output_path::String)
    try
        open("$(output_path).txt", "w") do io
            println(io, "Parameter Sweep Analysis")
            println(io, "======================\n")
            
            # Analyze effect of force and friction on success rate
            println(io, "Effect of Parameters on Success Rate:")
            for agent_type in unique(results_df.agent_type)
                data = filter(row -> row.agent_type == agent_type, results_df)
                
                # Force effect
                force_effect = combine(groupby(data, :force)) do group_df
                    (success_rate = mean(group_df.success) * 100,)
                end
                println(io, "\n$(titlecase(agent_type)) - Force Effect:")
                p = create_lineplot(
                    force_effect.force,
                    force_effect.success_rate,
                    "Success Rate vs Force"
                )
                println(io, p)
                
                # Friction effect
                friction_effect = combine(groupby(data, :friction)) do group_df
                    (success_rate = mean(group_df.success) * 100,)
                end
                println(io, "\n$(titlecase(agent_type)) - Friction Effect:")
                p = create_lineplot(
                    friction_effect.friction,
                    friction_effect.success_rate,
                    "Success Rate vs Friction"
                )
                println(io, p)
            end
        end
    catch e
        @warn "Failed to create parameter sweep analysis" exception=e
    end
end

"""
    plot_trajectory_analysis(results_df::DataFrame, output_path::String)

Create trajectory analysis plots.
"""
function plot_trajectory_analysis(results_df::DataFrame, output_path::String)
    try
        open("$(output_path).txt", "w") do io
            println(io, "Trajectory Analysis")
            println(io, "==================\n")
            
            # Analyze successful vs failed trajectories
            println(io, "Success vs Failure Analysis:")
            for agent_type in unique(results_df.agent_type)
                data = filter(row -> row.agent_type == agent_type, results_df)
                success_data = filter(row -> row.success, data)
                failure_data = filter(row -> !row.success, data)
                
                println(io, "\n$(titlecase(agent_type)):")
                println(io, "  Successful Trajectories: $(nrow(success_data))")
                println(io, "  Failed Trajectories: $(nrow(failure_data))")
                
                if !isempty(success_data)
                    println(io, "\n  Successful Trajectory Characteristics:")
                    println(io, "    Average Final Position: $(round(mean(success_data.final_position), digits=3))")
                    println(io, "    Average Final Velocity: $(round(mean(success_data.final_velocity), digits=3))")
                    println(io, "    Average Completion Time: $(round(mean(success_data.completion_time), digits=1)) steps")
                end
            end
        end
    catch e
        @warn "Failed to create trajectory analysis" exception=e
    end
end

"""
    create_comparative_heatmaps(results_df::DataFrame, config::Dict)

Create detailed heatmaps comparing Active Inference and Naive agents across parameter space.
"""
function create_comparative_heatmaps(results_df::DataFrame, config::Dict)
    try
        # Create output directory with timestamp
        timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
        base_output_dir = joinpath(@__DIR__, "meta_analysis_results", timestamp)
        output_dir = joinpath(base_output_dir, "heatmaps")
        ensure_directory(output_dir)

        # Calculate success rates and other metrics
        metrics_df = combine(groupby(results_df, [:agent_type, :force, :friction])) do group_df
            (
                success_rate = mean(group_df.success) * 100,
                avg_completion_time = mean(skipmissing(group_df.completion_time)),
                avg_energy = mean(skipmissing(group_df.total_energy)),
                avg_control = mean(skipmissing(group_df.control_effort))
            )
        end

        # Extract parameter ranges
        force_range = sort(unique(results_df.force))
        friction_range = sort(unique(results_df.friction))

        # Initialize plot collection
        plot_collection = []

        # Create heatmaps for each metric
        metrics = [:success_rate, :avg_completion_time, :avg_energy, :avg_control]
        metric_names = ["Success Rate (%)", "Completion Time", "Energy Usage", "Control Effort"]

        for (metric, name) in zip(metrics, metric_names)
            # Create matrices for heatmaps
            active_matrix = zeros(length(force_range), length(friction_range))
            naive_matrix = zeros(length(force_range), length(friction_range))
            
            # Fill matrices
            for (i, force) in enumerate(force_range)
                for (j, friction) in enumerate(friction_range)
                    active_data = filter(row -> row.agent_type == "active_inference" && 
                                              row.force ≈ force && 
                                              row.friction ≈ friction, metrics_df)
                    naive_data = filter(row -> row.agent_type == "naive" && 
                                             row.force ≈ force && 
                                             row.friction ≈ friction, metrics_df)
                    
                    if !isempty(active_data)
                        active_matrix[i, j] = first(active_data[!, metric])
                    end
                    if !isempty(naive_data)
                        naive_matrix[i, j] = first(naive_data[!, metric])
                    end
                end
            end

            # Create individual plots
            p_active = Plots.heatmap(friction_range, force_range, active_matrix,
                title="Active Inference: $name",
                xlabel="Friction",
                ylabel="Force",
                c=:viridis,
                aspect_ratio=:equal)

            p_naive = Plots.heatmap(friction_range, force_range, naive_matrix,
                title="Naive Agent: $name",
                xlabel="Friction",
                ylabel="Force",
                c=:viridis,
                aspect_ratio=:equal)

            diff_matrix = active_matrix .- naive_matrix
            p_diff = Plots.heatmap(friction_range, force_range, diff_matrix,
                title="Difference (Active - Naive)",
                xlabel="Friction",
                ylabel="Force",
                c=:RdBu,
                aspect_ratio=:equal)

            # Create combined plot
            p_combined = Plots.plot(p_active, p_naive, p_diff, 
                layout=(1,3), 
                size=(1800,600),
                margin=5Plots.mm)
            
            push!(plot_collection, p_combined)

            # Save individual plot
            output_file = joinpath(output_dir, "$(String(metric))_comparison.png")
            save_plot_safely(p_combined, output_file, String(metric))
        end

        # Create summary statistics plots
        summary_plots = []
        for (i, (metric, name)) in enumerate(zip(metrics, metric_names))
            active_data = collect(skipmissing(filter(row -> row.agent_type == "active_inference", metrics_df)[!, metric]))
            naive_data = collect(skipmissing(filter(row -> row.agent_type == "naive", metrics_df)[!, metric]))
            
            if !isempty(active_data) && !isempty(naive_data)
                # Create violin plots instead of boxplots
                p = StatsPlots.violin(
                    ["Active Inference" for _ in 1:length(active_data)],
                    active_data,
                    side=:left,
                    label="Active",
                    title=name,
                    ylabel="Value",
                    legend=:topleft
                )
                
                StatsPlots.violin!(
                    ["Naive" for _ in 1:length(naive_data)],
                    naive_data,
                    side=:right,
                    label="Naive"
                )
                
                push!(summary_plots, p)
            else
                @warn "Insufficient data for summary plot" metric=metric
            end
        end
        
        if !isempty(summary_plots)
            # Combine summary plots
            summary_stats = Plots.plot(summary_plots..., 
                layout=(2,2), 
                size=(1200,1000),
                margin=5Plots.mm)
            
            # Save summary statistics
            output_file = joinpath(output_dir, "summary_statistics.png")
            save_plot_safely(summary_stats, output_file, "summary")
        else
            @warn "No summary plots were generated"
            summary_stats = nothing
        end

        return plot_collection, summary_stats
    catch e
        @error "Failed to create comparative heatmaps" exception=e
        return nothing, nothing
    end
end

"""
    perform_anova_analysis(results_df::DataFrame)

Perform detailed ANOVA analysis comparing the effects of agent type, force, and friction
on various performance metrics.
"""
function perform_anova_analysis(results_df::DataFrame)
    try
        # Create output directory with timestamp
        timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
        base_output_dir = joinpath(@__DIR__, "meta_analysis_results", timestamp)
        output_dir = joinpath(base_output_dir, "analysis")
        mkpath(output_dir)

        # Define metrics to analyze
        metrics = [:success, :completion_time, :total_energy, :control_effort]
        metric_names = ["Success Rate", "Completion Time", "Energy Usage", "Control Effort"]
        analysis_results = Dict()

        for (metric, name) in zip(metrics, metric_names)
            # Prepare data for analysis
            if hasproperty(results_df, metric)
                active_data = collect(skipmissing(filter(row -> row.agent_type == "active_inference", results_df)[!, metric]))
                naive_data = collect(skipmissing(filter(row -> row.agent_type == "naive", results_df)[!, metric]))

                if !isempty(active_data) && !isempty(naive_data)
                    # Perform statistical tests
                    ttest_result = UnequalVarianceTTest(active_data, naive_data)
                    
                    # Store results
                    analysis_results[name] = Dict(
                        "mean_active" => mean(active_data),
                        "std_active" => std(active_data),
                        "mean_naive" => mean(naive_data),
                        "std_naive" => std(naive_data),
                        "p_value" => pvalue(ttest_result),
                        "t_statistic" => ttest_result.t,
                        "dof" => ttest_result.df
                    )
                else
                    @warn "Insufficient data for metric" metric=name
                end
            else
                @warn "Metric not found in results" metric=name
            end
        end

        if !isempty(analysis_results)
            # Save analysis results
            output_file = joinpath(output_dir, "statistical_analysis.txt")
            open(output_file, "w") do io
                println(io, "Statistical Analysis Results")
                println(io, "=========================\n")
                
                for (metric, results) in analysis_results
                    println(io, "Metric: $metric")
                    println(io, "-----------------")
                    println(io, "Active Inference:")
                    println(io, "  Mean: $(round(results["mean_active"], digits=4))")
                    println(io, "  Std:  $(round(results["std_active"], digits=4))")
                    println(io, "Naive Agent:")
                    println(io, "  Mean: $(round(results["mean_naive"], digits=4))")
                    println(io, "  Std:  $(round(results["std_naive"], digits=4))")
                    println(io, "\nStatistical Test Results:")
                    println(io, "  t-statistic: $(round(results["t_statistic"], digits=4))")
                    println(io, "  p-value:     $(round(results["p_value"], digits=4))")
                    println(io, "  DOF:         $(round(results["dof"], digits=4))")
                    println(io, "\n")
                end
            end
        end

        return analysis_results
    catch e
        @error "Failed to perform statistical analysis" exception=e
        return nothing
    end
end

end # module MetaAnalysisVisualization 