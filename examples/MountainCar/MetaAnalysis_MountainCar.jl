"""
    MetaAnalysis_MountainCar.jl

Main script for running meta-analysis of the Mountain Car environment.
Compares performance of Active Inference and Naive agents across different physics parameters.
"""

# Ensure we're in the right environment
import Pkg
Pkg.activate(@__DIR__)

# Base imports
using RxInfer
using DataFrames
using CSV
using TOML
using Dates
using Printf
using Statistics
using StatsBase
using HypothesisTests

# Plotting imports - explicitly import needed functions
import UnicodePlots
import Plots: savefig, plot, heatmap, scatter!, plot!
import StatsPlots: violin, violin!

# First, import the standalone MountainCar implementation
include("MountainCar_Standalone_12-26-2024.jl")

# Then import meta-analysis modules (which depend on MountainCar)
include("meta_analysis_simulation.jl")
include("meta_analysis_utils.jl")
include("visualization_functions.jl")

# Import all needed modules
using .MountainCar
using .MetaAnalysisSimulation
using .MetaAnalysisVisualization
using .MetaAnalysisVisualization: save_plot_safely, VisualizationPlot, ASCIIPlot, StandardPlot

# Create custom plotting functions that handle namespaces
function create_success_heatmap(data::Matrix, force_vals, friction_vals, title::String)
    ASCIIPlot(
        UnicodePlots.heatmap(
            friction_vals,
            force_vals,
            data,
            title=title,
            xlabel="Friction",
            ylabel="Force",
            width=150,
            height=50,
            border=:ascii
        )
    )
end

function create_metric_plot(x::AbstractVector, y::AbstractVector, title::String, xlabel::String, ylabel::String)
    ASCIIPlot(
        UnicodePlots.lineplot(
            x, y,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            width=150,
            height=50,
            border=:ascii
        )
    )
end

function create_distribution_plot(data::AbstractVector, title::String, xlabel::String)
    ASCIIPlot(
        UnicodePlots.histogram(
            data,
            title=title,
            xlabel=xlabel,
            ylabel="Count",
            width=150,
            height=50,
            border=:ascii
        )
    )
end

# Load configuration
@info "Loading configuration..."
config = TOML.parsefile(joinpath(@__DIR__, "config.toml"))

# Set plot dimensions from config
ENV["COLUMNS"] = string(config["visualization"]["plot_width"])
ENV["LINES"] = string(config["visualization"]["plot_height"])

# Create output directory with timestamp
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
base_output_dir = joinpath(@__DIR__, "meta_analysis_results", timestamp)
mkpath(base_output_dir)

# Create analysis directories
analysis_dirs = Dict(
    "success_rates" => joinpath(base_output_dir, "success_rates"),
    "performance_metrics" => joinpath(base_output_dir, "performance_metrics"),
    "energy_analysis" => joinpath(base_output_dir, "energy_analysis"),
    "control_analysis" => joinpath(base_output_dir, "control_analysis"),
    "parameter_analysis" => joinpath(base_output_dir, "parameter_analysis"),
    "trajectory_analysis" => joinpath(base_output_dir, "trajectory_analysis"),
    "heatmaps" => joinpath(base_output_dir, "heatmaps"),
    "analysis" => joinpath(base_output_dir, "analysis"),
    "ascii_plots" => joinpath(base_output_dir, "ascii_plots")  # New directory for ASCII plot outputs
)

# Create all directories
for dir in values(analysis_dirs)
    mkpath(dir)
end

# Copy config file to output directory for reference
cp(joinpath(@__DIR__, "config.toml"), joinpath(base_output_dir, "config.toml"), force=true)

# Setup parameter ranges
@info "Setting up parameter ranges..."
force_range = range(
    Float64(config["meta_analysis"]["min_force"]),
    Float64(config["meta_analysis"]["max_force"]),
    length=Int(config["meta_analysis"]["force_steps"])
)
friction_range = range(
    Float64(config["meta_analysis"]["min_friction"]),
    Float64(config["meta_analysis"]["max_friction"]),
    length=Int(config["meta_analysis"]["friction_steps"])
)

# Create simulation batch
@info "Creating simulation batch..."
batch = SimulationBatch(;
    force_range=force_range,
    friction_range=friction_range,
    n_episodes=Int(config["simulation"]["n_episodes"]),
    max_steps=Int(config["simulation"]["max_steps"]),
    planning_horizon=Int(config["simulation"]["planning_horizon"]),
    initial_state=config["initial_state"],
    target_state=config["target_state"]
)

# Run simulations
@info "Running simulation batch..."
results = run_simulation_batch(batch)

# Convert results to DataFrame
@info "Processing results..."
results_df = DataFrame(results)
CSV.write(joinpath(base_output_dir, "raw_results.csv"), results_df)
@info "Saved raw results to CSV"

# Generate visualizations and analyses
@info "Generating visualizations and analyses..."

# Create a function to safely handle plot saving
function save_plots_with_logging(plots::Vector{<:VisualizationPlot}, base_path::String, plot_type::String)
    @info "Saving $(plot_type) plots..."
    for (i, plot) in enumerate(plots)
        try
            save_plot_safely(plot, 
                joinpath(base_path, "$(plot_type)_$(i)"),
                "$(plot_type)_$(i)"
            )
        catch e
            @warn "Failed to save plot" plot_type=plot_type index=i exception=e
        end
    end
end

# Success rate analysis
@info "Analyzing success rates..."
if (success_plots = plot_success_rate_comparison(results_df, joinpath(analysis_dirs["success_rates"], "success_rates"))) !== nothing
    save_plots_with_logging(success_plots, analysis_dirs["ascii_plots"], "success_rates")
end

# Performance metrics analysis
@info "Analyzing performance metrics..."
if (perf_plots = plot_performance_metrics(results_df, joinpath(analysis_dirs["performance_metrics"], "performance_metrics"))) !== nothing
    save_plots_with_logging(perf_plots, analysis_dirs["ascii_plots"], "performance_metrics")
end

# Energy analysis
@info "Analyzing energy usage..."
if (energy_plots = plot_energy_comparison(results_df, joinpath(analysis_dirs["energy_analysis"], "energy_analysis"))) !== nothing
    save_plots_with_logging(energy_plots, analysis_dirs["ascii_plots"], "energy_analysis")
end

# Control analysis
@info "Analyzing control strategies..."
if (control_plots = plot_control_comparison(results_df, joinpath(analysis_dirs["control_analysis"], "control_analysis"))) !== nothing
    save_plots_with_logging(control_plots, analysis_dirs["ascii_plots"], "control_analysis")
end

# Parameter sweep analysis
@info "Analyzing parameter effects..."
if (param_plots = plot_parameter_sweep_analysis(results_df, joinpath(analysis_dirs["parameter_analysis"], "parameter_analysis"))) !== nothing
    save_plots_with_logging(param_plots, analysis_dirs["ascii_plots"], "parameter_analysis")
end

# Trajectory analysis
@info "Analyzing trajectories..."
if (traj_plots = plot_trajectory_analysis(results_df, joinpath(analysis_dirs["trajectory_analysis"], "trajectory_analysis"))) !== nothing
    save_plots_with_logging(traj_plots, analysis_dirs["ascii_plots"], "trajectory_analysis")
end

# Generate summary report
@info "Generating summary report..."
generate_summary_report(results_df, base_output_dir)

# Generate detailed heatmaps and statistical analysis
@info "Generating detailed heatmaps and statistical analysis..."
try
    heatmap_plots, summary_stats = create_comparative_heatmaps(results_df, config)
    analysis_results = perform_anova_analysis(results_df)
    
    # Save heatmap plots
    if !isempty(heatmap_plots)
        @info "Saving heatmap plots..."
        for (i, plot) in enumerate(heatmap_plots)
            save_plot_safely(plot, 
                joinpath(analysis_dirs["heatmaps"], "heatmap_$(i)"),
                "heatmap_$(i)"
            )
        end
    end
    
    # Save summary statistics plot
    if !isnothing(summary_stats)
        @info "Saving summary statistics plot..."
        save_plot_safely(summary_stats,
            joinpath(analysis_dirs["heatmaps"], "summary_statistics"),
            "summary"
        )
    end
    
    @info "Generated detailed heatmaps and statistical analysis" output_dir=analysis_dirs["heatmaps"]
catch e
    @error "Failed to generate heatmaps and statistical analysis" exception=(e, catch_backtrace())
end

# Create a master summary file with links to all plots and detailed logging
@info "Creating master summary file..."
open(joinpath(base_output_dir, "plot_summary.txt"), "w") do io
    println(io, "Meta-Analysis Plot Summary")
    println(io, "=======================\n")
    println(io, "Generated on: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))\n")
    
    println(io, "Configuration:")
    println(io, "-------------")
    println(io, "Force range: $(minimum(force_range)) to $(maximum(force_range)) ($(length(force_range)) steps)")
    println(io, "Friction range: $(minimum(friction_range)) to $(maximum(friction_range)) ($(length(friction_range)) steps)")
    println(io, "Episodes per configuration: $(config["simulation"]["n_episodes"])")
    println(io, "Maximum steps per episode: $(config["simulation"]["max_steps"])\n")
    
    println(io, "Directory Structure:")
    println(io, "------------------")
    println(io, "ASCII Plots: $(relpath(analysis_dirs["ascii_plots"], base_output_dir))")
    println(io, "Regular Plots: $(relpath(analysis_dirs["heatmaps"], base_output_dir))\n")
    
    println(io, "Generated Files:")
    println(io, "---------------")
    for dir in sort(collect(values(analysis_dirs)))
        if isdir(dir)
            rel_path = relpath(dir, base_output_dir)
            files = sort(readdir(dir))
            if !isempty(files)
                println(io, "\n$rel_path/")
                for file in files
                    println(io, "  - $file")
                end
            end
        end
    end
    
    println(io, "\nAnalysis Summary:")
    println(io, "-----------------")
    println(io, "Total simulations: $(nrow(results_df))")
    println(io, "Success rate: $(round(100 * mean(results_df.success), digits=2))%")
    println(io, "Average completion time: $(round(mean(skipmissing(results_df.completion_time)), digits=2)) steps")
    println(io, "Average energy usage: $(round(mean(skipmissing(results_df.total_energy)), digits=2))")
    
    # Add performance comparison
    println(io, "\nPerformance Comparison:")
    println(io, "---------------------")
    for agent_type in unique(results_df.agent_type)
        agent_data = filter(row -> row.agent_type == agent_type, results_df)
        println(io, "\n$(titlecase(agent_type)):")
        println(io, "  Success rate: $(round(100 * mean(agent_data.success), digits=2))%")
        println(io, "  Avg completion time: $(round(mean(skipmissing(agent_data.completion_time)), digits=2)) steps")
        println(io, "  Avg energy usage: $(round(mean(skipmissing(agent_data.total_energy)), digits=2))")
        println(io, "  Avg control effort: $(round(mean(skipmissing(agent_data.control_effort)), digits=2))")
    end
end

@info "Meta-analysis complete!" output_dir=base_output_dir

