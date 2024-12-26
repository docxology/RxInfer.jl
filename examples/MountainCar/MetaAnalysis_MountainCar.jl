"""
    MetaAnalysis_MountainCar.jl

Main script for running meta-analysis of the Mountain Car environment.
Compares performance of Active Inference and Naive agents across different physics parameters.
"""

# Ensure we're in the right environment
import Pkg
Pkg.activate(@__DIR__)

using RxInfer
using Logging
using TOML
using DataFrames
using CSV
using UnicodePlots
using Statistics
using Printf
using Dates

# Import MountainCar module
include("MountainCar_Standalone_12-26-2024.jl")
using .MountainCar

# Import meta-analysis modules
include("meta_analysis_simulation.jl")
using .MetaAnalysisSimulation

include("meta_analysis_utils.jl")
include("visualization_functions.jl")

# Load configuration
@info "Loading configuration..."
config = TOML.parsefile(joinpath(@__DIR__, "config.toml"))

# Create output directory with timestamp
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
base_output_dir = joinpath(@__DIR__, "meta_analysis_results", timestamp)
mkpath(base_output_dir)

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
batch = SimulationBatch(
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

# Save raw results
@info "Saving raw results..."
if config["output"]["save_raw_data"]
    CSV.write(joinpath(base_output_dir, "raw_results.csv"), results)
end

# Create analysis directories
analysis_dirs = Dict(
    "success_rates" => joinpath(base_output_dir, "success_rates"),
    "performance_metrics" => joinpath(base_output_dir, "performance_metrics"),
    "energy_analysis" => joinpath(base_output_dir, "energy_analysis"),
    "control_analysis" => joinpath(base_output_dir, "control_analysis"),
    "parameter_analysis" => joinpath(base_output_dir, "parameter_analysis"),
    "trajectory_analysis" => joinpath(base_output_dir, "trajectory_analysis")
)

for dir in values(analysis_dirs)
    mkpath(dir)
end

# Set visualization parameters
ENV["COLUMNS"] = config["visualization"]["plot_width"]
ENV["LINES"] = config["visualization"]["plot_height"]

# Generate visualizations and analyses
@info "Generating visualizations and analyses..."

# Success rate analysis
@info "Analyzing success rates..."
plot_success_rate_comparison(results, joinpath(analysis_dirs["success_rates"], "success_rates"))

# Performance metrics analysis
@info "Analyzing performance metrics..."
plot_performance_metrics(results, joinpath(analysis_dirs["performance_metrics"], "performance_metrics"))

# Energy analysis
@info "Analyzing energy usage..."
plot_energy_comparison(results, joinpath(analysis_dirs["energy_analysis"], "energy_analysis"))

# Control analysis
@info "Analyzing control strategies..."
plot_control_comparison(results, joinpath(analysis_dirs["control_analysis"], "control_analysis"))

# Parameter sweep analysis
@info "Analyzing parameter effects..."
plot_parameter_sweep_analysis(results, joinpath(analysis_dirs["parameter_analysis"], "parameter_analysis"))

# Trajectory analysis
@info "Analyzing trajectories..."
plot_trajectory_analysis(results, joinpath(analysis_dirs["trajectory_analysis"], "trajectory_analysis"))

# Generate summary report
@info "Generating summary report..."
generate_summary_report(results, base_output_dir)

@info "Meta-analysis complete!" output_dir=base_output_dir

