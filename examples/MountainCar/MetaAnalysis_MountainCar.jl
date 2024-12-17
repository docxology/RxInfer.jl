# Meta-analysis of Mountain Car with Active Inference

# Ensure we're in the right environment
import Pkg
if !isfile(joinpath(@__DIR__, "Project.toml"))
    error("Project.toml not found. Please run Setup.jl first")
end

# Activate the project environment
try
    Pkg.activate(@__DIR__)
    Pkg.instantiate()  # Ensure all packages are installed
catch e
    error("Failed to activate project environment: $e")
end

# Load required packages
using Distributed
using SharedArrays
using Statistics
using DataFrames
using CSV
using Dates
using Printf
using TOML
using ProgressMeter
using Logging
using LoggingExtras
using Plots
using Measures

# Add workers if not already added
if nworkers() == 1
    num_threads = Threads.nthreads()
    addprocs(num_threads; exeflags=`--project=$(Base.active_project())`)
    @info "Setting up distributed workers" num_threads=num_threads
    @info "Workers initialized" num_workers=nworkers()
end

# First, include and load modules on main process
include(joinpath(@__DIR__, "MountainCar.jl"))
include(joinpath(@__DIR__, "meta_analysis_utils.jl"))
include(joinpath(@__DIR__, "meta_analysis_simulation.jl"))
include(joinpath(@__DIR__, "meta_analysis_visualization.jl"))

# Import modules on main process
using .MountainCar
using .MetaAnalysisUtils
using .MetaAnalysisSimulation
using .MetaAnalysisVisualization

# Load all required modules on all workers
@everywhere begin
    # Activate project environment on each worker
    import Pkg
    Pkg.activate(@__DIR__)
    
    # Load required packages
    using RxInfer
    using RxInfer: getmodel, getreturnval, getvarref, getvariable
    using RxInfer.ReactiveMP: getrecent, messageout
    using HypergeometricFunctions
    using LinearAlgebra
    using Statistics
    using Plots
    using SharedArrays
    using Distributed
    
    # Include modules in dependency order
    include(joinpath(@__DIR__, "MountainCar.jl"))
    include(joinpath(@__DIR__, "meta_analysis_utils.jl"))
    include(joinpath(@__DIR__, "meta_analysis_simulation.jl"))
    include(joinpath(@__DIR__, "meta_analysis_visualization.jl"))
    
    # Import modules
    using .MountainCar
    using .MetaAnalysisUtils
    using .MetaAnalysisSimulation
    using .MetaAnalysisVisualization
end

# Setup logging
const timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
const output_dir = mkpath(joinpath(@__DIR__, "MetaAnalysis_Outputs", timestamp))
const LOG_FILE = joinpath(output_dir, "meta_analysis.log")

console_logger = ConsoleLogger(stdout, Logging.Info)
file_logger = SimpleLogger(open(LOG_FILE, "w"), Logging.Debug)
global_logger(TeeLogger(console_logger, file_logger))

@info "Starting meta-analysis" timestamp output_dir

# Load configuration
config_path = joinpath(@__DIR__, "config.toml")
@info "Loading configuration" config_path

if !isfile(config_path)
    error("Configuration file not found at: $config_path")
end

config = TOML.parsefile(config_path)

# Extract meta-analysis parameters
n_force = 5  # Number of engine force values to test
n_friction = 5  # Number of friction values to test
timesteps = config["simulation"]["timesteps"]
planning_horizon = config["simulation"]["planning_horizon"]

@info "Configuration loaded" n_force n_friction timesteps planning_horizon

# Create parameter grids
force_values = range(0.02, 0.06, length=n_force)
friction_values = range(0.05, 0.15, length=n_friction)

# Run parameter sweep
@info "Starting parameter sweep..."
batch = MetaAnalysisSimulation.run_simulation_batch(force_values, friction_values, config)

# Process and visualize results
@info "Processing results..."
df = MetaAnalysisVisualization.process_results(batch, output_dir)

# Create visualizations
@info "Generating visualizations..."
MetaAnalysisVisualization.create_parameter_sweep_plots(df, output_dir)
MetaAnalysisVisualization.create_performance_plots(df, output_dir)
MetaAnalysisVisualization.create_correlation_plots(df, output_dir)

@info "Meta-analysis complete" output_dir

