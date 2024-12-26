"""
    MetaAnalysisVisualization

Module for creating visualizations from meta-analysis results.
Provides comprehensive functions for plotting and analyzing mountain car
simulation results across different parameters and agent types.
"""
module MetaAnalysisVisualization

using UnicodePlots
using DataFrames
using Statistics
using Printf
using Dates

# Create a timestamped directory for outputs
function create_output_directory()
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    base_dir = joinpath(@__DIR__, "MetaAnalysis_Outputs", timestamp)
    mkpath(base_dir)
    @info "Created output directory" path=base_dir
    return base_dir
end

"""
    generate_all_visualizations(results::DataFrame, base_dir::String)

Generate all available visualizations and analyses for the meta-analysis results.
"""
function generate_all_visualizations(results::DataFrame, base_dir::String)
    @info "Starting visualization generation" output_dir=base_dir
    
    # Create subdirectories for different types of visualizations
    dirs = Dict(
        "success" => joinpath(base_dir, "success_rates"),
        "performance" => joinpath(base_dir, "performance_metrics"),
        "energy" => joinpath(base_dir, "energy_analysis"),
        "control" => joinpath(base_dir, "control_analysis"),
        "parameter" => joinpath(base_dir, "parameter_analysis"),
        "trajectory" => joinpath(base_dir, "trajectory_analysis")
    )
    
    # Create all directories
    for dir in values(dirs)
        mkpath(dir)
        @info "Created visualization subdirectory" path=dir
    end
    
    # Generate all visualizations
    plot_success_rate_comparison(results, joinpath(dirs["success"], "success_rates"))
    plot_performance_metrics(results, joinpath(dirs["performance"], "performance_metrics"))
    plot_energy_comparison(results, joinpath(dirs["energy"], "energy_analysis"))
    plot_control_comparison(results, joinpath(dirs["control"], "control_analysis"))
    plot_parameter_sweep_analysis(results, joinpath(dirs["parameter"], "parameter_sweep"))
    plot_trajectory_analysis(results, joinpath(dirs["trajectory"], "trajectory_analysis"))
    
    # Generate summary report
    generate_summary_report(results, base_dir)
    
    @info "Completed all visualizations" output_dir=base_dir
end

# Include all visualization functions
include("visualization_functions.jl")

# Export the main visualization function
export generate_all_visualizations

end # module 