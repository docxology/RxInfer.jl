module MetaAnalysisVisualization

using Plots
using Statistics
using DataFrames
using CSV
using Dates
using Printf
using Measures

# Import required modules
using ..MountainCar
using ..MetaAnalysisUtils
using ..MetaAnalysisSimulation

export process_results, create_parameter_sweep_plots, create_performance_plots, create_correlation_plots

"""
    process_results(batch, output_dir)

Process simulation results into a DataFrame and save to CSV.
"""
function process_results(batch, output_dir)
    # Create DataFrame from results
    df = DataFrame(
        force = Float64[],
        friction = Float64[],
        max_position = Float64[],
        avg_velocity = Float64[],
        success = Bool[],
        target_time = Float64[],
        total_energy = Float64[],
        oscillations = Float64[],
        control_effort = Float64[],
        stability = Float64[],
        efficiency = Float64[]
    )
    
    for result in batch.results
        push!(df, (
            result["force"],
            result["friction"],
            result["metrics"].max_position,
            result["metrics"].avg_velocity,
            result["metrics"].success,
            result["metrics"].target_time,
            result["metrics"].total_energy,
            result["metrics"].oscillations,
            result["metrics"].control_effort,
            result["metrics"].stability,
            result["metrics"].efficiency
        ))
    end
    
    # Save results
    CSV.write(joinpath(output_dir, "meta_analysis_results.csv"), df)
    
    return df
end

"""
    create_parameter_sweep_plots(df, output_dir)

Create heatmaps showing how different metrics vary with force and friction.
"""
function create_parameter_sweep_plots(df, output_dir)
    metrics = [
        ("success", "Success Rate"),
        ("target_time", "Time to Target"),
        ("total_energy", "Total Energy"),
        ("efficiency", "Overall Efficiency")
    ]
    
    force_values = sort(unique(df.force))
    friction_values = sort(unique(df.friction))
    n_force = length(force_values)
    n_friction = length(friction_values)
    
    for (metric, title) in metrics
        data = reshape(df[:, metric], n_force, n_friction)
        
        p = heatmap(
            force_values,
            friction_values,
            data',
            title=title,
            xlabel="Engine Force",
            ylabel="Friction Coefficient",
            color=:viridis,
            aspect_ratio=:equal,
            margin=5mm,
            size=(800, 600),
            dpi=300
        )
        
        savefig(p, joinpath(output_dir, "$(metric)_heatmap.png"))
    end
end

"""
    create_performance_plots(df, output_dir)

Create plots showing performance metrics distributions and relationships.
"""
function create_performance_plots(df, output_dir)
    # Success rate vs parameters
    p1 = scatter(
        df.force,
        df.friction,
        color=df.success,
        title="Success by Parameters",
        xlabel="Engine Force",
        ylabel="Friction Coefficient",
        marker_z=df.success,
        colorbar_title="Success",
        legend=false,
        size=(800, 600),
        dpi=300,
        margin=5mm
    )
    savefig(p1, joinpath(output_dir, "success_scatter.png"))
    
    # Time to target distribution
    p2 = histogram(
        df[df.success, :target_time],
        title="Time to Target Distribution",
        xlabel="Time Steps",
        ylabel="Count",
        legend=false,
        size=(800, 600),
        dpi=300,
        margin=5mm
    )
    savefig(p2, joinpath(output_dir, "target_time_hist.png"))
    
    # Energy vs Control Effort
    p3 = scatter(
        df.total_energy,
        df.control_effort,
        color=df.success,
        title="Energy vs Control Effort",
        xlabel="Total Energy",
        ylabel="Control Effort",
        marker_z=df.success,
        colorbar_title="Success",
        legend=false,
        size=(800, 600),
        dpi=300,
        margin=5mm
    )
    savefig(p3, joinpath(output_dir, "energy_control_scatter.png"))
end

"""
    create_correlation_plots(df, output_dir)

Create correlation matrix and plots between different metrics.
"""
function create_correlation_plots(df, output_dir)
    # Select numeric columns for correlation
    numeric_cols = [
        :max_position, :avg_velocity, :target_time,
        :total_energy, :oscillations, :control_effort,
        :stability, :efficiency
    ]
    
    # Calculate correlation matrix
    cor_matrix = cor(Matrix(df[:, numeric_cols]))
    
    # Create correlation heatmap
    p = heatmap(
        string.(numeric_cols),
        string.(numeric_cols),
        cor_matrix,
        title="Metric Correlations",
        color=:RdBu,
        clim=(-1, 1),
        aspect_ratio=:equal,
        xrotation=45,
        size=(1000, 800),
        dpi=300,
        margin=10mm
    )
    
    savefig(p, joinpath(output_dir, "correlation_matrix.png"))
end

end # module 