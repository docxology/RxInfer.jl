"""
    meta_analysis_utils.jl

Utility functions for meta-analysis of the Mountain Car environment.
"""

using Statistics
using DataFrames

export calculate_oscillations, calculate_control_effort, calculate_stability, 
       calculate_efficiency, analyze_results, generate_summary_report

"""
    calculate_oscillations(positions::Vector{Float64})

Calculate the number of oscillations in a position trajectory.
"""
function calculate_oscillations(positions::Vector{Float64})
    if length(positions) < 3
        return 0
    end
    
    # Count direction changes
    oscillations = 0
    prev_direction = sign(positions[2] - positions[1])
    
    for i in 3:length(positions)
        direction = sign(positions[i] - positions[i-1])
        if direction != prev_direction && direction != 0
            oscillations += 1
            prev_direction = direction
        end
    end
    
    return oscillations
end

"""
    calculate_control_effort(actions::Vector{Float64})

Calculate the total control effort from a sequence of actions.
"""
function calculate_control_effort(actions::Vector{Float64})
    return sum(abs.(actions))
end

"""
    calculate_stability(positions::Vector{Float64}, target_position::Float64)

Calculate the stability metric based on position variance around target.
"""
function calculate_stability(positions::Vector{Float64}, target_position::Float64)
    if isempty(positions)
        return 0.0
    end
    
    # Calculate variance of positions relative to target
    deviations = positions .- target_position
    variance = var(deviations)
    
    # Convert to stability metric (higher is more stable)
    return 1.0 / (1.0 + variance)
end

"""
    calculate_efficiency(success::Bool, target_time::Float64, total_energy::Float64)

Calculate the efficiency metric combining success, time, and energy usage.
"""
function calculate_efficiency(success::Bool, target_time::Float64, total_energy::Float64)
    if !success || isinf(target_time) || isinf(total_energy)
        return 0.0
    end
    
    # Normalize time and energy to [0,1] range using reasonable maximum values
    max_time = 1000.0  # maximum reasonable time steps
    max_energy = 1000.0  # maximum reasonable energy usage
    
    norm_time = min(target_time / max_time, 1.0)
    norm_energy = min(total_energy / max_energy, 1.0)
    
    # Combine metrics with weights
    time_weight = 0.4
    energy_weight = 0.6
    
    return 1.0 - (time_weight * norm_time + energy_weight * norm_energy)
end

"""
    analyze_results(results::DataFrame)

Analyze the results of the meta-analysis and return summary statistics.
"""
function analyze_results(results::DataFrame)
    # Group results by agent type
    grouped = groupby(results, :agent_type)
    
    # Calculate summary statistics for each agent type
    summary_stats = Dict()
    
    for group in grouped
        agent = group.agent_type[1]
        summary_stats[agent] = Dict(
            "success_rate" => mean(group.success),
            "avg_target_time" => mean(group[group.success, :target_time]),
            "avg_energy" => mean(group.total_energy),
            "avg_efficiency" => mean(group.efficiency),
            "avg_stability" => mean(group.stability),
            "avg_oscillations" => mean(group.oscillations),
            "avg_control_effort" => mean(group.control_effort),
            "max_position_reached" => maximum(group.max_position),
            "avg_velocity" => mean(group.avg_velocity)
        )
    end
    
    return summary_stats
end

"""
    generate_summary_report(results::DataFrame, output_dir::String)

Generate a detailed summary report of the meta-analysis results.
"""
function generate_summary_report(results::DataFrame, output_dir::String)
    # Analyze results
    summary_stats = analyze_results(results)
    
    # Create report string
    report = """
    # Mountain Car Meta-Analysis Summary Report
    
    ## Overview
    - Total simulations: $(nrow(results))
    - Parameter combinations tested: $(length(unique(zip(results.force, results.friction))))
    - Timestamp: $(Dates.now())
    
    ## Performance Comparison
    
    ### Success Rates
    """
    
    for (agent, stats) in summary_stats
        report *= """
        
        ### $agent Agent Performance
        - Success Rate: $(round(stats["success_rate"] * 100, digits=2))%
        - Average Time to Target: $(round(stats["avg_target_time"], digits=2)) steps
        - Average Energy Usage: $(round(stats["avg_energy"], digits=2))
        - Average Efficiency: $(round(stats["avg_efficiency"], digits=2))
        - Average Stability: $(round(stats["avg_stability"], digits=2))
        - Average Oscillations: $(round(stats["avg_oscillations"], digits=2))
        - Average Control Effort: $(round(stats["avg_control_effort"], digits=2))
        - Maximum Position Reached: $(round(stats["max_position_reached"], digits=2))
        - Average Velocity: $(round(stats["avg_velocity"], digits=2))
        """
    end
    
    # Add parameter analysis
    report *= """
    
    ## Parameter Analysis
    
    ### Effect of Engine Force
    - Minimum force tested: $(minimum(results.force))
    - Maximum force tested: $(maximum(results.force))
    - Optimal force range: $(mean(results[results.success, :force]) ± std(results[results.success, :force]))
    
    ### Effect of Friction
    - Minimum friction tested: $(minimum(results.friction))
    - Maximum friction tested: $(maximum(results.friction))
    - Optimal friction range: $(mean(results[results.success, :friction]) ± std(results[results.success, :friction]))
    """
    
    # Write report to file
    open(joinpath(output_dir, "summary_report.md"), "w") do io
        write(io, report)
    end
end 