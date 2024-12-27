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

Generate a comprehensive summary report of the meta-analysis results.
"""
function generate_summary_report(results::DataFrame, output_dir::String)
    try
        open(joinpath(output_dir, "summary_report.txt"), "w") do io
            println(io, "=== Meta-Analysis Summary Report ===\n")
            println(io, "Analysis timestamp: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
            println(io, "-" ^ 50, "\n")
            
            # Overall statistics
            println(io, "Overall Statistics:")
            println(io, "-" ^ 20)
            
            n_total = nrow(results)
            n_successful = count(results.success)
            overall_success_rate = n_successful / n_total * 100
            
            println(io, "Total simulations: $n_total")
            println(io, "Successful runs: $n_successful")
            println(io, "Overall success rate: $(@sprintf("%.2f%%", overall_success_rate))")
            
            # Agent comparison
            println(io, "\nAgent Comparison:")
            println(io, "-" ^ 20)
            
            for agent in unique(results.agent_type)
                agent_results = filter(r -> r.agent_type == agent, results)
                n_agent = nrow(agent_results)
                n_agent_success = count(agent_results.success)
                agent_success_rate = n_agent_success / n_agent * 100
                
                println(io, "\n$(titlecase(agent)) Agent:")
                println(io, "  Total runs: $n_agent")
                println(io, "  Successful runs: $n_agent_success")
                println(io, "  Success rate: $(@sprintf("%.2f%%", agent_success_rate))")
                
                # Performance metrics with confidence intervals
                if :target_time in propertynames(agent_results)
                    successful_times = filter(isfinite, agent_results.target_time)
                    if !isempty(successful_times)
                        mean_time = mean(successful_times)
                        std_time = std(successful_times)
                        ci_95_time = 1.96 * std_time / sqrt(length(successful_times))
                        println(io, "  Average completion time: $(@sprintf("%.2f", mean_time)) (95% CI: $(@sprintf("%.2f", mean_time - ci_95_time)) - $(@sprintf("%.2f", mean_time + ci_95_time)))")
                    end
                end
                
                if :total_energy in propertynames(agent_results)
                    mean_energy = mean(agent_results.total_energy)
                    std_energy = std(agent_results.total_energy)
                    ci_95_energy = 1.96 * std_energy / sqrt(n_agent)
                    println(io, "  Average energy usage: $(@sprintf("%.3f", mean_energy)) (95% CI: $(@sprintf("%.3f", mean_energy - ci_95_energy)) - $(@sprintf("%.3f", mean_energy + ci_95_energy)))")
                end
                
                if :control_effort in propertynames(agent_results)
                    mean_effort = mean(agent_results.control_effort)
                    std_effort = std(agent_results.control_effort)
                    ci_95_effort = 1.96 * std_effort / sqrt(n_agent)
                    println(io, "  Average control effort: $(@sprintf("%.3f", mean_effort)) (95% CI: $(@sprintf("%.3f", mean_effort - ci_95_effort)) - $(@sprintf("%.3f", mean_effort + ci_95_effort)))")
                end
            end
            
            # Parameter analysis
            println(io, "\nParameter Analysis:")
            println(io, "-" ^ 20)
            
            for param in [:force, :friction]
                println(io, "\n$(titlecase(string(param))) Effect:")
                param_values = sort(unique(results[:, param]))
                println(io, "  Range: $(@sprintf("%.3f", minimum(param_values))) - $(@sprintf("%.3f", maximum(param_values)))")
                
                for agent in unique(results.agent_type)
                    agent_results = filter(r -> r.agent_type == agent, results)
                    best_param_idx = argmax([
                        mean(filter(r -> r[param] == val, agent_results).success)
                        for val in param_values
                    ])
                    best_param = param_values[best_param_idx]
                    success_at_best = mean(filter(r -> r[param] == best_param, agent_results).success) * 100
                    
                    println(io, "  $(titlecase(agent)) Agent optimal $(param): $(@sprintf("%.3f", best_param)) (Success rate: $(@sprintf("%.2f%%", success_at_best)))")
                end
            end
            
            # Success rate comparison
            if length(unique(results.agent_type)) == 2
                println(io, "\nAgent Performance Comparison:")
                println(io, "-" ^ 20)
                
                agent_types = collect(unique(results.agent_type))
                agent1_results = filter(r -> r.agent_type == agent_types[1], results)
                agent2_results = filter(r -> r.agent_type == agent_types[2], results)
                
                success_rate1 = mean(agent1_results.success) * 100
                success_rate2 = mean(agent2_results.success) * 100
                
                # Calculate confidence intervals for success rates
                n1 = nrow(agent1_results)
                n2 = nrow(agent2_results)
                ci_95_1 = 1.96 * sqrt(success_rate1 * (100 - success_rate1) / n1)
                ci_95_2 = 1.96 * sqrt(success_rate2 * (100 - success_rate2) / n2)
                
                println(io, "$(titlecase(agent_types[1])) success rate: $(@sprintf("%.2f%%", success_rate1)) (95% CI: $(@sprintf("%.2f%%", success_rate1 - ci_95_1)) - $(@sprintf("%.2f%%", success_rate1 + ci_95_1)))")
                println(io, "$(titlecase(agent_types[2])) success rate: $(@sprintf("%.2f%%", success_rate2)) (95% CI: $(@sprintf("%.2f%%", success_rate2 - ci_95_2)) - $(@sprintf("%.2f%%", success_rate2 + ci_95_2)))")
                
                diff = success_rate2 - success_rate1
                ci_95_diff = 1.96 * sqrt((ci_95_1/1.96)^2 + (ci_95_2/1.96)^2)
                
                println(io, "\nDifference ($(titlecase(agent_types[2])) - $(titlecase(agent_types[1]))):")
                println(io, "  $(@sprintf("%.2f%%", diff)) (95% CI: $(@sprintf("%.2f%%", diff - ci_95_diff)) - $(@sprintf("%.2f%%", diff + ci_95_diff)))")
                println(io, "  Relative improvement: $(@sprintf("%.2f%%", abs(diff) / success_rate1 * 100))")
                println(io, "  Better agent: $(diff > 0 ? titlecase(agent_types[2]) : titlecase(agent_types[1]))")
            end
        end
    catch e
        @error "Failed to generate summary report" exception=(e, catch_backtrace())
    end
end 