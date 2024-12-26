# Mountain Car Meta-Analysis

This directory contains the implementation of a meta-analysis comparing the performance of Active Inference and Naive agents in the Mountain Car environment across different physics parameters.

## Overview

The meta-analysis explores how different combinations of engine force and friction coefficients affect the performance of two types of agents:
1. **Active Inference Agent**: Uses Bayesian inference to plan actions based on a generative model of the environment
2. **Naive Agent**: Uses a simple heuristic strategy (push in direction of motion)

## Files

- `MetaAnalysis_MountainCar.jl`: Main script for running the meta-analysis
- `meta_analysis_simulation.jl`: Module for running batches of simulations
- `meta_analysis_utils.jl`: Utility functions for metrics calculation
- `visualization_functions.jl`: Functions for generating visualizations and analyses
- `MountainCar.jl`: Implementation of the Mountain Car environment and agents
- `config.toml`: Configuration file for meta-analysis parameters

## Running the Analysis

1. Ensure you have Julia installed and the required packages:
   ```julia
   using Pkg
   Pkg.activate(".")
   Pkg.instantiate()
   ```

2. Run the meta-analysis:
   ```julia
   include("MetaAnalysis_MountainCar.jl")
   ```

3. Results will be saved in the `meta_analysis_results` directory with a timestamp.

## Analysis Outputs

The meta-analysis generates several types of analyses:

1. **Success Rate Analysis**
   - Heatmaps showing success rates for different parameter combinations
   - Comparison between agent types

2. **Performance Metrics**
   - Success rate
   - Average completion time
   - Energy usage
   - Control efficiency
   - Stability
   - Oscillations

3. **Energy Analysis**
   - Total energy distribution
   - Energy vs Success Rate
   - Energy vs Completion Time
   - Energy Efficiency

4. **Control Analysis**
   - Control effort distribution
   - Stability metrics
   - Oscillation patterns
   - Control strategy comparison

5. **Parameter Analysis**
   - Effect of force on success rate
   - Effect of friction on success rate
   - Best parameter combinations

6. **Trajectory Analysis**
   - Position distributions
   - Velocity patterns
   - Success characteristics

## Configuration

The `config.toml` file allows customization of:
- Number of episodes per parameter combination
- Maximum steps per episode
- Initial and target states
- Force and friction parameter ranges
- Visualization settings

## Metrics

The analysis calculates several metrics:

1. **Success Rate**: Percentage of episodes where the agent reaches the target
2. **Completion Time**: Steps taken to reach the target
3. **Energy Usage**: Total energy consumed during the episode
4. **Control Effort**: Sum of absolute control actions
5. **Stability**: Variance of position around target
6. **Oscillations**: Number of direction changes
7. **Efficiency**: Combined metric of success, time, and energy usage

## Results Format

Results are saved in text files with detailed analyses and plots:
- Raw data in CSV format
- Success rate matrices
- Performance comparisons
- Energy usage analysis
- Control strategy analysis
- Parameter sweep results
- Trajectory characteristics
- Summary report with key findings

## Extending the Analysis

To modify or extend the analysis:
1. Adjust parameters in `config.toml`
2. Add new metrics in `meta_analysis_utils.jl`
3. Create new visualization functions in `visualization_functions.jl`
4. Modify the main script to include additional analyses