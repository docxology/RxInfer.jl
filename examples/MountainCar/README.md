# RxInfer Mountain Car Implementation and Meta-Analysis

This directory contains a comprehensive implementation of the Mountain Car environment with both standalone simulation capabilities and meta-analysis tools. The implementation features a comparison between Active Inference and Naive control strategies across various physics parameters.

## Project Structure

```
MountainCar/
├── MountainCar_Standalone_12-26-2024.jl  # Standalone simulation
├── MetaAnalysis_MountainCar.jl           # Meta-analysis main script
├── meta_analysis_simulation.jl           # Simulation batch handling
├── meta_analysis_utils.jl                # Analysis utilities
├── meta_analysis_visualization.jl        # Visualization module
├── visualization_functions.jl            # Detailed visualization functions
├── MountainCar.jl                       # Core environment implementation
├── MountainCar_Methods.jl               # Core methods and physics
├── shared_logging.jl                    # Logging utilities
├── config.toml                          # Configuration file
└── Project.toml                         # Project dependencies
```

## Features

### Standalone Implementation
- **Physics Engine**: Configurable physics with adjustable engine force and friction
- **Active Inference Agent**: 
  - Bayesian inference-based planning
  - Configurable planning horizon
  - Adaptive belief updates
  - Future state prediction
- **Naive Agent**: 
  - Momentum-based heuristic control
  - Direction-based decision making
- **Real-time Visualization**: 
  - Position and velocity tracking
  - Energy consumption monitoring
  - Control action visualization

### Meta-Analysis Framework
- **Parameter Space Exploration**:
  - Engine force range: ${config["meta_analysis"]["min_force"]} to ${config["meta_analysis"]["max_force"]}
  - Friction range: ${config["meta_analysis"]["min_friction"]} to ${config["meta_analysis"]["max_friction"]}
  - ${config["meta_analysis"]["force_steps"]} × ${config["meta_analysis"]["friction_steps"]} parameter combinations
  - ${config["simulation"]["n_episodes"]} episodes per combination

- **Comprehensive Metrics**:
  1. Success Rate Analysis
     - Parameter-dependent success rates
     - Agent performance comparison
     - Statistical significance testing
  
  2. Performance Metrics
     - Completion time statistics
     - Energy efficiency metrics
     - Control effort analysis
     - Stability measures
     - Oscillation patterns
  
  3. Trajectory Analysis
     - Position and velocity distributions
     - Phase space analysis
     - Energy landscape mapping
     - Control strategy characterization

- **Visualization Suite**:
  - Success rate heatmaps
  - Performance metric plots
  - Energy usage comparisons
  - Control effort visualizations
  - Parameter sweep analyses
  - Trajectory visualizations

## Setup and Usage

1. **Environment Setup**:
   ```julia
   # Run setup script
   julia Setup.jl
   ```
   This will:
   - Activate the project environment
   - Install required dependencies
   - Build RxInfer
   - Create necessary directories

2. **Running Standalone Simulation**:
   ```julia
   # Run standalone simulation
   julia --project=. MountainCar_Standalone_12-26-2024.jl
   ```
   This provides:
   - Single episode simulation
   - Real-time visualization
   - Detailed trajectory analysis

3. **Running Meta-Analysis**:
   ```julia
   # Run meta-analysis
   julia --project=. MetaAnalysis_MountainCar.jl
   ```
   This generates:
   - Parameter sweep results
   - Comparative analyses
   - Visualization outputs
   - Summary reports

## Configuration

### Standalone Configuration
- Initial state: position = ${config["initial_state"]["position"]}, velocity = ${config["initial_state"]["velocity"]}
- Target state: position = ${config["target_state"]["position"]}, velocity = ${config["target_state"]["velocity"]}
- Maximum steps: ${config["simulation"]["max_steps"]}
- Planning horizon: ${config["simulation"]["planning_horizon"]}

### Meta-Analysis Configuration
Edit `config.toml` to adjust:
- Parameter ranges
- Number of episodes
- Simulation parameters
- Visualization settings
- Output preferences

## Output Structure

Meta-analysis results are organized as follows:
```
meta_analysis_results/
└── TIMESTAMP/
    ├── success_rates/          # Success rate analyses
    ├── performance_metrics/    # Performance comparisons
    ├── energy_analysis/       # Energy usage studies
    ├── control_analysis/      # Control strategy analysis
    ├── parameter_analysis/    # Parameter effect studies
    ├── trajectory_analysis/   # Trajectory characteristics
    ├── raw_results.csv       # Raw simulation data
    ├── summary_report.txt    # Comprehensive summary
    └── config.toml           # Configuration snapshot
```

## Current Status

### Standalone Implementation
- ✅ Core physics engine
- ✅ Active Inference agent
- ✅ Naive control agent
- ✅ Naive control agent
- ✅ Real-time visualization
- ✅ Energy tracking
- ✅ Performance metrics

### Meta-Analysis Framework
- ✅ Parameter sweep functionality
- ✅ Batch simulation handling
- ✅ Success rate analysis
- ✅ Performance metrics
- ✅ Energy analysis
- ✅ Control analysis
- ✅ Trajectory analysis
- ✅ Visualization suite
- ✅ Summary reporting

## Dependencies

Key packages:
- RxInfer: Active Inference framework
- DataFrames: Data manipulation
- UnicodePlots: Terminal-based plotting
- Statistics: Statistical analysis
- Printf: Formatted output
- TOML: Configuration handling

## Contributing

To extend or modify:
1. Fork the repository
2. Create a feature branch
3. Add new metrics in `meta_analysis_utils.jl`
4. Add visualizations in `visualization_functions.jl`
5. Update configuration in `config.toml`
6. Submit a pull request

## License

This project is part of RxInfer.jl and is licensed under the same terms.

## Acknowledgments

This implementation builds on the RxInfer.jl framework and extends it with comprehensive meta-analysis capabilities for the Mountain Car environment.