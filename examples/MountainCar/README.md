# Mountain Car Active Inference

This example demonstrates Active Inference in action using the classic mountain car problem. The car must escape a valley by building up momentum through oscillatory movements, as direct approach with limited engine power is impossible.

## Overview

The implementation is based on the Active Inference formalism, where the agent:
- Maintains beliefs about its position and velocity (hidden states)
- Has preferences about desired states (being at the target position)
- Performs inference to estimate current states
- Selects actions that minimize expected free energy

## Implementation Status

### Core Components
- ✅ Physics engine with configurable parameters
- ✅ Active Inference agent implementation
- ✅ Naive approach for comparison
- ✅ Visualization system with ghosted predictions
- ✅ Parameter space exploration and analysis
- ✅ Multi-threaded performance analysis
- ✅ Configuration management via TOML

### Key Features
- Dynamic force application with configurable limits
- Friction and gravity physics simulation
- Belief state tracking and updating
- Expected free energy minimization
- Parallel parameter space exploration
- Comprehensive statistical analysis
- Rich visualization suite

## Project Structure

### Key Files

- `MountainCar.jl`: Main simulation runner
  - Implements both naive and Active Inference approaches
  - Handles visualization and analysis
  - Supports ghosted prediction visualization
  - Normalizes actions for fair comparison

- `MountainCar_Methods.jl`: Core Active Inference implementation
  - Physics engine with configurable parameters
  - State transition dynamics and generative model
  - Belief updating using message passing
  - Action selection through expected free energy minimization
  - Agent creation and world simulation functions

- `MultiThread_MetaAnalysis_MountainCar.jl`: Meta-analysis framework
  - Parallel parameter space exploration
  - Statistical analysis of performance metrics
  - Success rate and completion time comparisons
  - Visualization of results through heatmaps and plots
  - Generation of detailed analysis reports

- `visualization.jl`: Visualization utilities
  - Mountain valley landscape plotting
  - Car position and trajectory animation
  - State and belief evolution plots
  - Performance comparison visualizations
  - Ghosted prediction visualization

- `config.toml`: Configuration management
  - Physics parameters (engine force, friction)
  - Initial and target states
  - Simulation settings
  - Visualization parameters

- `Setup.jl`: Environment setup
  - Package dependency management
  - Project environment initialization
  - Required statistical packages
  - Visualization dependencies

### Output Files

The simulation generates several output files in the `Outputs/` directory:
- Environment visualization
- Naive approach summary and animation
- Active Inference summary and animation
- Performance comparison visualizations
- Statistical analysis reports
- Parameter space exploration results
- Best parameters configuration

## Running the Example

1. Ensure Julia 1.6+ is installed on your system

2. Clone the RxInfer repository and navigate to this example:
   ```bash
   git clone https://github.com/biaslab/RxInfer.jl.git
   cd RxInfer.jl/examples/MountainCar
   ```

3. Initialize the environment:
   ```bash
   julia Setup.jl
   ```

4. Run the main simulation:
   ```bash
   julia MountainCar.jl
   ```

5. For parameter space exploration:
   ```bash
   # Single-threaded version
   julia MetaAnalysis_MountainCar.jl
   
   # Multi-threaded version (recommended)
   JULIA_NUM_THREADS=4 julia MultiThread_MetaAnalysis_MountainCar.jl
   ```

## Current Performance

### Comparison: Naive vs Active Inference
- The Active Inference agent consistently outperforms the naive approach
- Both approaches have equal force capabilities (±2.0 force units)
- Active Inference shows more strategic oscillatory behavior
- Ghosted predictions demonstrate forward planning capability

### Parameter Space Exploration
- Engine force range: 0.01 to 0.05
- Friction coefficient range: 0.01 to 0.05
- Comprehensive analysis of success rates and completion times
- Statistical significance testing of performance differences
- Visualization of parameter space through heatmaps

### Visualization Improvements
- Enhanced trajectory visualization
- Ghosted predictions display
- State space exploration plots
- Performance comparison animations
- Statistical analysis plots

## Configuration

The `config.toml` file allows customization of:
- Physics parameters
  - Engine force limits
  - Friction coefficients
- Initial conditions
  - Starting position
  - Initial velocity
- Target state
  - Goal position
  - Desired velocity
- Simulation parameters
  - Number of timesteps
  - Planning horizon
- Visualization settings
  - Plot limits
  - Animation parameters

## Analysis Tools

The meta-analysis framework provides:
- Parameter space exploration
- Statistical significance testing
- Performance metrics computation
- Success rate analysis
- Completion time statistics
- Visualization generation
- Detailed report generation

## Future Improvements

Potential areas for enhancement:
- Extended parameter space exploration
- Additional performance metrics
- Real-time visualization options
- Alternative control strategies
- Enhanced prediction visualization
- Performance optimization