# Mountain Car Active Inference

This example demonstrates Active Inference in action using the classic mountain car problem. The car must escape a valley by building up momentum through oscillatory movements, as direct approach with limited engine power is impossible.

The implementation is based on the Active Inference formalism, where the agent:
- Maintains beliefs about its position and velocity (hidden states)
- Has preferences about desired states (being at the target position)
- Performs inference to estimate current states
- Selects actions that minimize expected free energy

## Key Files

- `MountainCar.jl`: Main simulation script implementing:
  - Physics engine with configurable parameters (engine force, friction)
  - State transition dynamics
  - Reward/cost functions
  - Simulation loop and trajectory tracking

- `MountainCar_Methods.jl`: Core Active Inference implementation with:
  - Generative model definition
  - Variational free energy computation
  - State estimation via gradient descent
  - Action selection through expected free energy minimization
  - Belief updating using message passing

- `visualization.jl`: Visualization utilities for:
  - Mountain valley landscape plotting
  - Car position and trajectory animation
  - State and belief evolution plots

- `Setup.jl`: Project setup and dependencies

## Running the Example