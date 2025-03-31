# Phase Reconstruction

A Python framework for electromagnetic field reconstruction using holographic phase retrieval methods.

## Overview

This project implements advanced electromagnetic field reconstruction techniques using holographic phase retrieval algorithms (particularly Gerchberg-Saxton) with enhanced features like adaptive regularization, perturbation strategies, and momentum-based optimization. It allows scientists and engineers to reconstruct source current distributions from field magnitude measurements.

## Features

- **Holographic Phase Retrieval**: Implementation of the Gerchberg-Saxton algorithm with multiple enhancements
- **Perturbation Strategies**: Improved convergence by escaping local minima
- **Visualization Tools**: Animations and plots showing field reconstruction process
- **Sensitivity Analysis**: Tools to analyze algorithm sensitivity to parameters
- **Performance Optimizations**: Profiling and benchmarking for computational efficiency

## Project Structure

```
.
├── src/                           # Core implementation
│   ├── config_types.py            # Configuration dataclasses
│   ├── create_channel_matrix.py   # Channel matrix generation
│   ├── create_test_pointcloud.py  # Synthetic data generation
│   ├── holographic_phase_retrieval.py  # Main phase retrieval algorithm
│   ├── perturbation_analysis.py   # Perturbation analysis utilities
│   ├── simulation_config_*.py     # Configuration files
│   ├── visualization.py           # Visualization utilities
│   └── utils/                     # Utility functions
├── measurement_data/              # Stored measurement data
├── papers/                        # Academic papers
├── figs/                          # Generated figures
├── sensitivity_analysis/          # Parameter sensitivity analysis
├── perturbation_analysis/         # Perturbation effect analysis
├── profiling/                     # Performance profiling tools
├── main.py                        # Main script entry point
└── README.md                      # This documentation
```

## Getting Started

### Prerequisites

- Python 3.8+
- NumPy
- Matplotlib
- Pickle (for saving/loading data)

### Installation

Clone the repository:

```bash
git clone https://github.ugent.be/rwydaegh/phase-reconstruction.git
cd phase-reconstruction
```

### Usage

Run the main script with default parameters:

```bash
python main.py
```

Or customize parameters:

```bash
python main.py --resolution 100 --gs_iterations 300 --enable_perturbations True
```

Run sensitivity analysis:

```bash
python sensitivity_analysis/run_sensitivity_analysis.py
```

## Documentation

### Key Concepts

- **Phase Retrieval**: The process of recovering phase information from magnitude-only measurements
- **Channel Matrix**: Maps source currents to field measurements
- **Perturbation Strategies**: Techniques to escape local minima in the optimization
- **Digital Twin Reconstruction**: Two approaches are implemented:
  1. Normal Reconstruction (Physically Accurate) - Uses perturbed geometry with fixed original currents
  2. Current Optimization (Measurement-Matching) - Adjusts currents to match observed fields despite geometric errors

### Configuration Parameters

The `SimulationConfig` class provides configurable parameters including:

- `resolution`: Resolution of the measurement plane
- `gs_iterations`: Number of Gerchberg-Saxton iterations
- `enable_perturbations`: Whether to use perturbation strategies
- `perturbation_intensity`: Strength of perturbations
- `convergence_threshold`: Threshold for determining convergence

## Results

The algorithm generates several output files:

- `results.png`: Final field comparison 
- `gs_animation.gif`: Animation of the GS iteration process
- `current_field_animation.gif`: Animation of current distribution and field reconstruction

## Findings

As documented in `crucial_realization.md`, the timing of perturbations relative to magnitude constraints is critical for algorithm effectiveness. The project also explores trade-offs between physical accuracy and field prediction accuracy in digital twin simulations.

## License

[Specify appropriate license]

## Contact

[Your contact information]
