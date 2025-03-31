# Field Reconstruction Sensitivity Analysis

This module provides tools for analyzing the sensitivity of electromagnetic field reconstruction to various parameters. The analysis systematically varies pairs of parameters and measures how they affect reconstruction accuracy.

## Overview

The sensitivity analysis framework allows you to:

1. Specify ranges for multiple parameters
2. Analyze how pairs of parameters jointly affect reconstruction quality
3. Visualize the results as heatmaps
4. Run simulations in parallel for faster execution

## Files

- `sensitivity_analysis.py`: Core implementation of the sensitivity analysis framework
- `run_sensitivity_analysis.py`: Command-line interface for running sensitivity analysis

## Usage

### Basic Usage

To run a sensitivity analysis with default parameters:

```bash
python run_sensitivity_analysis.py
```

This will create a directory called `sensitivity_results` containing heatmaps for each parameter pair.

### Advanced Usage

You can customize the analysis with command-line arguments:

```bash
python run_sensitivity_analysis.py --output-dir custom_results --parallel --max-workers 8 --resolution 40 --gs-iterations 150
```

Arguments:
- `--output-dir`: Directory to save results (default: "sensitivity_results")
- `--parallel`: Enable parallel processing (recommended for faster execution)
- `--max-workers`: Number of parallel workers (default: 4)
- `--resolution`: Base resolution for the simulation (default: 30)
- `--gs-iterations`: Base number of Gerchberg-Saxton iterations (default: 100)

### Custom Parameter Ranges

To customize which parameters are analyzed and their ranges, edit the `parameter_ranges` list in `run_sensitivity_analysis.py`.

Example of adding a new parameter range:

```python
# Wavelength variation
ParameterRange(
    param_name="wavelength",
    start=5e-3,
    end=15e-3,
    num_steps=5,
    log_scale=False
)
```

## Parameter Descriptions

The following parameters can be analyzed:

- `wall_points`: Number of points per wall edge, controls wall discretization
- `num_sources`: Number of active sources randomly placed on walls
- `resolution`: Grid resolution of the measurement plane
- `gs_iterations`: Maximum iterations for the Gerchberg-Saxton algorithm
- `convergence_threshold`: Threshold for determining GS algorithm convergence
- `wavelength`: EM wavelength (28GHz = 10.7mm)
- `room_size`: Size of the cubic room
- `plane_size`: Size of the measurement plane

## Output

The analysis generates the following outputs for each parameter pair:

1. **RMSE Heatmap**: Shows how reconstruction error varies with parameter values
2. **Correlation Heatmap**: Shows how reconstruction correlation varies
3. **Time Heatmap**: Shows computation time for different parameter values
4. **NPZ Data File**: Raw numerical data for further analysis
5. **Summary Plot**: Overview of all parameter pair results

## Example Results Interpretation

When analyzing the heatmaps:

- **Low RMSE (blue regions)**: Parameter combinations that yield better reconstruction
- **High correlation (yellow/white regions)**: Parameter combinations with better reconstruction
- **Gradients**: Areas where small parameter changes significantly affect results
- **Plateaus**: Regions where increasing parameters provides diminishing returns

## Advanced Analysis

For further analysis, you can load the saved NPZ files:

```python
import numpy as np
import matplotlib.pyplot as plt

# Load results
data = np.load('sensitivity_results/sensitivity_wall_points_vs_num_sources_data.npz')

# Access the data
X = data['X']  # Parameter 1 values (2D grid)
Y = data['Y']  # Parameter 2 values (2D grid)
rmse = data['rmse']  # RMSE values
correlation = data['correlation']  # Correlation values

# Custom visualization or further processing
plt.figure(figsize=(10, 8))
plt.contour(X, Y, rmse, levels=10, cmap='viridis')
plt.xlabel(str(data['param1_name']))
plt.ylabel(str(data['param2_name']))
plt.colorbar(label='RMSE')
plt.title('Contour Plot of Reconstruction Error')
plt.savefig('custom_contour.png')