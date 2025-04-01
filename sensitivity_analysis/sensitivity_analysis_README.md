# Field Reconstruction Sensitivity Analysis

This module provides tools for analyzing the sensitivity of the electromagnetic field reconstruction algorithm to various simulation parameters. It systematically varies pairs of parameters across specified ranges and measures their impact on reconstruction accuracy and performance.

## Overview

The sensitivity analysis framework allows you to:

1.  Define ranges for multiple simulation parameters (e.g., `wall_points`, `num_sources`, `resolution`).
2.  Automatically run simulations for combinations of parameter pairs.
3.  Measure key performance metrics like Normalized RMSE and Correlation for each simulation.
4.  Optionally run simulations in parallel to speed up execution.
5.  Generate visualizations (heatmaps, quality maps) summarizing the results.
6.  Save raw numerical data for further custom analysis.

The analysis is configured and executed using the [Hydra](https://hydra.cc/) framework.

## Files

-   `sensitivity_analysis.py`: Core implementation of the sensitivity analysis logic (parameter sweeping, simulation execution, metric calculation).
-   `run_sensitivity_analysis.py`: The main script to launch the sensitivity analysis using Hydra.
-   `improved_sensitivity_visualization.py`: Generates enhanced plots from the saved analysis results.
-   `conf/sensitivity_analysis.yaml`: Hydra configuration file defining parameter ranges, execution settings, and the base simulation setup.
-   `conf/simulated_data.yaml` / `conf/measured_data.yaml`: Base configuration files for the underlying simulation, selected within `conf/sensitivity_analysis.yaml`.

## Usage

### Running the Analysis

To run the sensitivity analysis, execute the `run_sensitivity_analysis.py` script as a module from the project's root directory (`/workspaces/phase-reconstruction`):

```bash
python -m sensitivity_analysis.run_sensitivity_analysis
```

This command uses Hydra to:

1.  Load the configuration from `conf/sensitivity_analysis.yaml`.
2.  Compose the final configuration by merging defaults (like the base simulation settings).
3.  Execute the analysis defined in `run_sensitivity_analysis.py`.
4.  Save all outputs (logs, `.npz` data files, plots) into a timestamped directory specified by `hydra.run.dir` in the configuration (default: `outputs/sensitivity/YYYY-MM-DD_HH-MM-SS/`).

### Configuration

The primary way to configure the analysis is by editing `conf/sensitivity_analysis.yaml`:

-   **`defaults`**: Choose the base simulation configuration (e.g., `simulated_data` or `measured_data`).
-   **`parallel` / `max_workers`**: Control parallel execution (set `parallel: false` for sequential runs).
-   **`parameter_ranges`**: This list defines which parameters are varied. Each entry specifies:
    -   `param_name`: The name of the parameter (must match a key in the base simulation config).
    -   `start`, `end`: The range of values to test.
    -   `num_steps`: How many values to test within the range.
    -   `log_scale`: Whether to use a logarithmic scale for the range (True/False).

    ```yaml
    # Example parameter range definition in conf/sensitivity_analysis.yaml
    parameter_ranges:
      - param_name: "resolution"
        start: 10
        end: 60
        num_steps: 4
        log_scale: False
      - param_name: "num_sources"
        start: 10
        end: 200
        num_steps: 5
        log_scale: True
      # Add or comment out parameters as needed
      # - param_name: "gs_iterations"
      #   ...
    ```

    *Note: The analysis runs simulations for every pair of parameters defined in this list.*

-   **`hydra.run.dir`**: Controls the output directory pattern.

### Command-Line Overrides

You can temporarily override configuration values directly from the command line using Hydra's syntax:

```bash
# Disable parallel execution for this run
python -m sensitivity_analysis.run_sensitivity_analysis parallel=False

# Change the number of steps for a specific parameter (if defined in the YAML)
# Note: This syntax for overriding list elements can be complex. Editing the YAML is often easier.
# python -m sensitivity_analysis.run_sensitivity_analysis parameter_ranges.0.num_steps=5

# Override a parameter from the base simulation configuration
python -m sensitivity_analysis.run_sensitivity_analysis base_simulation.gs_iterations=150
```

## Output

For each pair of analyzed parameters (e.g., `param1` vs `param2`), the analysis generates the following files in the run-specific output directory:

1.  **`sensitivity_{param1}_vs_{param2}_data.npz`**: A NumPy data file containing:
    -   `X`, `Y`: Meshgrids of the parameter values.
    -   `rmse`: Grid of Normalized RMSE values.
    -   `correlation`: Grid of Correlation values.
    -   `time`: Grid of computation times (in seconds).
    -   `param1_name`, `param2_name`: Names of the parameters.
2.  **`enhanced_{param1}_vs_{param2}_rmse.png`**: An enhanced heatmap visualizing the RMSE results, highlighting outliers.
3.  **`quality_{param1}_vs_{param2}.png`**: A categorical map visualizing reconstruction quality based on correlation thresholds.
4.  **`enhanced_sensitivity_summary.png`**: A summary plot showing the RMSE heatmaps for all analyzed parameter pairs side-by-side.

## Interpreting Results

-   **RMSE Plots**: Lower RMSE values (often blue/purple regions in default colormaps) indicate better reconstruction accuracy. Look for regions where RMSE is consistently low. Outliers (marked in red) indicate specific parameter combinations that performed poorly.
-   **Quality Maps**: "Excellent" or "Very Good" regions (often green/yellow) indicate high correlation between the true and reconstructed fields, signifying better reconstruction quality.
-   **Summary Plot**: Provides a quick comparison of how different parameter pairs affect the RMSE landscape.
-   **Gradients/Plateaus**: Observe how metrics change with parameters. Steep gradients indicate high sensitivity, while flat plateaus might suggest diminishing returns from increasing a parameter further.

## Advanced Analysis

The saved `.npz` files allow for further custom analysis:

```python
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
# Replace with the actual path to your output directory
output_dir = 'outputs/sensitivity/YYYY-MM-DD_HH-MM-SS/'
param1 = 'wall_points'
param2 = 'num_sources'
# --- End Configuration ---

data_filename = f'sensitivity_{param1}_vs_{param2}_data.npz'
data_path = os.path.join(output_dir, data_filename)

if not os.path.exists(data_path):
    print(f"Error: Data file not found at {data_path}")
else:
    data = np.load(data_path)

    # Access the data
    X = data['X']  # Parameter 1 values (meshgrid)
    Y = data['Y']  # Parameter 2 values (meshgrid)
    rmse = data['rmse']
    correlation = data['correlation']
    time = data['time']
    p1_name = str(data['param1_name']) # Use loaded names
    p2_name = str(data['param2_name']) # Use loaded names

    print(f"Loaded data for: {p1_name} vs {p2_name}")
    print(f"RMSE shape: {rmse.shape}")

    # Example: Create a custom contour plot for RMSE
    plt.figure(figsize=(10, 8))
    levels = np.logspace(np.log10(np.nanmin(rmse)), np.log10(np.nanmax(rmse)), 10) # Log levels
    CS = plt.contourf(X, Y, rmse, levels=levels, cmap='viridis', norm=plt.matplotlib.colors.LogNorm())
    plt.contour(CS, colors='k', linewidths=0.5) # Add contour lines

    plt.xlabel(p1_name)
    plt.ylabel(p2_name)
    if data.get('log_scale_x', False): # Check if log scale was used (optional: save this info in npz)
         plt.xscale('log')
    if data.get('log_scale_y', False):
         plt.yscale('log')

    plt.colorbar(CS, label='Normalized RMSE (log scale)')
    plt.title(f'Custom RMSE Contour Plot: {p1_name} vs {p2_name}')
    plt.savefig(os.path.join(output_dir, f'custom_rmse_contour_{p1_name}_vs_{p2_name}.png'))
    plt.show()