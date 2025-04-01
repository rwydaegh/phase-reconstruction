import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def load_measurement_file(file_path):
    """Load a measurement pickle file and return the data"""
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def get_measurement_stats(data, file_path):
    """Extract and print statistics from measurement data"""
    results = data["results"]
    if isinstance(results, list):
        # Convert list to numpy array with proper NaN padding
        max_signal_length = max(len(signal) for signal in results)
        processed_results = []
        for signal in results:
            if len(signal) < max_signal_length:
                padded_signal = signal + [np.nan] * (max_signal_length - len(signal))
                processed_results.append(padded_signal)
            else:
                processed_results.append(signal)
        results = np.array(processed_results).astype(float)

    # Calculate statistics for non-NaN values
    non_nan_values = results[~np.isnan(results)]
    stats = {
        "min": np.min(non_nan_values),
        "max": np.max(non_nan_values),
        "mean": np.mean(non_nan_values),
        "median": np.median(non_nan_values),
        "std": np.std(non_nan_values),
        "nan_count": np.sum(np.isnan(results)),
        "total_points": results.size,
    }

    # Print statistics
    filename = os.path.basename(file_path)
    print(f"Statistics for {filename}:")
    print(f"  Measurement plane: x={extract_position_from_filename(filename)} mm")
    print(f"  Size: {results.shape}")
    print(f"  Range: {stats['min']:.6f} to {stats['max']:.6f} V/m")
    print(f"  Mean: {stats['mean']:.6f} V/m, Median: {stats['median']:.6f} V/m")
    print(f"  Standard deviation: {stats['std']:.6f} V/m")
    print(
        f"  Valid points: {stats['total_points'] - stats['nan_count']} "
        f"out of {stats['total_points']} ({100*(1-stats['nan_count']/stats['total_points']):.2f}%)"
    )

    continuous_axis = data["continuous_axis"]
    discrete_axis = data["discrete_axis"]
    points_continuous = data["points_continuous"]
    points_discrete = data["points_discrete"]

    if isinstance(points_discrete, list):
        points_discrete = np.array(points_discrete)

    print(
        f"  {continuous_axis}-axis: {len(points_continuous)} points from "
        f"{np.min(points_continuous):.1f} to {np.max(points_continuous):.1f} mm"
    )
    print(
        f"  {discrete_axis}-axis: {len(points_discrete)} points from "
        f"{np.min(points_discrete):.1f} to {np.max(points_discrete):.1f} mm"
    )

    if "frequency" in data:
        print(
            f"  Frequency: {data['frequency']/1e9:.6f} GHz "
            f"(wavelength: {299792458/(data['frequency']/1e9)/1e6:.6f} mm)"
        )

    return stats, results, data


def extract_position_from_filename(filename):
    """Extract the position value from the measurement filename"""
    # Example: "x400_zy.pickle" -> 400
    parts = filename.split("_")[0]  # "x400"
    return parts[1:]  # "400"


def plot_measurement(
    data, file_path, in_log=False, method="contourf", output_dir="measurement_plots"
):
    """Create a visualization of the measurement data"""
    results = data["results"]
    if isinstance(results, list):
        # Convert list to numpy array with proper NaN padding
        max_signal_length = max(len(signal) for signal in results)
        processed_results = []
        for signal in results:
            if len(signal) < max_signal_length:
                padded_signal = signal + [np.nan] * (max_signal_length - len(signal))
                processed_results.append(padded_signal)
            else:
                processed_results.append(signal)
        results = np.array(processed_results).astype(float)

    continuous_axis = data["continuous_axis"]
    discrete_axis = data["discrete_axis"]
    points_continuous = data["points_continuous"]
    points_discrete = data["points_discrete"]

    if isinstance(points_discrete, list):
        points_discrete = np.array(points_discrete)

    cc, dd = np.meshgrid(points_continuous, points_discrete)

    # Get frequency for title
    frequency = data.get("frequency", 28e9)  # Default to 28 GHz if not provided

    # Process data for plotting
    if in_log:
        # Add a small value to avoid log(0)
        min_positive = np.min(results[results > 0]) if np.any(results > 0) else 1e-10
        results_to_plot = np.log10(np.maximum(results, min_positive))
        plot_title = f"Electric field magnitude (log scale) at {frequency/1e9:.2f} GHz"
        cbar_label = "log10(E-field) [V/m]"
    else:
        results_to_plot = results
        plot_title = f"Electric field magnitude at {frequency/1e9:.2f} GHz"
        cbar_label = "E-field [V/m]"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract base filename without extension
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    # Create filename for output
    suffix = "_log" if in_log else ""
    output_filename = f"{output_dir}/{base_name}_{method}{suffix}.png"

    plt.figure(figsize=(12, 9))

    if method == "contourf":
        plt.contourf(cc, dd, results_to_plot, cmap="jet")
    elif method == "imshow":
        plt.imshow(
            results_to_plot,
            cmap="jet",
            aspect="auto",
            extent=[
                np.min(points_continuous),
                np.max(points_continuous),
                np.min(points_discrete),
                np.max(points_discrete),
            ],
        )
    elif method == "pcolor":
        plt.pcolor(cc, dd, results_to_plot, cmap="jet")

    cbar = plt.colorbar()
    cbar.set_label(cbar_label)

    plt.xlabel(f"{continuous_axis} [mm]")
    plt.ylabel(f"{discrete_axis} [mm]")
    plt.xlim([np.min(points_continuous), np.max(points_continuous)])
    plt.ylim([np.min(points_discrete), np.max(points_discrete)])

    # Add X position to title
    x_pos = extract_position_from_filename(os.path.basename(file_path))
    plt.title(f"{plot_title} - x={x_pos} mm, {continuous_axis}{discrete_axis} plane")

    # Add statistics as text
    non_nan_values = results[~np.isnan(results)]
    plt.figtext(
        0.02,
        0.02,
        f"Min: {np.min(non_nan_values):.4f}, Max: {np.max(non_nan_values):.4f}, "
        f"Mean: {np.mean(non_nan_values):.4f} V/m",
        bbox={"facecolor": "white", "alpha": 0.8},
    )

    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved as '{output_filename}'")

    return output_filename


def plot_3d_slice_positions(measurement_files, output_dir="measurement_plots"):
    """Create a 3D visualization of where the measurement slices are located"""
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection="3d")

    # Colors for different planes
    colors = ["r", "g", "b", "c", "m", "y"]

    # Create a legend mapping
    legend_handles = []
    legend_labels = []

    # Process each measurement file
    for i, file_path in enumerate(measurement_files):
        data = load_measurement_file(file_path)

        # Extract position and axes
        filename = os.path.basename(file_path)
        if filename.startswith("x"):
            # x-constant plane (zy plane)
            x_pos = float(extract_position_from_filename(filename))
            z_vals = data["points_continuous"]
            y_vals = (
                data["points_discrete"]
                if isinstance(data["points_discrete"], np.ndarray)
                else np.array(data["points_discrete"])
            )

            # Create a grid of points
            z_grid, y_grid = np.meshgrid(z_vals, y_vals)
            x_grid = np.full_like(z_grid, x_pos)

            # Plot a subset of points to avoid overcrowding
            stride = max(1, len(z_vals) // 50)
            ax.scatter(
                x_grid[::stride, ::stride],
                y_grid[::stride, ::stride],
                z_grid[::stride, ::stride],
                color=colors[i % len(colors)],
                alpha=0.5,
                s=10,
            )

            # Add a wireframe outline of the plane
            ax.plot_wireframe(
                np.array([[x_pos, x_pos], [x_pos, x_pos]]),
                np.array([[np.min(y_vals), np.min(y_vals)], [np.max(y_vals), np.max(y_vals)]]),
                np.array([[np.min(z_vals), np.max(z_vals)], [np.min(z_vals), np.max(z_vals)]]),
                color=colors[i % len(colors)],
            )

            legend_handles.append(
                plt.Line2D([0], [0], linestyle="none", marker="o", color=colors[i % len(colors)])
            )
            legend_labels.append(f"x={x_pos}mm (zy plane)")

        elif filename.startswith("y"):
            # y-constant plane (zx plane)
            y_pos = float(extract_position_from_filename(filename))
            z_vals = data["points_continuous"]
            x_vals = (
                data["points_discrete"]
                if isinstance(data["points_discrete"], np.ndarray)
                else np.array(data["points_discrete"])
            )

            # Create a grid of points
            z_grid, x_grid = np.meshgrid(z_vals, x_vals)
            y_grid = np.full_like(z_grid, y_pos)

            # Plot a subset of points to avoid overcrowding
            stride = max(1, len(z_vals) // 50)
            ax.scatter(
                x_grid[::stride, ::stride],
                y_grid[::stride, ::stride],
                z_grid[::stride, ::stride],
                color=colors[i % len(colors)],
                alpha=0.5,
                s=10,
            )

            # Add a wireframe outline of the plane
            ax.plot_wireframe(
                np.array([[np.min(x_vals), np.max(x_vals)], [np.min(x_vals), np.max(x_vals)]]),
                np.array([[y_pos, y_pos], [y_pos, y_pos]]),
                np.array([[np.min(z_vals), np.min(z_vals)], [np.max(z_vals), np.max(z_vals)]]),
                color=colors[i % len(colors)],
            )

            legend_handles.append(
                plt.Line2D([0], [0], linestyle="none", marker="o", color=colors[i % len(colors)])
            )
            legend_labels.append(f"y={y_pos}mm (zx plane)")

    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.set_title("3D Visualization of Measurement Planes")
    ax.legend(legend_handles, legend_labels)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{output_dir}/measurement_planes_3d.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"3D visualization saved as '{output_filename}'")


if __name__ == "__main__":
    measurement_dir = "measurement_data"

    # Find all pickle files in the measurement directory
    measurement_files = [
        os.path.join(measurement_dir, f)
        for f in os.listdir(measurement_dir)
        if f.endswith(".pickle")
    ]

    if not measurement_files:
        print(f"No measurement files found in {measurement_dir}")
        exit(1)

    print(f"Found {len(measurement_files)} measurement files:")
    for f in measurement_files:
        print(f"  - {os.path.basename(f)}")
    print()

    # Create a directory for output plots
    # output_dir = "measurement_analysis" # replaced with hydra config
    output_dir = os.path.join(os.getcwd(), "measurement_analysis")
    os.makedirs(output_dir, exist_ok=True)

    # Process each measurement file
    stats_dict = {}
    for file_path in measurement_files:
        print(f"\nProcessing {os.path.basename(file_path)}...")
        print("-" * 50)

        # Load and analyze data
        data = load_measurement_file(file_path)
        stats, results, processed_data = get_measurement_stats(data, file_path)
        stats_dict[os.path.basename(file_path)] = stats

        # Generate plots
        plot_measurement(data, file_path, in_log=False, method="contourf", output_dir=output_dir)
        plot_measurement(data, file_path, in_log=True, method="contourf", output_dir=output_dir)
        plot_measurement(data, file_path, in_log=False, method="pcolor", output_dir=output_dir)

    # Create a 3D visualization of plane positions
    print("\nGenerating 3D visualization of measurement planes...")
    plot_3d_slice_positions(measurement_files, output_dir=output_dir)

    # Create a summary table of all measurements
    print("\nGenerating summary table...")

    # Create a figure for the summary table
    plt.figure(figsize=(12, len(measurement_files) * 0.6 + 1))
    plt.axis("off")

    # Create table data
    file_names = list(stats_dict.keys())
    table_data = []
    for fname in file_names:
        stats = stats_dict[fname]
        table_data.append(
            [
                fname,
                f"{stats['min']:.4f}",
                f"{stats['max']:.4f}",
                f"{stats['mean']:.4f}",
                f"{stats['median']:.4f}",
                f"{stats['std']:.4f}",
                f"{100*(1-stats['nan_count']/stats['total_points']):.1f}%",
            ]
        )

    # Create the table
    column_labels = [
        "File",
        "Min (V/m)",
        "Max (V/m)",
        "Mean (V/m)",
        "Median (V/m)",
        "Std Dev (V/m)",
        "Valid Data",
    ]
    table = plt.table(cellText=table_data, colLabels=column_labels, loc="center", cellLoc="center")

    # Configure the table appearance
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Save the table
    plt.savefig(f"{output_dir}/measurement_summary_table.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Summary table saved as '{output_dir}/measurement_summary_table.png'")

    print("\nAnalysis complete. Results saved in the '{output_dir}' directory.")
