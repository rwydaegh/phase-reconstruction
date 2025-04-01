import pickle

import matplotlib.pyplot as plt
import numpy as np


def examine_pickle_file(file_path):
    """
    Load and examine the content of a pickle file
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    # Print the keys in the dictionary
    print("Keys in the pickle file:", list(data.keys()))

    # Print info about each key
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: numpy array with shape {value.shape} and dtype {value.dtype}")
            if key == "results":
                non_nan_values = value[~np.isnan(value)]
                print(f"  Value range: {np.min(non_nan_values)} to {np.max(non_nan_values)}")
                print(f"  Mean value: {np.mean(non_nan_values)}")
                print(f"  Median value: {np.median(non_nan_values)}")
                print(f"  NaN count: {np.sum(np.isnan(value))}")
            elif key == "points_continuous" or key == "points_discrete":
                print(f"  Range: {np.min(value)} to {np.max(value)} mm")
                print(f"  Number of points: {value.shape[0]}")
                spacing = (
                    (np.max(value) - np.min(value)) / (value.shape[0] - 1)
                    if value.shape[0] > 1
                    else 0
                )
                print(f"  Spacing: {spacing} mm (average)")
        elif isinstance(value, list):
            print(f"{key}: list with length {len(value)}")
            if len(value) > 0:
                print(f"  First element type: {type(value[0])}")
                if isinstance(value[0], list):
                    print(f"  First element length: {len(value[0])}")
        else:
            print(f"{key}: {type(value)} - {value}")
            if key == "frequency":
                print(f"  Frequency in GHz: {value / 1e9} GHz")

    # If the data has the expected structure, plot it
    if (
        "results" in data
        and "continuous_axis" in data
        and "discrete_axis" in data
        and "points_continuous" in data
        and "points_discrete" in data
    ):
        plot_data(data)

    return data


def plot_data(data, method="contourf", in_log=False, number_of_levels=None, filename=None):
    """
    Plot the data similarly to the provided plot_from_pickle function
    """
    results = data["results"]
    if isinstance(results, list):
        # Handle variable length lists by padding with np.nan
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
    frequency = data.get("frequency", 28e9)  # Default to 28 GHz if not provided

    cc, dd = np.meshgrid(points_continuous, points_discrete)

    plot_title = f"Ex-field in V/m at {frequency/1e9:.1f} GHz"
    if in_log:
        # Add a small value to avoid log(0)
        min_positive = np.min(results[results > 0]) if np.any(results > 0) else 1e-10
        log_results = np.log10(np.maximum(results, min_positive))
        results = log_results
        plot_title = f"Ex-field in V/m (log scale) at {frequency/1e9:.1f} GHz"

    plt.figure(figsize=(12, 8))

    if method == "contourf":
        plt.contourf(cc, dd, results, levels=number_of_levels, cmap="jet")
    elif method == "imshow":
        plt.imshow(
            results,
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
        plt.pcolor(cc, dd, results, cmap="jet")

    cbar = plt.colorbar()
    if in_log:
        cbar.set_label("log10(E-field magnitude) [V/m]")
    else:
        cbar.set_label("E-field magnitude [V/m]")

    plt.xlabel(f"{continuous_axis} [mm]")
    plt.ylabel(f"{discrete_axis} [mm]")
    plt.xlim([np.min(points_continuous), np.max(points_continuous)])
    plt.ylim([np.min(points_discrete), np.max(points_discrete)])
    plt.title(plot_title)

    # Add text with data range info
    non_nan_values = results[~np.isnan(results)]
    plt.figtext(
        0.02,
        0.02,
        f"Min: {np.min(non_nan_values):.3f}, Max: {np.max(non_nan_values):.3f}, "
        f"Mean: {np.mean(non_nan_values):.3f}",
        bbox={"facecolor": "white", "alpha": 0.8},
    )

    if filename is None:
        filename = (
            "measurement_visualization_log.png" if in_log else "measurement_visualization.png"
        )

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved as '{filename}'")


if __name__ == "__main__":
    file_path = "measurement_data/x400_zy.pickle"
    print(f"\nAnalyzing file: {file_path}")
    print("-" * 50)
    data = examine_pickle_file(file_path)

    # Plot with normal scale
    print("\nCreating standard plot...")
    plot_data(data, method="contourf", in_log=False, filename="measurement_visualization.png")

    # Also plot with log scale
    print("\nCreating log-scale plot...")
    plot_data(data, method="contourf", in_log=True, filename="measurement_visualization_log.png")

    # Also try other visualization methods
    print("\nCreating pcolor visualization...")
    plot_data(data, method="pcolor", in_log=False, filename="measurement_visualization_pcolor.png")
