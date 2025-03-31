"""
Benchmark script to compare original vs optimized implementations.
Measures performance improvements for each optimization.
"""

import logging
import os
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

# Import original implementations
from src.holographic_phase_retrieval import holographic_phase_retrieval
from src.holographic_phase_retrieval_optimized import (
    holographic_phase_retrieval_optimized,
    svd_randomized,
    svd_standard,
)
from src.utils.field_utils import compute_fields, create_channel_matrix, reconstruct_field

# Import optimized implementations
from src.utils.optimized_field_utils import (
    compute_fields_optimized,
    create_channel_matrix_optimized,
    reconstruct_field_optimized,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure figs directory exists
os.makedirs("figs", exist_ok=True)
@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking tests."""

    min_resolution: int = 50
    max_resolution: int = 150
    step_size: int = 25
    num_sources: int = 200
    gs_iterations: int = 100
    convergence_threshold: float = 1e-3
    repetitions: int = 3  # Number of times to repeat each test for averaging
    wavelength: float = 10.7e-3  # 28GHz wavelength in meters
    room_size: float = 2.0


def create_test_scenario(config: BenchmarkConfig, resolution: int):
    """Create a test scenario with specified resolution."""
    # Create sample points
    num_points = config.num_sources * 3  # Create more points than sources
    points = np.random.uniform(0, config.room_size, (num_points, 3))

    # Create random currents
    currents = np.zeros(len(points), dtype=complex)
    source_indices = np.random.choice(len(points), config.num_sources, replace=False)
    amplitudes = np.random.lognormal(mean=0, sigma=1, size=config.num_sources)
    phases = np.random.uniform(0, 2 * np.pi, size=config.num_sources)

    # Set complex currents
    for i, idx in enumerate(source_indices):
        currents[idx] = amplitudes[i] * np.exp(1j * phases[i])

    # Create measurement plane
    x = np.linspace(0, config.room_size, resolution)
    y = np.linspace(0, config.room_size, resolution)
    X, Y = np.meshgrid(x, y)
    measurement_plane = np.stack([X, Y, np.ones_like(X) * config.room_size / 2], axis=-1)

    # Wave number
    k = 2 * np.pi / config.wavelength

    return points, currents, measurement_plane, k


def benchmark_channel_matrix(config: BenchmarkConfig):
    """Benchmark channel matrix creation."""
    resolutions = range(config.min_resolution, config.max_resolution + 1, config.step_size)

    original_times = []
    optimized_times = []

    for resolution in resolutions:
        logger.info(f"Benchmarking channel matrix with resolution {resolution}")

        # Create test scenario
        points, _, measurement_plane, k = create_test_scenario(config, resolution)

        # Benchmark original implementation
        original_total = 0
        for _ in range(config.repetitions):
            start_time = time.time()
            _ = create_channel_matrix(points, measurement_plane, k)
            original_total += time.time() - start_time
        original_avg = original_total / config.repetitions
        original_times.append(original_avg)

        # Benchmark optimized implementation
        optimized_total = 0
        for _ in range(config.repetitions):
            start_time = time.time()
            _ = create_channel_matrix_optimized(points, measurement_plane, k)
            optimized_total += time.time() - start_time
        optimized_avg = optimized_total / config.repetitions
        optimized_times.append(optimized_avg)

        logger.info(
            f"  Original: {original_avg:.4f}s, Optimized: {optimized_avg:.4f}s, "
            f"Speedup: {original_avg/optimized_avg:.2f}x"
        )

    return resolutions, original_times, optimized_times


def benchmark_svd_methods(config: BenchmarkConfig):
    """Benchmark different SVD implementations."""
    resolutions = range(config.min_resolution, config.max_resolution + 1, config.step_size)

    standard_times = []
    randomized_times = []

    for resolution in resolutions:
        logger.info(f"Benchmarking SVD with resolution {resolution}")

        # Create test scenario
        points, _, measurement_plane, k = create_test_scenario(config, resolution)

        # Create channel matrix (same for both methods)
        H = create_channel_matrix_optimized(points, measurement_plane, k)

        # Benchmark standard SVD
        standard_total = 0
        for _ in range(config.repetitions):
            start_time = time.time()
            _ = svd_standard(H)
            standard_total += time.time() - start_time
        standard_avg = standard_total / config.repetitions
        standard_times.append(standard_avg)

        # Benchmark randomized SVD
        randomized_total = 0
        for _ in range(config.repetitions):
            try:
                start_time = time.time()
                _ = svd_randomized(H, n_components=min(H.shape) - 1)
                randomized_total += time.time() - start_time
            except Exception as e:
                logger.error(f"Randomized SVD failed: {str(e)}")
                randomized_total = float("nan")
                break

        if not np.isnan(randomized_total):
            randomized_avg = randomized_total / config.repetitions
            randomized_times.append(randomized_avg)

            logger.info(
                f"  Standard: {standard_avg:.4f}s, Randomized: {randomized_avg:.4f}s, "
                f"Speedup: {standard_avg/randomized_avg:.2f}x"
            )
        else:
            randomized_times.append(float("nan"))
            logger.info(f"  Standard: {standard_avg:.4f}s, Randomized: Failed")

    return resolutions, standard_times, randomized_times


def benchmark_holographic_phase_retrieval(config: BenchmarkConfig):
    """Benchmark holographic phase retrieval."""
    resolutions = range(config.min_resolution, config.max_resolution + 1, config.step_size)

    original_times = []
    optimized_times = []
    optimized_randomized_times = []

    for resolution in resolutions:
        logger.info(f"Benchmarking holographic phase retrieval with resolution {resolution}")

        # Create test scenario
        points, currents, measurement_plane, k = create_test_scenario(config, resolution)

        # Create channel matrix and compute ground truth field
        H = create_channel_matrix_optimized(points, measurement_plane, k)
        true_field = compute_fields_optimized(points, currents, measurement_plane, k, H)
        measured_magnitude = np.abs(true_field)

        # Benchmark original implementation
        original_total = 0
        for _ in range(config.repetitions):
            start_time = time.time()
            _ = holographic_phase_retrieval(
                H,
                measured_magnitude,
                num_iterations=config.gs_iterations,
                convergence_threshold=config.convergence_threshold,
                return_history=False,
                debug=False,
            )
            original_total += time.time() - start_time
        original_avg = original_total / config.repetitions
        original_times.append(original_avg)

        # Benchmark optimized implementation with standard SVD
        optimized_total = 0
        for _ in range(config.repetitions):
            start_time = time.time()
            _ = holographic_phase_retrieval_optimized(
                H,
                measured_magnitude,
                num_iterations=config.gs_iterations,
                convergence_threshold=config.convergence_threshold,
                return_history=False,
                debug=False,
                svd_method="standard",
                verbose=False,
            )
            optimized_total += time.time() - start_time
        optimized_avg = optimized_total / config.repetitions
        optimized_times.append(optimized_avg)

        # Benchmark optimized implementation with randomized SVD
        try:
            optimized_randomized_total = 0
            for _ in range(config.repetitions):
                start_time = time.time()
                _ = holographic_phase_retrieval_optimized(
                    H,
                    measured_magnitude,
                    num_iterations=config.gs_iterations,
                    convergence_threshold=config.convergence_threshold,
                    return_history=False,
                    debug=False,
                    svd_method="randomized",
                    verbose=False,
                )
                optimized_randomized_total += time.time() - start_time
            optimized_randomized_avg = optimized_randomized_total / config.repetitions
            optimized_randomized_times.append(optimized_randomized_avg)

            logger.info(
                f"  Original: {original_avg:.4f}s, Optimized: {optimized_avg:.4f}s, "
                f"Optimized+RandomizedSVD: {optimized_randomized_avg:.4f}s"
            )
            logger.info(
                f"  Speedup (Opt/Orig): {original_avg/optimized_avg:.2f}x, "
                f"Speedup (OptRand/Orig): {original_avg/optimized_randomized_avg:.2f}x"
            )
        except Exception as e:
            logger.error(f"Randomized SVD benchmark failed: {str(e)}")
            optimized_randomized_times.append(float("nan"))

            logger.info(f"  Original: {original_avg:.4f}s, Optimized: {optimized_avg:.4f}s")
            logger.info(f"  Speedup (Opt/Orig): {original_avg/optimized_avg:.2f}x")

    return resolutions, original_times, optimized_times, optimized_randomized_times


def visualize_results(
    resolutions,
    title,
    y_label,
    data_sets,
    labels,
    colors,
    styles,
    filename,
    y_log=False,
    speedup=True,
):
    """Visualize benchmark results."""
    plt.figure(figsize=(12, 6))

    # Create the primary plot
    for i, data in enumerate(data_sets):
        plt.plot(
            resolutions, data, marker="o", linestyle=styles[i], color=colors[i], label=labels[i]
        )

    plt.xlabel("Resolution")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.7)

    if y_log:
        plt.yscale("log")

    # Create the speedup subplot if requested
    if speedup and len(data_sets) > 1:
        # Calculate speedups
        ax1 = plt.gca()
        ax2 = ax1.twinx()

        speedups = []
        for i in range(1, len(data_sets)):
            # Calculate speedup of each optimization over the baseline
            speedup_data = []
            for j in range(len(resolutions)):
                if np.isnan(data_sets[i][j]) or np.isnan(data_sets[0][j]):
                    speedup_data.append(np.nan)
                else:
                    speedup_data.append(data_sets[0][j] / data_sets[i][j])
            speedups.append(speedup_data)

        speedup_labels = [f"{labels[i]} Speedup" for i in range(1, len(labels))]
        speedup_colors = ["darkgreen", "darkred", "darkblue", "darkorange"]
        speedup_styles = ["-.", "-.", "-.", "-."]

        # Plot speedups
        for i, data in enumerate(speedups):
            ax2.plot(
                resolutions,
                data,
                marker="s",
                linestyle=speedup_styles[i],
                color=speedup_colors[i],
                label=speedup_labels[i],
            )

        ax2.set_ylabel("Speedup Factor (x)")

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    else:
        plt.legend()

    plt.tight_layout()
    plt.savefig(f"figs/{filename}.png", dpi=300)
    logger.info(f"Saved visualization to figs/{filename}.png")


def run_all_benchmarks():
    """Run all benchmarks and visualize results."""
    config = BenchmarkConfig()

    # Benchmark channel matrix creation
    logger.info("Benchmarking channel matrix creation...")
    resolutions, original_times, optimized_times = benchmark_channel_matrix(config)
    visualize_results(
        resolutions,
        "Channel Matrix Creation Performance",
        "Execution Time (seconds)",
        [original_times, optimized_times],
        ["Original", "Optimized (scipy.cdist)"],
        ["blue", "green"],
        ["-", "-"],
        "channel_matrix_benchmark",
    )

    # Benchmark SVD methods
    logger.info("Benchmarking SVD methods...")
    resolutions, standard_times, randomized_times = benchmark_svd_methods(config)
    visualize_results(
        resolutions,
        "SVD Methods Performance",
        "Execution Time (seconds)",
        [standard_times, randomized_times],
        ["Standard SVD", "Randomized SVD"],
        ["blue", "green"],
        ["-", "-"],
        "svd_benchmark",
    )

    # Benchmark holographic phase retrieval
    logger.info("Benchmarking holographic phase retrieval...")
    resolutions, original_times, optimized_times, optimized_randomized_times = (
        benchmark_holographic_phase_retrieval(config)
    )
    visualize_results(
        resolutions,
        "Holographic Phase Retrieval Performance",
        "Execution Time (seconds)",
        [original_times, optimized_times, optimized_randomized_times],
        ["Original", "Optimized", "Optimized + Randomized SVD"],
        ["blue", "green", "red"],
        ["-", "-", "-"],
        "phase_retrieval_benchmark",
        y_log=True,
    )

    # Generate summary
    generate_summary(
        resolutions,
        original_channel_times=original_times,
        optimized_channel_times=optimized_times,
        standard_svd_times=standard_times,
        randomized_svd_times=randomized_times,
        original_hpr_times=original_times,
        optimized_hpr_times=optimized_times,
        optimized_randomized_hpr_times=optimized_randomized_times,
    )


def generate_summary(
    resolutions,
    original_channel_times,
    optimized_channel_times,
    standard_svd_times,
    randomized_svd_times,
    original_hpr_times,
    optimized_hpr_times,
    optimized_randomized_hpr_times,
):
    """Generate a summary of benchmark results."""
    # Calculate average speedups
    channel_speedups = [
        o / n
        for o, n in zip(original_channel_times, optimized_channel_times)
        if not np.isnan(o) and not np.isnan(n)
    ]
    svd_speedups = [
        o / n
        for o, n in zip(standard_svd_times, randomized_svd_times)
        if not np.isnan(o) and not np.isnan(n)
    ]
    hpr_speedups = [
        o / n
        for o, n in zip(original_hpr_times, optimized_hpr_times)
        if not np.isnan(o) and not np.isnan(n)
    ]
    hpr_rand_speedups = [
        o / n
        for o, n in zip(original_hpr_times, optimized_randomized_hpr_times)
        if not np.isnan(o) and not np.isnan(n)
    ]

    avg_channel_speedup = np.mean(channel_speedups) if channel_speedups else float("nan")
    avg_svd_speedup = np.mean(svd_speedups) if svd_speedups else float("nan")
    avg_hpr_speedup = np.mean(hpr_speedups) if hpr_speedups else float("nan")
    avg_hpr_rand_speedup = np.mean(hpr_rand_speedups) if hpr_rand_speedups else float("nan")

    # Create a summary figure
    plt.figure(figsize=(10, 8))

    # Optimization speedups
    labels = [
        "Channel Matrix\n(scipy.cdist)",
        "SVD\n(Randomized)",
        "Phase Retrieval\n(Optimized)",
        "Phase Retrieval\n(Opt+RandSVD)",
    ]

    values = [avg_channel_speedup, avg_svd_speedup, avg_hpr_speedup, avg_hpr_rand_speedup]

    colors = ["#4CAF50", "#2196F3", "#FFC107", "#FF5722"]

    # Check for NaN values and replace with 0
    for i in range(len(values)):
        if np.isnan(values[i]):
            values[i] = 0
            labels[i] += "\n(Failed)"

    plt.bar(labels, values, color=colors)
    plt.axhline(y=1.0, color="r", linestyle="--", label="Baseline (No Speedup)")

    plt.title("Performance Optimization Speedup Factors", fontsize=16)
    plt.ylabel("Average Speedup Factor (x)", fontsize=14)
    plt.ylim(bottom=0)

    # Add value labels on top of each bar
    for i, v in enumerate(values):
        plt.text(i, v + 0.1, f"{v:.2f}x", ha="center", fontweight="bold")

    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("figs/optimization_summary.png", dpi=300)
    logger.info("Saved summary visualization to figs/optimization_summary.png")

    # Create textual summary
    with open("optimization_summary.txt", "w") as f:
        f.write("===== PERFORMANCE OPTIMIZATION SUMMARY =====\n\n")
        f.write(f"Channel Matrix Creation (scipy.cdist): {avg_channel_speedup:.2f}x speedup\n")
        f.write(f"SVD Computation (Randomized): {avg_svd_speedup:.2f}x speedup\n")
        f.write(f"Phase Retrieval (Optimized): {avg_hpr_speedup:.2f}x speedup\n")
        f.write(
            f"Phase Retrieval (Optimized + Randomized SVD): {avg_hpr_rand_speedup:.2f}x speedup\n\n"
        )

        f.write("Detailed Results by Resolution:\n")
        f.write("-------------------------------\n")
        for i, res in enumerate(resolutions):
            f.write(f"\nResolution: {res}x{res}\n")
            f.write(
                f"  Channel Matrix: {original_channel_times[i]:.4f}s -> "
                f"{optimized_channel_times[i]:.4f}s "
            )
            if not np.isnan(original_channel_times[i]) and not np.isnan(optimized_channel_times[i]):
                f.write(f"({original_channel_times[i]/optimized_channel_times[i]:.2f}x)\n")
            else:
                f.write("(N/A)\n")

            f.write(f"  SVD: {standard_svd_times[i]:.4f}s -> {randomized_svd_times[i]:.4f}s ")
            if not np.isnan(standard_svd_times[i]) and not np.isnan(randomized_svd_times[i]):
                f.write(f"({standard_svd_times[i]/randomized_svd_times[i]:.2f}x)\n")
            else:
                f.write("(N/A)\n")

            f.write(
                f"  Phase Retrieval: {original_hpr_times[i]:.4f}s -> {optimized_hpr_times[i]:.4f}s "
            )
            if not np.isnan(original_hpr_times[i]) and not np.isnan(optimized_hpr_times[i]):
                f.write(f"({original_hpr_times[i]/optimized_hpr_times[i]:.2f}x)\n")
            else:
                f.write("(N/A)\n")

            f.write(
                f"  Phase Retrieval + Randomized SVD: {original_hpr_times[i]:.4f}s -> "
                f"{optimized_randomized_hpr_times[i]:.4f}s "
            )
            if not np.isnan(original_hpr_times[i]) and not np.isnan(
                optimized_randomized_hpr_times[i]
            ):
                f.write(f"({original_hpr_times[i]/optimized_randomized_hpr_times[i]:.2f}x)\n")
            else:
                f.write("(N/A)\n")

    logger.info("Saved detailed summary to optimization_summary.txt")


if __name__ == "__main__":
    run_all_benchmarks()
