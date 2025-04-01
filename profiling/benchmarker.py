import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

# Optional plotting
try:
    import matplotlib.pyplot as plt

    PLOTTING_ENABLED = True
except ImportError:
    PLOTTING_ENABLED = False

# Ensure the main project directory is in the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import functions to benchmark
try:
    # Example imports (adjust as needed)
    from src.utils.field_utils import compute_fields as compute_fields_original
    from src.utils.field_utils import create_channel_matrix as create_channel_matrix_original

    # Add imports for optimized versions if available, e.g.:
    # from src.utils.optimized_field_utils import create_channel_matrix_optimized
    create_channel_matrix_optimized = None  # Placeholder
    holographic_phase_retrieval = None  # Placeholder
    holographic_phase_retrieval_optimized = None  # Placeholder

except ImportError as e:
    logging.error(f"Failed to import necessary modules: {e}. Benchmarking may fail.")
    # Define placeholders if imports fail
    if "create_channel_matrix_original" not in locals():
        create_channel_matrix_original = None
    if "compute_fields_original" not in locals():
        compute_fields_original = None
    if "create_channel_matrix_optimized" not in locals():
        create_channel_matrix_optimized = None
    if "holographic_phase_retrieval" not in locals():
        holographic_phase_retrieval = None
    if "holographic_phase_retrieval_optimized" not in locals():
        holographic_phase_retrieval_optimized = None


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark suite."""

    resolutions: List[int] = field(default_factory=lambda: [32, 64, 96])
    num_sources: int = 100
    gs_iterations: int = 50
    convergence_threshold: float = 1e-3
    repetitions: int = 5
    wavelength: float = 10.7e-3
    room_size: float = 2.0
    plot_dir: str = "outputs/benchmarks"


@dataclass
class BenchmarkResult:
    """Stores results for a single benchmark function."""

    name: str
    parameters: Dict[str, Any]
    timings: List[float]
    average_time: float = 0.0
    std_dev: float = 0.0

    def __post_init__(self):
        if self.timings:
            self.average_time = np.mean(self.timings)
            self.std_dev = np.std(self.timings)


@dataclass
class BenchmarkComparison:
    """Stores results comparing multiple implementations for a task."""

    task_name: str
    parameter_name: str
    parameter_values: List[Any]
    results: Dict[str, List[BenchmarkResult]]


# --- Test Scenario Creation ---


def create_test_scenario(resolution: int, config: BenchmarkConfig):
    """Creates a consistent test scenario for a given resolution."""
    num_points = config.num_sources * 2
    points = np.random.uniform(0, config.room_size, (num_points, 3))

    # Create random currents for the sources
    currents = np.zeros(len(points), dtype=complex)
    source_indices = np.random.choice(len(points), config.num_sources, replace=False)
    amplitudes = np.random.lognormal(mean=0, sigma=1, size=config.num_sources)
    phases = np.random.uniform(0, 2 * np.pi, size=config.num_sources)
    for i, idx in enumerate(source_indices):
        currents[idx] = amplitudes[i] * np.exp(1j * phases[i])

    # Create measurement plane
    x = np.linspace(0, config.room_size, resolution)
    y = np.linspace(0, config.room_size, resolution)
    X, Y = np.meshgrid(x, y)
    measurement_plane = np.stack([X, Y, np.ones_like(X) * config.room_size / 2], axis=-1)

    k = 2 * np.pi / config.wavelength

    # Return data needed for benchmarking
    return {
        "points": points,
        "currents": currents,
        "measurement_plane": measurement_plane,
        "k": k,
        # Add other derived data if needed
    }


# --- Benchmarking Core ---


def time_function(func: Callable, args: Dict[str, Any], repetitions: int) -> BenchmarkResult:
    """Times a function call multiple times and returns results."""
    timings = []
    for i in range(repetitions):
        start_time = time.perf_counter()
        try:
            _ = func(**args)
        except Exception as e:
            logger.error(
                f"Error during benchmark of {func.__name__} (Rep {i + 1}): {e}", exc_info=True
            )
            timings.append(np.nan)  # Mark repetition as failed
            continue
        end_time = time.perf_counter()
        timings.append(end_time - start_time)

    # Filter out NaNs from failed repetitions before calculating stats
    valid_timings = [t for t in timings if not np.isnan(t)]
    return BenchmarkResult(name=func.__name__, parameters=args, timings=valid_timings)


# --- Specific Benchmark Tasks ---


def benchmark_channel_matrix_creation(config: BenchmarkConfig):
    """Benchmarks different channel matrix creation functions."""
    if not create_channel_matrix_original or not create_channel_matrix_optimized:
        logger.warning(
            "Skipping channel matrix benchmark: Original or Optimized function not available."
        )
        return None

    comparison = BenchmarkComparison(
        task_name="Channel Matrix Creation",
        parameter_name="Resolution",
        parameter_values=config.resolutions,
        results={"Original": [], "Optimized": []},
    )

    for res in config.resolutions:
        logger.info(f"Benchmarking Channel Matrix for Resolution: {res}x{res}")
        scenario = create_test_scenario(res, config)
        args = {
            "points": scenario["points"],
            "measurement_plane": scenario["measurement_plane"],
            "k": scenario["k"],
        }

        # Benchmark Original
        result_orig = time_function(create_channel_matrix_original, args, config.repetitions)
        comparison.results["Original"].append(result_orig)
        logger.info(
            f"  Original Avg Time: {result_orig.average_time:.4f}s ± {result_orig.std_dev:.4f}s"
        )

        # Benchmark Optimized
        result_opt = time_function(create_channel_matrix_optimized, args, config.repetitions)
        comparison.results["Optimized"].append(result_opt)
        logger.info(
            f"  Optimized Avg Time: {result_opt.average_time:.4f}s ± {result_opt.std_dev:.4f}s"
        )

        if result_orig.average_time > 0 and result_opt.average_time > 0:
            speedup = result_orig.average_time / result_opt.average_time
            logger.info(f"  Speedup: {speedup:.2f}x")

    return comparison


# TODO: Add more benchmark functions (e.g., benchmark_svd, benchmark_phase_retrieval)

# --- Visualization ---


def plot_comparison(comparison: BenchmarkComparison, config: BenchmarkConfig):
    """Generates and saves a plot comparing benchmark results."""
    if not PLOTTING_ENABLED:
        logger.warning("Plotting disabled because matplotlib is not installed.")
        return
    if not comparison:
        logger.warning("No comparison data to plot.")
        return

    plt.figure(figsize=(10, 6))
    parameter_values = comparison.parameter_values
    task_name = comparison.task_name
    param_name = comparison.parameter_name

    colors = plt.cm.viridis(np.linspace(0, 1, len(comparison.results)))
    markers = ["o", "s", "^", "d", "v", "<", ">"]

    for i, (impl_name, results_list) in enumerate(comparison.results.items()):
        avg_times = [res.average_time for res in results_list]
        std_devs = [res.std_dev for res in results_list]
        plt.errorbar(
            parameter_values,
            avg_times,
            yerr=std_devs,
            label=impl_name,
            marker=markers[i % len(markers)],
            color=colors[i],
            capsize=3,
            linestyle="-",
        )

    plt.xlabel(param_name)
    plt.ylabel("Average Execution Time (s)")
    plt.title(f"Benchmark: {task_name} vs {param_name}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(parameter_values)

    # Ensure plot directory exists
    os.makedirs(config.plot_dir, exist_ok=True)
    filename = f"{task_name.lower().replace(' ', '_')}_vs_{param_name.lower()}.png"
    filepath = os.path.join(config.plot_dir, filename)
    try:
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        logger.info(f"Benchmark plot saved to: {filepath}")
    except Exception as e:
        logger.error(f"Failed to save plot {filepath}: {e}")
    plt.close()


# --- Main Execution ---


def run_benchmarks(config: BenchmarkConfig):
    """Runs all defined benchmarks."""
    logger.info("Starting benchmark suite...")

    # Run specific benchmarks
    channel_matrix_comparison = benchmark_channel_matrix_creation(config)
    # Add calls to other benchmark functions here
    # svd_comparison = benchmark_svd(config)
    # phase_retrieval_comparison = benchmark_phase_retrieval(config)

    # Plot results
    plot_comparison(channel_matrix_comparison, config)
    # plot_comparison(svd_comparison, config)
    # plot_comparison(phase_retrieval_comparison, config)

    logger.info("Benchmark suite finished.")
    # TODO: Optionally, print a summary table of results


if __name__ == "__main__":
    logger.info("Running benchmarker script...")
    benchmark_config = BenchmarkConfig(
        # Example config override for a quick test run
        resolutions=[32, 64, 96, 128],
        repetitions=3,
    )
    run_benchmarks(benchmark_config)
