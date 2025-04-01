import cProfile
import logging
import os
import pstats
import sys
from io import StringIO

# Third-party imports
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

# Local application imports
import simulated_data_reconstruction

# Ensure the main project directory is in the Python path *after* imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging (after imports and path modification)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_profiling_config() -> DictConfig:
    """
    Creates a configuration suitable for profiling using Hydra's programmatic API.
    Loads the 'simulated_data' config and applies profiling-specific overrides.
    """
    # Clear Hydra's global state to avoid conflicts if run multiple times
    GlobalHydra.instance().clear()

    # Initialize Hydra
    with initialize(config_path="../conf", job_name="profiling_job", version_base="1.2"):
        # Define profiling overrides
        profiling_overrides = [
            "show_plot=False",
            "no_anim=True",
            "no_plot=True",
            "+save_plots=False",
            "+save_data=False",
            "verbose=False",
            # Optional overrides for faster runs:
            # "gs_iterations=50", "resolution=64",
        ]

        # Compose the configuration using Hydra
        cfg = compose(config_name="simulated_data", overrides=profiling_overrides)
        return cfg


def run_simulation_for_profiling():
    """Runs the main simulation with the profiling configuration."""
    config = create_profiling_config()
    try:
        # Hydra handles output directory creation
        simulated_data_reconstruction.main(config)
    except Exception as e:
        logger.error(f"Simulation failed during profiling: {e}", exc_info=True)
        raise


def run_profile(sort_by="tottime", top_n=30, filter_path=None):
    """
    Profiles the simulation run and prints statistics.

    Args:
        sort_by (str): The statistic to sort by (e.g., 'tottime', 'cumtime').
        top_n (int): The number of top functions to display.
        filter_path (str, optional): If provided, filters stats to show only functions
                                     containing this path string (e.g., 'src').
    """
    profiler = cProfile.Profile()
    profiler.enable()

    try:
        run_simulation_for_profiling()
    finally:
        profiler.disable()

    s = StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats(sort_by)

    if filter_path:
        stats.print_stats(filter_path, top_n)
    else:
        stats.print_stats(top_n)

    print("\n===== cProfile Statistics =====")
    print(f"Sorted by: {sort_by}")
    if filter_path:
        print(f"Filtered by path: {filter_path}")
    print(f"Showing top {top_n} functions:")
    print("=" * 30)
    print(s.getvalue())
    print("=" * 30)

    # Save stats to a file for more detailed analysis
    stats_filename = "profiling_results.prof"
    profiler.dump_stats(stats_filename)
    logger.info(f"Profiling data saved to '{stats_filename}'.")
    logger.info(f"Use 'snakeviz {stats_filename}' or other tools to visualize.")


if __name__ == "__main__":
    # Default execution: profile and show top 30 functions sorted by total time
    run_profile(sort_by="tottime", top_n=30)

    # Example: profile and show top 20 functions from 'src' sorted by cumulative time
    # run_profile(sort_by='cumtime', top_n=20, filter_path='src')
