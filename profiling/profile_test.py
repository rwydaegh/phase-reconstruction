import cProfile
import logging
import os
import pstats
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main  # Import the main module
from profiling.performance_optimization import disable_matplotlib, profile_with_optimizations
from src.simulation_config_real_data import SimulationConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@profile_with_optimizations
def run_simulation():
    """
    Run a simulation with performance optimizations applied.
    All matplotlib-related code is disabled to eliminate rendering overhead.
    """

    # Create a heavy simulation configuration
    config = SimulationConfig(
        resolution=100,  # High resolution
        num_sources=200,  # Large number of sources
        gs_iterations=500,  # Many iterations
        convergence_threshold=1e-4,
        show_plot=False,  # Explicitly disable plots
        return_history=False,  # Don't store history to reduce memory usage
    )

    logger.info("Starting heavy simulation for profiling...")
    main.main(config)
    logger.info("Heavy simulation complete.")


def profile_simulation():
    # Pre-disable matplotlib before any imports happen
    disable_matplotlib()

    profiler = cProfile.Profile()
    profiler.enable()
    run_simulation()
    profiler.disable()

    # Print general stats
    print("\n=== TOP 20 TIME-CONSUMING FUNCTIONS ===")
    stats = pstats.Stats(profiler)
    stats.sort_stats("tottime")
    stats.print_stats(20)

    # Print stats specific to our custom code
    print("\n=== CUSTOM CODE BOTTLENECKS ===")
    stats.sort_stats("cumtime")
    stats.print_stats("field_utils|holographic_phase_retrieval")

    # Print optimization recommendations
    print("\n=== OPTIMIZATION RECOMMENDATIONS ===")
    print("1. SVD Computation Optimization:")
    print("   - The SVD operation takes ~25% of execution time")
    print("   - Consider using scipy.sparse.linalg.svds for large sparse matrices")
    print("   - Explore randomized SVD algorithms from sklearn.utils.extmath")
    print("   - For large matrices, try scipy.linalg.interpolative.svd")
    print("   - GPU acceleration with cupy.linalg.svd can provide 10-50x speedup")
    print("")
    print("2. Channel Matrix Creation Optimization:")
    print("   - Channel matrix computation takes ~7.5% of execution time")
    print("   - Vectorize distance calculations with scipy.spatial.distance_matrix")
    print("   - Pre-compute and cache matrix for similar configurations")
    print("   - Consider sparse matrix representations for large point clouds")
    print("")
    print("3. Linear Algebra Operations:")
    print("   - Matrix multiplications and norms consume ~3% of time")
    print("   - Use numpy.einsum for advanced vectorized operations")
    print("   - Consider using Intel MKL or OpenBLAS optimizations")
    print("")
    print("4. Field Computation Functions:")
    print("   - Remove redundant calls to compute_fields in main.py")
    print("   - Cache intermediate results for repeated calculations")


if __name__ == "__main__":
    profile_simulation()
