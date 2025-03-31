"""
Profiling and performance optimization package for field reconstruction framework.
"""

from .performance_optimization import (
    disable_matplotlib,
    profile_with_optimizations,
)

# Benchmark utilities
try:
    from .benchmark_optimizations import (
        benchmark_channel_matrix,
        benchmark_holographic_phase_retrieval,
        benchmark_svd_methods,
        run_all_benchmarks,
    )
except ImportError:
    # Benchmarking tools may have additional dependencies that aren't always available
    pass
