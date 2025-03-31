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
        run_all_benchmarks,
        benchmark_channel_matrix,
        benchmark_svd_methods,
        benchmark_holographic_phase_retrieval,
    )
except ImportError:
    # Benchmarking tools may have additional dependencies that aren't always available
    pass
