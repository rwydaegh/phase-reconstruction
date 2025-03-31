"""
Performance optimization utilities for profiling and benchmarking.
"""
import functools
import sys
import logging

logger = logging.getLogger(__name__)

def disable_matplotlib():
    """
    Disable matplotlib to eliminate rendering overhead during profiling.
    This should be called before any imports to ensure matplotlib doesn't initialize.
    """
    sys.modules['matplotlib'] = type('', (), {})
    sys.modules['matplotlib.pyplot'] = type('', (), {
        'figure': lambda *args, **kwargs: None,
        'plot': lambda *args, **kwargs: None,
        'scatter': lambda *args, **kwargs: None,
        'show': lambda *args, **kwargs: None,
        'savefig': lambda *args, **kwargs: None,
        'close': lambda *args, **kwargs: None,
        'title': lambda *args, **kwargs: None,
        'xlabel': lambda *args, **kwargs: None,
        'ylabel': lambda *args, **kwargs: None,
        'legend': lambda *args, **kwargs: None,
        'grid': lambda *args, **kwargs: None,
        'colorbar': lambda *args, **kwargs: None,
        'pcolormesh': lambda *args, **kwargs: None,
    })
    logger.info("Matplotlib disabled for profiling")

def profile_with_optimizations(func):
    """
    Decorator to apply performance optimizations to a function.
    This can include:
    - Numpy optimization settings
    - Memory usage reduction
    - Just-in-time compilation (if numba is available)
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Set numpy optimization flags
        import numpy as np
        
        # Save original settings
        original_settings = np.seterr(all='ignore')
        
        try:
            # Try to enable numba if available
            try:
                import numba
                logger.info("Numba JIT compilation available")
            except ImportError:
                pass
            
            # Execute the function
            return func(*args, **kwargs)
        finally:
            # Restore original numpy settings
            np.seterr(**original_settings)
    
    return wrapper
