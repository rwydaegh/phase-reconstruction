# Performance Scaling Analysis

This document analyzes how the performance of our optimized field reconstruction simulation scales with different parameters. It focuses on the three most critical parameters that affect execution time:

1. **Resolution** - Grid size of the measurement plane (N×N)
2. **Number of Sources** - Active source points contributing to the field
3. **Convergence Threshold** - Stopping criterion for the iteration process

## Scaling Visualizations

The following heatmaps show how execution time (in seconds) scales with different parameter combinations:

![Time Analysis: Sources vs Resolution](figs/time_analysis_sources_vs_resolution.png)

*Figure 1: Execution time scaling with number of sources and resolution*

![Time Analysis: Sources vs Threshold](figs/time_analysis_sources_vs_threshold.png)

*Figure 2: Execution time scaling with number of sources and convergence threshold*

![Time Analysis: Resolution vs Threshold](figs/time_analysis_resolution_vs_threshold.png)

*Figure 3: Execution time scaling with resolution and convergence threshold*

## Key Scaling Behaviors

### 1. Resolution Scaling (O(N²))

Resolution has the most significant impact on performance due to:

- **Matrix Size Growth**: The channel matrix H grows as O(N²) with resolution N
- **SVD Computation**: SVD complexity increases with matrix dimensions
- **Memory Access Patterns**: Larger matrices show less efficient cache utilization

Our optimizations improved resolution scaling through:
- More efficient memory layout with Fortran-order arrays
- Vectorized distance calculations
- Optimized matrix operations

### 2. Number of Sources Scaling (Linear to Sublinear)

The number of sources shows a more moderate impact on performance:

- **Matrix Width**: Affects only the width of the channel matrix, not its height
- **SVD Computation**: Increases computation time but less dramatically than resolution
- **Field Complexity**: More sources create more complex field patterns

Optimizations particularly improved source scaling by:
- Efficient channel matrix computation
- Vectorized operations

### 3. Convergence Threshold Scaling (Algorithmic)

Convergence threshold shows an interesting relationship with performance:

- **Early Termination**: Looser thresholds allow the algorithm to converge faster
- **Iteration Count**: Stricter thresholds require more iterations
- **Accuracy Trade-off**: Faster execution comes at the cost of potential accuracy loss

## Computational Complexity Analysis

The overall computational complexity of our simulation can be approximated as:

**O(R² × S × I)**

Where:
- R = Resolution (one dimension of the N×N grid)
- S = Number of sources
- I = Number of iterations (affected by convergence threshold)

The dominant operations are:
1. Channel matrix creation: O(R² × S)
2. SVD computation: O(min(R²,S)² × max(R²,S))
3. Matrix multiplications in GS algorithm: O(R² × S × I)

## Optimization Impact on Scaling

Our optimizations have significantly improved scaling behavior:

1. **Memory Layout Optimization**:
   - Fortran-order arrays provide better scaling with resolution
   - Cache efficiency improves as matrix size grows

2. **Vectorized Operations**:
   - Reduce the scaling coefficient for all parameters
   - Particularly effective for the channel matrix creation

3. **Algorithm Streamlining**:
   - More predictable scaling with convergence threshold
   - Lower overhead per iteration

## Recommendations for Large-Scale Simulations

Based on the scaling analysis, we recommend:

1. **For Speed-Critical Applications**:
   - Use moderate resolution (30-40)
   - Use stricter convergence thresholds only when necessary
   - Limit number of sources to what's physically relevant

2. **For Accuracy-Critical Applications**:
   - Increase resolution gradually while monitoring performance
   - Use adaptive convergence thresholds
   - Consider using GPU acceleration for SVD with very large matrices

3. **For Massive Simulations**:
   - Implement the parallel computing approach discussed in the optimization analysis
   - Consider specialized hardware acceleration
   - Use problem-specific optimizations based on the physical context

## Conclusion

The optimized simulation shows good scaling behavior with moderate parameter values. For extremely large simulations, additional optimizations like GPU acceleration would be necessary to maintain reasonable execution times.

The most significant scaling challenge remains the SVD computation, which dominates execution time and grows superlinearly with resolution. This represents a fundamental mathematical constraint rather than an implementation issue.
