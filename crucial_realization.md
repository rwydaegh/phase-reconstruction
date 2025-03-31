# Crucial Realization: Perturbation Timing in GS Algorithm

## Problem Observed

The Gerchberg-Saxton (GS) algorithm, even with advanced escape mechanisms (perturbations, restarts), fails to converge significantly beyond a certain error plateau (approx. 1e-1) when dealing with perturbed input data (`perturbation_factor=0.005`). The convergence plot (`figs/gs_convergence.png`) shows that while perturbations cause temporary error spikes, the algorithm quickly returns to the same error level.

## Initial Hypothesis & Debugging

The initial hypothesis was that the magnitude constraint, applied within the GS loop, might be immediately counteracting the effect of the perturbation intended to escape the local minimum.

Logging was added before and after the perturbation step and the magnitude constraint step.

## Refined Hypothesis (The Crucial Realization)

Analysis of the logs and the code structure revealed the exact interaction:

1.  **Original Code:** Perturbation was applied *before* the core GS forward/inverse transforms and the subsequent magnitude constraint.
    *   `Perturb field_values` -> `Calculate cluster_coeffs` -> `Calculate simulated_field` -> `Apply magnitude constraint to simulated_field to get next field_values`.
    *   **Effect:** The magnitude constraint effectively reset the state, nullifying the perturbation's effect before the next iteration began.

2.  **Modified Code (Attempt 1):** Perturbation logic was moved *after* the magnitude constraint.
    *   `Calculate cluster_coeffs` -> `Calculate simulated_field` -> `Apply magnitude constraint to simulated_field to get next field_values` -> `Perturb the constrained field_values`.
    *   **Effect:** This allowed the perturbation to directly influence the `field_values` used in the *next* iteration's `H_pinv @ field_values` step. This resulted in finding a slightly better minimum (error ~0.07 vs ~0.10), but convergence still stalled, and perturbations were largely ineffective long-term. The GS steps still pulled the solution back.

## Conclusion

The timing of the perturbation relative to the magnitude constraint is critical. Applying it *after* the constraint allows it to affect the next iteration, leading to slightly better results. However, the fundamental GS projection steps still dominate and prevent the algorithm from escaping the local minimum effectively in this perturbed scenario.

## Next Step (Plan)

To understand the intra-iteration dynamics better, add more detailed logging *within* the loop to track the estimated error at these key stages:
1.  Start of the iteration (using state from the end of the previous iteration).
2.  After the forward/inverse transforms (pre-magnitude constraint).
3.  After the magnitude constraint (pre-perturbation).
4.  After the perturbation (end of iteration state).

This will clarify precisely how the error changes at each step when a perturbation is applied.