#!/usr/bin/env python3
"""
Demo script to generate enhanced plots with the overfitting line more visible.
This is a simplified version to demonstrate the visualization improvements.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Sample data based on typical results
perturbation_factors = [0.0, 0.001, 0.01, 0.05, 0.1]

# Create more realistic sample data (approximately similar to actual results)
rmse_real = [0.001, 0.225, 0.292, 0.293, 0.293]        # Realistic (fixed currents)
rmse_overfit = [0.095, 0.005, 0.0005, 0.0005, 0.0005]  # Overfitting (optimized currents)

corr_real = [1.0, 0.40, 0.02, -0.01, -0.005]           # Realistic (fixed currents)
corr_overfit = [0.89, 0.999, 0.9999, 0.9999, 0.9999]   # Overfitting (optimized currents)

# Create output directory
output_dir = 'perturbation_analysis_revised/plots'
os.makedirs(output_dir, exist_ok=True)

# 1. Plot RMSE vs perturbation factor (log scale)
plt.figure(figsize=(10, 6))
plt.plot(perturbation_factors, rmse_real, 'o-', color='red', linewidth=2,
         label='Realistic: Perturbed geometry with true currents')
plt.plot(perturbation_factors, rmse_overfit, 'x-', color='purple', linewidth=2,
         label='Overfitting: Perturbed geometry with optimized currents')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Perturbation Factor', fontsize=12)
plt.ylabel('Normalized RMSE', fontsize=12)
plt.title('RMSE vs Perturbation Factor', fontsize=14)
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend(fontsize=10)
plt.tight_layout()

rmse_filename = os.path.join(output_dir, "rmse_vs_perturbation.png")
plt.savefig(rmse_filename, dpi=300)
print(f"Saved RMSE vs perturbation plot to {rmse_filename}")
plt.close()

# 2. Plot correlation vs perturbation factor with enhanced overfitting line
plt.figure(figsize=(10, 6))
plt.plot(perturbation_factors, corr_real, 'o-', color='red', linewidth=2,
         label='Realistic: Perturbed geometry with true currents')
# Enhanced visibility for overfitting line
plt.plot(perturbation_factors, corr_overfit, 'x-', color='purple', linewidth=3, markersize=10,
         label='Overfitting: Perturbed geometry with optimized currents')

plt.xscale('log')
plt.xlabel('Perturbation Factor', fontsize=12)
plt.ylabel('Correlation', fontsize=12)
plt.title('Correlation vs Perturbation Factor', fontsize=14)
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend(fontsize=10)
plt.tight_layout()

corr_filename = os.path.join(output_dir, "correlation_vs_perturbation.png")
plt.savefig(corr_filename, dpi=300)
print(f"Saved correlation vs perturbation plot to {corr_filename}")
plt.close()

# 3. Combined plot showing comparison between realistic and overfitting approaches
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.plot(perturbation_factors, rmse_real, 'o-', color='red', linewidth=2,
        label='Realistic (fixed currents)')
ax1.plot(perturbation_factors, rmse_overfit, 'x-', color='purple', linewidth=2,
        label='Overfitting (optimized currents)')

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Perturbation Factor', fontsize=12)
ax1.set_ylabel('Normalized RMSE (log scale)', fontsize=12)
ax1.set_title('RMSE: Realistic vs Overfitting', fontsize=14)
ax1.grid(True, which='both', linestyle='--', alpha=0.6)
ax1.legend(fontsize=10)

ax2.plot(perturbation_factors, corr_real, 'o-', color='red', linewidth=2,
        label='Realistic (fixed currents)')
# Enhanced visibility for overfitting line
ax2.plot(perturbation_factors, corr_overfit, 'x-', color='purple', linewidth=3, markersize=10,
        label='Overfitting (optimized currents)')

ax2.set_xscale('log')
ax2.set_xlabel('Perturbation Factor', fontsize=12)
ax2.set_ylabel('Correlation', fontsize=12)
ax2.set_title('Correlation: Realistic vs Overfitting', fontsize=14)
ax2.grid(True, which='both', linestyle='--', alpha=0.6)
ax2.legend(fontsize=10)

plt.tight_layout()
comparison_filename = os.path.join(output_dir, "realistic_vs_overfitting.png")
plt.savefig(comparison_filename, dpi=300)
print(f"Saved comparison plot to {comparison_filename}")
plt.close()

print(f"\nAll demo plots saved to {output_dir}")
