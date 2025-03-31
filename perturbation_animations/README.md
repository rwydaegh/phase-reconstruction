# Perturbation Animations

This directory contains tools for generating animations that visualize the effects of perturbations on field reconstruction.

## Overview

The main script `generate_perturbation_animations.py` runs simulations with different perturbation factors and generates rotating 3D visualizations of the current distribution and field reconstruction process.

## Contents

- `generate_perturbation_animations.py`: Main script to generate perturbation animations
- `perturbation_results/`: Directory containing the generated animation files
  - `current_field_animation_0.1percent.gif`: Animation with 0.1% perturbation
  - `current_field_animation_0.5percent.gif`: Animation with 0.5% perturbation
  - `current_field_animation_1.0percent.gif`: Animation with 1.0% perturbation

## Usage

Run the script from the project root directory:

```bash
python perturbation_animations/generate_perturbation_animations.py
```

This will:
1. Run simulations for 3 perturbation factors (0.1%, 0.5%, 1.0%)
2. Generate animations showing 3D rotating current distributions and field reconstructions
3. Save the animations to the `perturbation_results/` directory
4. Display a summary of reconstruction quality metrics

## Animation Details

Each animation contains 4 panels:
- **Top-left**: 3D rotating visualization of current density distribution
- **Top-right**: Reconstructed field magnitude
- **Bottom-left**: True field magnitude
- **Bottom-right**: Error (absolute difference between true and reconstructed fields)

The 3D visualization rotates during the animation to provide a comprehensive view of the current distribution from all angles.
