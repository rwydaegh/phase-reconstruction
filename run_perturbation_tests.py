import os
import subprocess
import sys

# Define the perturbation factors to test
perturbation_factors = [0.001, 0.005, 0.01, 0.02, 0.05]  # Example factors (0.1%, 0.5%, 1%, 2%, 5%)

# Construct the Hydra multirun command
# We use 'poetry run' to ensure the correct environment
# '-m' enables multirun
# 'perturb_points=true' enables perturbation for all runs
# 'perturbation_factor=' specifies the parameter to sweep, followed by the comma-separated values
command = [
    "poetry",
    "run",
    "python",
    "simulated_data_reconstruction.py",
    "-m",
    "perturb_points=true",
    f"perturbation_factor={','.join(map(str, perturbation_factors))}",
    # Optional: Customize Hydra output directory if needed
    # "hydra.sweep.dir=outputs/perturbation_sweep/${now:%Y-%m-%d_%H-%M-%S}"
]

print("=" * 50)
print("Running Perturbation Sweep using Hydra Multirun")
print(f"Factors: {perturbation_factors}")
print(f"Command: {' '.join(command)}")
print("=" * 50)

# Execute the command
try:
    # Use check=True to raise an error if the command fails
    process = subprocess.run(command, check=True, text=True, stdout=sys.stdout, stderr=sys.stderr)
    print("\n" + "=" * 50)
    print("Perturbation sweep completed successfully.")
    print("Outputs are located in the 'multirun/' directory (or custom Hydra output path).")
    print("=" * 50)
except subprocess.CalledProcessError as e:
    print("\n" + "=" * 50)
    print(f"Error during perturbation sweep: {e}")
    print("=" * 50)
except FileNotFoundError:
    print("\n" + "=" * 50)
    print("Error: 'poetry' command not found. Make sure Poetry is installed and in your PATH.")
    print("=" * 50)
