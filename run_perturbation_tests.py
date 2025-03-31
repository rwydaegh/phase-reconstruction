import os
import subprocess

# Create directory for perturbation results if it doesn't exist
os.makedirs("perturbation_results", exist_ok=True)

# Define the perturbation factors
perturbation_factors = [0.001, 0.005, 0.01]  # 0.1%, 0.5%, 1%

for factor in perturbation_factors:
    print(f"\n==== Running simulation with perturbation_factor={factor} ====\n")

    # Create command with the custom config
    cmd = [
        "python",
        "main.py",
        "--perturb_points",
        "True",
        "--perturbation_factor",
        str(factor),
        "--output_file",
        f"perturbation_results/results_{factor*100:.1f}percent.png",
    ]

    # Run the command
    process = subprocess.run(cmd)

    # Rename the animation file to include the perturbation factor
    animation_name = f"current_field_animation_{factor*100:.1f}percent.gif"
    if os.path.exists("current_field_animation.gif"):
        os.replace("current_field_animation.gif", f"perturbation_results/{animation_name}")

    print(f"✓ Completed simulation with perturbation_factor={factor}")

print("\nAll simulations completed.")
