"""
Main script to run the complete logistic regression pipeline:
1. Generate random points dataset
2. Run logistic regression with gradient descent
"""

import subprocess
import sys
import os

def run_script(script_name, description):
    """Run a Python script and handle output"""
    print("=" * 70)
    print(f"STEP: {description}")
    print("=" * 70)

    try:
        # Run the script as a subprocess
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=False,
            text=True,
            check=True
        )
        print(f"\n✓ {description} completed successfully!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running {script_name}:")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"\n✗ Script not found: {script_name}")
        return False

def main():
    """Main pipeline execution"""
    print("\n" + "=" * 70)
    print("LOGISTIC REGRESSION PIPELINE")
    print("=" * 70)
    print("\nThis script will:")
    print("  1. Generate random points dataset (2 groups)")
    print("  2. Train logistic regression model using gradient descent")
    print("  3. Generate visualizations and results")
    print("\n")

    # Step 1: Generate points
    if not run_script("generate_points.py", "Generating Random Points Dataset"):
        print("\n⚠ Pipeline stopped due to error in data generation.")
        sys.exit(1)

    # Check if dataset was created
    data_file = os.path.join("results_output", "random_points.csv")
    if not os.path.exists(data_file):
        print(f"\n⚠ Error: Dataset file not found at {data_file}")
        sys.exit(1)

    # Step 2: Run logistic regression
    if not run_script("logistic_regression.py", "Training Logistic Regression Model"):
        print("\n⚠ Pipeline stopped due to error in model training.")
        sys.exit(1)

    # Success summary
    print("\n" + "=" * 70)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nGenerated files in 'results_output/' folder:")
    print("  • random_points.csv                    - Generated dataset")
    print("  • points_visualization.png             - Initial data visualization")
    print("  • gradient_descent_history.csv         - Training history")
    print("  • logistic_regression_results.csv      - Final predictions and errors")
    print("  • classification_results_3d.png        - 3D visualization with sigmoid surface")
    print("  • training_metrics.png                 - Error and log-likelihood curves")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
