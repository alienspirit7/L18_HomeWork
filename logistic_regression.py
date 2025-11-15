import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Set random seed for reproducibility
np.random.seed(42)

# Create output directory
output_dir = 'results_output'
os.makedirs(output_dir, exist_ok=True)

# Load the dataset
csv_path = os.path.join(output_dir, 'random_points.csv')
df = pd.read_csv(csv_path)

# Extract features and labels
X = df[['x1', 'x2']].values
y = df['R'].values

# Add bias term (x0 = 1)
X_with_bias = np.column_stack([np.ones(len(X)), X])

# Initialize parameters
n_features = X_with_bias.shape[1]
beta = np.zeros(n_features)  # [β0, β1, β2]
learning_rate = 0.1
n_iterations = 10000
error_threshold = 0.01  # Stop when average error is below this threshold

# Storage for tracking
history = {
    'iteration': [],
    'beta0': [],
    'beta1': [],
    'beta2': [],
    'log_likelihood': [],
    'avg_error': []
}

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Log-likelihood function
def log_likelihood(X, y, beta):
    z = np.dot(X, beta)
    p = sigmoid(z)
    # Avoid log(0) by clipping
    p = np.clip(p, 1e-10, 1 - 1e-10)
    ll = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    return ll

# Error function (for individual predictions)
def compute_errors(y_true, p_pred):
    # Classification error: |y - p|
    return np.abs(y_true - p_pred)

print("Starting Gradient Descent...")
print(f"Initial β: {beta}")
print(f"Learning rate: {learning_rate}")
print(f"Max iterations: {n_iterations}")
print(f"Error threshold: {error_threshold}\n")

# Gradient Descent
converged = False
for iteration in range(n_iterations):
    # 1. Compute z (weighted sum)
    z = np.dot(X_with_bias, beta)

    # 2. Apply sigmoid to get probabilities
    p = sigmoid(z)

    # 3. Compute gradient: g = X^T · (y - p)
    gradient = np.dot(X_with_bias.T, (y - p))

    # 4. Update coefficients: β^(k+1) = β^(k) + η·g
    beta = beta + learning_rate * gradient

    # Track metrics
    ll = log_likelihood(X_with_bias, y, beta)
    errors = compute_errors(y, p)
    avg_error = np.mean(errors)

    history['iteration'].append(iteration)
    history['beta0'].append(beta[0])
    history['beta1'].append(beta[1])
    history['beta2'].append(beta[2])
    history['log_likelihood'].append(ll)
    history['avg_error'].append(avg_error)

    # Print progress every 100 iterations
    if iteration % 100 == 0:
        print(f"Iteration {iteration}: β = {beta}, Log-Likelihood = {ll:.4f}, Avg Error = {avg_error:.4f}")

    # Check convergence
    if avg_error < error_threshold:
        converged = True
        print(f"\n*** Converged at iteration {iteration} ***")
        print(f"Average error {avg_error:.6f} is below threshold {error_threshold}")
        break

if converged:
    print(f"\nFinal β: {beta}")
    print(f"Final Log-Likelihood: {history['log_likelihood'][-1]:.4f}")
    print(f"Final Average Error: {history['avg_error'][-1]:.6f}")
    print(f"Total iterations: {iteration + 1}\n")
else:
    print(f"\n*** Did not converge within {n_iterations} iterations ***")
    print(f"Final β: {beta}")
    print(f"Final Log-Likelihood: {history['log_likelihood'][-1]:.4f}")
    print(f"Final Average Error: {history['avg_error'][-1]:.6f}\n")

# Save iteration history
history_df = pd.DataFrame(history)
history_path = os.path.join(output_dir, 'gradient_descent_history.csv')
history_df.to_csv(history_path, index=False)
print(f"Gradient descent history saved to '{history_path}'")

# Compute final predictions
z_final = np.dot(X_with_bias, beta)
p_final = sigmoid(z_final)
errors_final = compute_errors(y, p_final)
predicted_class = (p_final >= 0.5).astype(int)

# Create results table
results_df = pd.DataFrame({
    'x1': df['x1'],
    'x2': df['x2'],
    'R': y,
    'sigma(x1,x2)': p_final,
    'error': errors_final
})

# Add average error row
avg_row = pd.DataFrame({
    'x1': [''],
    'x2': [''],
    'R': ['Average Error:'],
    'sigma(x1,x2)': [''],
    'error': [np.mean(errors_final)]
})
results_df = pd.concat([results_df, avg_row], ignore_index=True)

results_path = os.path.join(output_dir, 'logistic_regression_results.csv')
results_df.to_csv(results_path, index=False)
print(f"Results table saved to '{results_path}'")

# ===== VISUALIZATION 1: 3D Dataset with Sigmoid Surface =====
fig1 = plt.figure(figsize=(14, 10))
ax1 = fig1.add_subplot(111, projection='3d')

# Group 0 (blue) - actual
group0_actual = (y == 0)
# Predicted 0 (round)
group0_pred0 = group0_actual & (predicted_class == 0)
group0_pred1 = group0_actual & (predicted_class == 1)

# Plot points at their actual probability height
ax1.scatter(df.loc[group0_pred0, 'x1'], df.loc[group0_pred0, 'x2'],
            p_final[group0_pred0],
            c='blue', marker='o', label='Actual 0, Predicted 0', alpha=0.7, s=30)
ax1.scatter(df.loc[group0_pred1, 'x1'], df.loc[group0_pred1, 'x2'],
            p_final[group0_pred1],
            c='blue', marker='^', label='Actual 0, Predicted 1', alpha=0.7, s=30)

# Group 1 (red) - actual
group1_actual = (y == 1)
# Predicted 0 (round)
group1_pred0 = group1_actual & (predicted_class == 0)
group1_pred1 = group1_actual & (predicted_class == 1)

ax1.scatter(df.loc[group1_pred0, 'x1'], df.loc[group1_pred0, 'x2'],
            p_final[group1_pred0],
            c='red', marker='o', label='Actual 1, Predicted 0', alpha=0.7, s=30)
ax1.scatter(df.loc[group1_pred1, 'x1'], df.loc[group1_pred1, 'x2'],
            p_final[group1_pred1],
            c='red', marker='^', label='Actual 1, Predicted 1', alpha=0.7, s=30)

# Create sigmoid surface
x1_min, x1_max = 0, 1
x2_min, x2_max = 0, 1
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100),
                       np.linspace(x2_min, x2_max, 100))
X_grid = np.column_stack([np.ones(xx1.ravel().shape[0]), xx1.ravel(), xx2.ravel()])
Z = sigmoid(np.dot(X_grid, beta)).reshape(xx1.shape)

# Plot sigmoid surface
surf = ax1.plot_surface(xx1, xx2, Z, alpha=0.3, cmap='viridis',
                        edgecolor='none', label='Sigmoid Surface')

# Plot decision boundary plane at z=0.5
ax1.contour(xx1, xx2, Z, levels=[0.5], colors='black', linewidths=2,
            linestyles='--', offset=0.5)

ax1.set_xlabel('x1', fontsize=12)
ax1.set_ylabel('x2', fontsize=12)
ax1.set_zlabel('σ(x) - Probability', fontsize=12)
ax1.set_title('Logistic Regression - 3D Visualization\n(Color = Actual Class, Shape = Predicted Class)', fontsize=14)
ax1.legend(fontsize=9, loc='upper left')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_zlim(0, 1)
ax1.view_init(elev=20, azim=45)

plot1_path = os.path.join(output_dir, 'classification_results_3d.png')
plt.savefig(plot1_path, dpi=150, bbox_inches='tight')
print(f"3D Classification plot saved to '{plot1_path}'")

# ===== VISUALIZATION 2: Error and Log-Likelihood over iterations =====
fig2, ax2 = plt.subplots(figsize=(12, 6))

# Create primary axis for average error
color_error = 'tab:blue'
ax2.set_xlabel('Iteration', fontsize=12)
ax2.set_ylabel('Average Error', fontsize=12, color=color_error)
ax2.plot(history['iteration'], history['avg_error'], color=color_error,
         linewidth=2, label='Average Error')
ax2.tick_params(axis='y', labelcolor=color_error)
ax2.grid(True, alpha=0.3)

# Create secondary axis for log-likelihood
ax2_twin = ax2.twinx()
color_likelihood = 'tab:green'
ax2_twin.set_ylabel('Log-Likelihood', fontsize=12, color=color_likelihood)
ax2_twin.plot(history['iteration'], history['log_likelihood'], color=color_likelihood,
              linewidth=2, label='Log-Likelihood')
ax2_twin.tick_params(axis='y', labelcolor=color_likelihood)

# Add title and legend
ax2.set_title('Training Metrics Over Iterations', fontsize=14)
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10)

plt.tight_layout()
plot2_path = os.path.join(output_dir, 'training_metrics.png')
plt.savefig(plot2_path, dpi=150, bbox_inches='tight')
print(f"Training metrics plot saved to '{plot2_path}'")

plt.show()

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"Final Coefficients:")
print(f"  β0 (bias)  = {beta[0]:.6f}")
print(f"  β1 (x1)    = {beta[1]:.6f}")
print(f"  β2 (x2)    = {beta[2]:.6f}")
print(f"\nAccuracy: {np.mean(predicted_class == y) * 100:.2f}%")
print(f"Average Error: {np.mean(errors_final):.6f}")
print(f"Final Log-Likelihood: {history['log_likelihood'][-1]:.4f}")
print("="*60)
