import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

np.random.seed(42)
output_dir = 'results_output'
os.makedirs(output_dir, exist_ok=True)

print("="*60 + "\nLOGISTIC REGRESSION COMPARISON: Manual vs Scikit-Learn\n" + "="*60)

# Load dataset
df = pd.read_csv(os.path.join(output_dir, 'random_points.csv'))
X, y = df[['x1', 'x2']].values, df['R'].values
print(f"\nDataset loaded: {len(X)} samples\nFeatures shape: {X.shape}\nLabels shape: {y.shape}")

# Scikit-Learn implementation
print("\n" + "="*60 + "\nSCIKIT-LEARN LOGISTIC REGRESSION\n" + "="*60)
sklearn_model = LogisticRegression(random_state=42, max_iter=10000, solver='lbfgs', fit_intercept=True)
print("\nTraining scikit-learn model...")
sklearn_model.fit(X, y)

y_pred_sklearn = sklearn_model.predict(X)
y_prob_sklearn = sklearn_model.predict_proba(X)[:, 1]
accuracy_sklearn = accuracy_score(y, y_pred_sklearn)
misclassified_sklearn = np.sum(y != y_pred_sklearn)

print(f"\nScikit-learn Results:")
print(f"  Coefficients: β0={sklearn_model.intercept_[0]:.6f}, β1={sklearn_model.coef_[0][0]:.6f}, β2={sklearn_model.coef_[0][1]:.6f}")
print(f"  Accuracy: {accuracy_sklearn*100:.2f}%\n  Misclassified points: {misclassified_sklearn}/{len(y)}\n  Number of iterations: {sklearn_model.n_iter_[0]}")

# Load manual results
print("\n" + "="*60 + "\nMANUAL IMPLEMENTATION (from logistic_regression.py)\n" + "="*60)
manual_results = pd.read_csv(os.path.join(output_dir, 'logistic_regression_results.csv'))
manual_history = pd.read_csv(os.path.join(output_dir, 'gradient_descent_history.csv'))
beta_manual = manual_history.iloc[-1][['beta0', 'beta1', 'beta2']].values
y_prob_manual = manual_results['sigma(x1,x2)'].values[:-1]
y_pred_manual = (y_prob_manual >= 0.5).astype(int)
accuracy_manual = accuracy_score(y, y_pred_manual)
misclassified_manual = np.sum(y != y_pred_manual)

print(f"\nManual Implementation Results:")
print(f"  Coefficients: β0={beta_manual[0]:.6f}, β1={beta_manual[1]:.6f}, β2={beta_manual[2]:.6f}")
print(f"  Accuracy: {accuracy_manual*100:.2f}%\n  Misclassified points: {misclassified_manual}/{len(y)}\n  Number of iterations: {len(manual_history)}")

# Comparison
print("\n" + "="*60 + "\nCOMPARISON SUMMARY\n" + "="*60)
comparison_df = pd.DataFrame({
    'Metric': ['Intercept (β0)', 'Coefficient β1', 'Coefficient β2', 'Accuracy (%)', 'Misclassified', 'Iterations'],
    'Manual': [f"{beta_manual[0]:.6f}", f"{beta_manual[1]:.6f}", f"{beta_manual[2]:.6f}",
               f"{accuracy_manual*100:.2f}", f"{misclassified_manual}", f"{len(manual_history)}"],
    'Scikit-Learn': [f"{sklearn_model.intercept_[0]:.6f}", f"{sklearn_model.coef_[0][0]:.6f}", f"{sklearn_model.coef_[0][1]:.6f}",
                     f"{accuracy_sklearn*100:.2f}", f"{misclassified_sklearn}", f"{sklearn_model.n_iter_[0]}"],
    'Difference': [f"{abs(beta_manual[0] - sklearn_model.intercept_[0]):.6f}",
                   f"{abs(beta_manual[1] - sklearn_model.coef_[0][0]):.6f}",
                   f"{abs(beta_manual[2] - sklearn_model.coef_[0][1]):.6f}",
                   f"{abs(accuracy_manual - accuracy_sklearn)*100:.2f}",
                   f"{abs(misclassified_manual - misclassified_sklearn)}",
                   f"{abs(len(manual_history) - sklearn_model.n_iter_[0])}"]
})

print("\n" + comparison_df.to_string(index=False))
comparison_path = os.path.join(output_dir, 'comparison_manual_vs_sklearn.csv')
comparison_df.to_csv(comparison_path, index=False)
print(f"\nComparison saved to '{comparison_path}'")

# Visualization 1: Comparison plots
fig = plt.figure(figsize=(18, 6))

# Subplot 1: Coefficient Comparison
ax1 = fig.add_subplot(131)
coefficients = ['β0 (bias)', 'β1 (x1)', 'β2 (x2)']
manual_coefs = [beta_manual[0], beta_manual[1], beta_manual[2]]
sklearn_coefs = [sklearn_model.intercept_[0], sklearn_model.coef_[0][0], sklearn_model.coef_[0][1]]
x_pos, width = np.arange(len(coefficients)), 0.35

bars1 = ax1.bar(x_pos - width/2, manual_coefs, width, label='Manual', color='steelblue', alpha=0.8)
bars2 = ax1.bar(x_pos + width/2, sklearn_coefs, width, label='Scikit-Learn', color='darkorange', alpha=0.8)
ax1.set_xlabel('Coefficients', fontsize=12)
ax1.set_ylabel('Value', fontsize=12)
ax1.set_title('Coefficient Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(coefficients)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}', ha='center', va='bottom', fontsize=9)

# Subplot 2: Probability Distribution
ax2 = fig.add_subplot(132)
scatter = ax2.scatter(y_prob_manual, y_prob_sklearn, c=y, cmap='RdYlBu', alpha=0.6, s=20, edgecolors='black', linewidth=0.5)
ax2.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Agreement')
ax2.set_xlabel('Manual Implementation Probability', fontsize=12)
ax2.set_ylabel('Scikit-Learn Probability', fontsize=12)
ax2.set_title('Prediction Probability Comparison', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
cbar = plt.colorbar(scatter, ax=ax2)
cbar.set_label('True Class', fontsize=10)
correlation = np.corrcoef(y_prob_manual, y_prob_sklearn)[0, 1]
ax2.text(0.05, 0.95, f'Correlation: {correlation:.6f}', transform=ax2.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Subplot 3: Performance Metrics
ax3 = fig.add_subplot(133)
metrics = ['Accuracy (%)', 'Iterations']
manual_metrics = [accuracy_manual * 100, len(manual_history)]
sklearn_metrics = [accuracy_sklearn * 100, sklearn_model.n_iter_[0]]
bars1 = ax3.bar(x_pos[:2] - width/2, manual_metrics, width, label='Manual', color='steelblue', alpha=0.8)
bars2 = ax3.bar(x_pos[:2] + width/2, sklearn_metrics, width, label='Scikit-Learn', color='darkorange', alpha=0.8)
ax3.set_xlabel('Metrics', fontsize=12)
ax3.set_ylabel('Value', fontsize=12)
ax3.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
ax3.set_xticks(x_pos[:2])
ax3.set_xticklabels(metrics)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
comparison_plot_path = os.path.join(output_dir, 'comparison_visualization.png')
plt.savefig(comparison_plot_path, dpi=150, bbox_inches='tight')
print(f"Comparison visualization saved to '{comparison_plot_path}'")

# Visualization 2: 3D Decision Surfaces
sigmoid = lambda z: 1 / (1 + np.exp(-z))
fig2 = plt.figure(figsize=(18, 7))
xx1, xx2 = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
X_grid = np.column_stack([xx1.ravel(), xx2.ravel()])
X_grid_with_bias = np.column_stack([np.ones(X_grid.shape[0]), X_grid])
Z_manual = sigmoid(np.dot(X_grid_with_bias, beta_manual)).reshape(xx1.shape)
Z_sklearn = sklearn_model.predict_proba(X_grid)[:, 1].reshape(xx1.shape)

group0, group1 = y == 0, y == 1

# Plot 1: Manual
ax1_3d = fig2.add_subplot(121, projection='3d')
ax1_3d.scatter(X[group0, 0], X[group0, 1], y_prob_manual[group0], c='blue', marker='o', label='Class 0', alpha=0.6, s=20)
ax1_3d.scatter(X[group1, 0], X[group1, 1], y_prob_manual[group1], c='red', marker='^', label='Class 1', alpha=0.6, s=20)
ax1_3d.plot_surface(xx1, xx2, Z_manual, alpha=0.3, cmap='viridis', edgecolor='none')
ax1_3d.contour(xx1, xx2, Z_manual, levels=[0.5], colors='black', linewidths=2, linestyles='--')
ax1_3d.set_xlabel('x1', fontsize=10)
ax1_3d.set_ylabel('x2', fontsize=10)
ax1_3d.set_zlabel('Probability', fontsize=10)
ax1_3d.set_title('Manual Implementation\nDecision Surface', fontsize=12, fontweight='bold')
ax1_3d.legend(fontsize=9)
ax1_3d.set_xlim(0, 1)
ax1_3d.set_ylim(0, 1)
ax1_3d.set_zlim(0, 1)
ax1_3d.view_init(elev=20, azim=45)

# Plot 2: Scikit-learn
ax2_3d = fig2.add_subplot(122, projection='3d')
ax2_3d.scatter(X[group0, 0], X[group0, 1], y_prob_sklearn[group0], c='blue', marker='o', label='Class 0', alpha=0.6, s=20)
ax2_3d.scatter(X[group1, 0], X[group1, 1], y_prob_sklearn[group1], c='red', marker='^', label='Class 1', alpha=0.6, s=20)
ax2_3d.plot_surface(xx1, xx2, Z_sklearn, alpha=0.3, cmap='viridis', edgecolor='none')
ax2_3d.contour(xx1, xx2, Z_sklearn, levels=[0.5], colors='black', linewidths=2, linestyles='--')
ax2_3d.set_xlabel('x1', fontsize=10)
ax2_3d.set_ylabel('x2', fontsize=10)
ax2_3d.set_zlabel('Probability', fontsize=10)
ax2_3d.set_title('Scikit-Learn\nDecision Surface', fontsize=12, fontweight='bold')
ax2_3d.legend(fontsize=9)
ax2_3d.set_xlim(0, 1)
ax2_3d.set_ylim(0, 1)
ax2_3d.set_zlim(0, 1)
ax2_3d.view_init(elev=20, azim=45)

plt.tight_layout()
decision_surface_path = os.path.join(output_dir, 'comparison_decision_surfaces.png')
plt.savefig(decision_surface_path, dpi=150, bbox_inches='tight')
print(f"Decision surface comparison saved to '{decision_surface_path}'")
plt.show()

# Final Analysis
print("\n" + "="*60 + "\nANALYSIS\n" + "="*60)
print("\nKey Findings:")
print(f"1. Both implementations achieve {accuracy_manual*100:.2f}% accuracy")
print(f"2. Coefficient correlation between methods:")
print(f"   - β0 difference: {abs(beta_manual[0] - sklearn_model.intercept_[0]):.6f}")
print(f"   - β1 difference: {abs(beta_manual[1] - sklearn_model.coef_[0][0]):.6f}")
print(f"   - β2 difference: {abs(beta_manual[2] - sklearn_model.coef_[0][1]):.6f}")
print(f"3. Probability prediction correlation: {correlation:.6f}")
print(f"4. Manual implementation converged in {len(manual_history)} iterations")
print(f"5. Scikit-learn converged in {sklearn_model.n_iter_[0]} iterations")

if correlation > 0.999:
    print("\n✓ Excellent agreement! Both implementations produce nearly identical results.")
elif correlation > 0.99:
    print("\n✓ Very good agreement between manual and library implementations.")
else:
    print("\n⚠ Some differences detected. This may be due to different optimization approaches.")

print("\n" + "="*60 + "\nCOMPARISON COMPLETE\n" + "="*60)
print(f"\nGenerated files:\n  1. {comparison_path}\n  2. {comparison_plot_path}\n  3. {decision_surface_path}")
