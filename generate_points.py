import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Set random seed for reproducibility
np.random.seed(42)

# Create output directory
output_dir = 'results_output'
os.makedirs(output_dir, exist_ok=True)

# Number of points per group
n_points_per_group = 2000

# Generate Group 0 (bottom-left cluster)
# Points centered around (0.2, 0.2) with some spread
x1_group0 = np.random.normal(0.2, 0.08, n_points_per_group)
x2_group0 = np.random.normal(0.2, 0.08, n_points_per_group)

# Generate Group 1 (top-right cluster)
# Points centered around (0.8, 0.8) with some spread
x1_group1 = np.random.normal(0.8, 0.08, n_points_per_group)
x2_group1 = np.random.normal(0.8, 0.08, n_points_per_group)

# Combine the groups
x1 = np.concatenate([x1_group0, x1_group1])
x2 = np.concatenate([x2_group0, x2_group1])
R = np.concatenate([np.zeros(n_points_per_group), np.ones(n_points_per_group)])

# Clip values to ensure they're between 0 and 1
x1 = np.clip(x1, 0, 1)
x2 = np.clip(x2, 0, 1)

# Create DataFrame
df = pd.DataFrame({
    'x1': x1,
    'x2': x2,
    'R': R.astype(int)
})

# Display first few rows
print("Generated Dataset:")
print(df.head(10))
print(f"\nDataset shape: {df.shape}")
print(f"\nClass distribution:\n{df['R'].value_counts()}")

# Save to CSV
csv_path = os.path.join(output_dir, 'random_points.csv')
df.to_csv(csv_path, index=False)
print(f"\nDataset saved to '{csv_path}'")

# Visualize the data
plt.figure(figsize=(8, 8))
plt.scatter(df[df['R']==0]['x1'], df[df['R']==0]['x2'],
            c='blue', label='Group 0', alpha=0.6, s=50)
plt.scatter(df[df['R']==1]['x1'], df[df['R']==1]['x2'],
            c='red', label='Group 1', alpha=0.6, s=50)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Two Separable Groups of Points')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 1)
plt.ylim(0, 1)
plot_path = os.path.join(output_dir, 'points_visualization.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Visualization saved to '{plot_path}'")
plt.show()
