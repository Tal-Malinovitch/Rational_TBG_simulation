"""Visualize the actual distribution of Dirac points in k-space"""

import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Load data
with open('Training_data/grouped_training_data.pkl', 'rb') as f:
    data = pickle.load(f)

samples = data['training_samples']

# Extract all k points
all_k_x = []
all_k_y = []
for sample in samples:
    for point in sample['dirac_points']:
        all_k_x.append(point[0])
        all_k_y.append(point[1])

all_k_x = np.array(all_k_x)
all_k_y = np.array(all_k_y)

print(f"Total Dirac points: {len(all_k_x)}")

# Create 2D histogram
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 2D density plot
ax = axes[0]
h = ax.hist2d(all_k_x, all_k_y, bins=50, cmap='viridis')
ax.set_xlabel('k_x')
ax.set_ylabel('k_y')
ax.set_title('2D Density of Dirac Points')
ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=1)
ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, linewidth=1)
ax.set_aspect('equal')
plt.colorbar(h[3], ax=ax, label='Count')

# k_x distribution
ax = axes[1]
ax.hist(all_k_x, bins=100, alpha=0.7, edgecolor='black')
ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='k_x=0')
ax.set_xlabel('k_x')
ax.set_ylabel('Count')
ax.set_title('k_x Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# k_y distribution
ax = axes[2]
ax.hist(all_k_y, bins=100, alpha=0.7, edgecolor='black', color='orange')
ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='k_y=0')
ax.set_xlabel('k_y')
ax.set_ylabel('Count')
ax.set_title('k_y Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dirac_points_distribution.png', dpi=150, bbox_inches='tight')
print("Saved visualization to dirac_points_distribution.png")

# Statistical analysis of axis alignment
radius = np.sqrt(all_k_x**2 + all_k_y**2)
angle = np.arctan2(all_k_y, all_k_x) * 180 / np.pi

print(f"\n=== AXIS ALIGNMENT ANALYSIS ===")
print(f"Points with |k_x| < 0.05: {np.sum(np.abs(all_k_x) < 0.05)} ({100*np.sum(np.abs(all_k_x) < 0.05)/len(all_k_x):.1f}%)")
print(f"Points with |k_y| < 0.05: {np.sum(np.abs(all_k_y) < 0.05)} ({100*np.sum(np.abs(all_k_y) < 0.05)/len(all_k_y):.1f}%)")
print(f"Points with BOTH |k_x| < 0.05 AND |k_y| < 0.05: {np.sum((np.abs(all_k_x) < 0.05) & (np.abs(all_k_y) < 0.05))} ({100*np.sum((np.abs(all_k_x) < 0.05) & (np.abs(all_k_y) < 0.05))/len(all_k_x):.1f}%)")

# Check if there's clustering along axes vs diagonals
near_x_axis = np.abs(all_k_y) < 0.05  # Near k_x axis
near_y_axis = np.abs(all_k_x) < 0.05  # Near k_y axis
near_diagonal = np.abs(np.abs(all_k_x) - np.abs(all_k_y)) < 0.05  # Near |k_x|=|k_y|

print(f"\nPoints near k_x axis (|k_y| < 0.05): {np.sum(near_x_axis)} ({100*np.sum(near_x_axis)/len(all_k_x):.1f}%)")
print(f"Points near k_y axis (|k_x| < 0.05): {np.sum(near_y_axis)} ({100*np.sum(near_y_axis)/len(all_k_y):.1f}%)")
print(f"Points near diagonal (||k_x|-|k_y|| < 0.05): {np.sum(near_diagonal)} ({100*np.sum(near_diagonal)/len(all_k_x):.1f}%)")

# Check angle distribution
print(f"\n=== ANGLE DISTRIBUTION ===")
for angle_range in [(0, 15), (15, 30), (30, 45), (45, 60), (60, 75), (75, 90)]:
    count = np.sum((np.abs(angle) >= angle_range[0]) & (np.abs(angle) < angle_range[1]))
    print(f"Angles {angle_range[0]}-{angle_range[1]} degrees: {count} ({100*count/len(angle):.1f}%)")
