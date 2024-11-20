import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate 600 random points with normal distribution centered at (0,0) and variance of 10
np.random.seed(0)  # 固定隨機數種子以重現結果
num_points = 600
mean = 0
variance = 10
x1 = np.random.normal(mean, np.sqrt(variance), num_points)
x2 = np.random.normal(mean, np.sqrt(variance), num_points)

# Step 2: Calculate distances from the origin
distances = np.sqrt(x1**2 + x2**2)

# Step 3: Assign labels
Y = np.where(distances < 4, 0, 1)

# Step 4: Scatter plot with colors based on labels
plt.figure(figsize=(8, 8))
plt.scatter(x1[Y == 0], x2[Y == 0], c='blue', label='Y=0 (distance < 4)', alpha=0.7, edgecolors='k')
plt.scatter(x1[Y == 1], x2[Y == 1], c='red', label='Y=1 (distance >= 4)', alpha=0.7, edgecolors='k')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')

# Adding labels and legend
plt.title('Scatter Plot of Points with Labels Based on Distance from Origin')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
