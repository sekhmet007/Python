import matplotlib.pyplot as plt
import random

# Number of points to draw
n = 100

# Starting point
p = (0, 0)

# Lists to store x and y coordinates
x_points = []
y_points = []

for i in range(n):
    x, y = p

    # Store the points
    x_points.append(x)
    y_points.append(y)

    r = random.uniform(0, 1)
    if r < 0.01:
        p = (0, 0.16 * y)
    elif r < 0.86:
        p = (0.85 * x + 0.04 * y, -0.04 * x + 0.85 * y + 1.6)
    elif r < 0.93:
        p = (0.2 * x - 0.26 * y, 0.23 * x + 0.22 * y + 1.6)
    else:
        p = (-0.15 * x + 0.28 * y, 0.26 * x + 0.24 * y + 0.44)

   # if i % 1000 == 0:
        # Update the plot for every 1000 moves
        plt.scatter(x_points, y_points, s=5, c='green', marker='.')
        plt.title("Barnsley: 10^2")
        plt.axis('equal')
        plt.axis('off')
        plt.pause(0.001)
        x_points = []
        y_points = []

plt.show()