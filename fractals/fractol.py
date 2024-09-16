
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random

# Constants
MAX_ITER = 100000  # Augmentez si nécessaire
FRACTAL_BARNSEY = "Barnsley"

class FractalApp:
    def __init__(self):
        self.fractal_type = FRACTAL_BARNSEY

    def generate_barnsley(self):
        # Transformation matrices for the Barnsley Fern
        transformations = [
            (0.01, 0.16, 0, 0, 0, 0.01),  # f1
            (0.85, 0.04, 0, -0.04, 1.6, 0.85),  # f2
            (0.2, -0.26, 0, 0.23, 1.6, 0.22),  # f3
            (-0.15, 0.28, 0, 0.26, 0.44, 0.24)  # f4
        ]

        # Starting point
        x, y = 0, 0

        # Lists to store points
        x_points = []
        y_points = []

        # Generate the Barnsley Fern
        for _ in range(MAX_ITER):
            # Choose a transformation with weighted probability
            prob = random.uniform(0, 1)
            if prob < 0.01:
                a, b, c, d, e, f = transformations[0]
            elif prob < 0.86:
                a, b, c, d, e, f = transformations[1]
            elif prob < 0.93:
                a, b, c, d, e, f = transformations[2]
            else:
                a, b, c, d, e, f = transformations[3]

            # Apply the transformation
            x_new = a * x + b * y + c
            y_new = d * x + e * y + f

            # Store the new point
            x_points.append(x_new)
            y_points.append(y_new)

            # Mettez à jour les coordonnées
            x, y = x_new, y_new

        # Afficher la fractale de Barnsley
        plt.scatter(x_points, y_points, s=1, c='green', marker='.')
        plt.title("Fractale de Barnsley")
        plt.show()

if __name__ == "__main__":
    app = FractalApp()
    app.generate_barnsley()
