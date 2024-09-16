import math
import matplotlib.pyplot as plt

def midpoint(p1, p2):
    return [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]

def sierpinski(triangle, n):
    if n == 0:
        return [triangle]

    triangles = []

    mid_ab = midpoint(triangle[0], triangle[1])
    mid_bc = midpoint(triangle[1], triangle[2])
    mid_ca = midpoint(triangle[2], triangle[0])

    triangles.extend(sierpinski([triangle[0], mid_ab, mid_ca], n - 1))
    triangles.extend(sierpinski([mid_ab, triangle[1], mid_bc], n - 1))
    triangles.extend(sierpinski([mid_ca, mid_bc, triangle[2]], n - 1))

    return triangles

init_triangle = [[0, 0], [0.5, math.sqrt(3) / 2], [1, 0]]
triangles = sierpinski(init_triangle, 8)# iteration du point

# Affichage graphique des triangles
for t in triangles:
    x, y = zip(*t)  # Séparation des coordonnées x et y
    plt.fill(x, y, 'b')  # Remplissage du triangle

plt.title("Triangle de Sierpinski2")
plt.gca().set_aspect('equal')  # Assurez-vous que l'aspect est égal
plt.axis('off')  # Masquer les axes
plt.show()

