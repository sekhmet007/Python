import math
import matplotlib.pyplot as plt

def midpoint(p1, p2):
    return [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]

def rotate_point(point, center, angle):
    x0, y0 = center
    x, y = point
    cos_theta = math.cos(angle)
    sin_theta = math.sin(angle)
    x_new = (x - x0) * cos_theta - (y - y0) * sin_theta + x0
    y_new = (x - x0) * sin_theta + (y - y0) * cos_theta + y0
    return [x_new, y_new]

def von_koch(segment, n):
    if n == 0:
        return [segment[0], segment[1]]

    points = []

    # Calcule les coordonnées des points intermédiaires
    P, Q = segment
    O = [(2 * P[0] + Q[0]) / 3, (2 * P[1] + Q[1]) / 3]
    M = midpoint(P, Q)
    theta = math.pi / 3
    M_prime = rotate_point(M, O, theta)

    # Appel récursif sur les segments
    points.extend(von_koch([P, O], n - 1))
    points.extend(von_koch([O, M_prime], n - 1))
    points.extend(von_koch([M_prime, M], n - 1))
    points.extend(von_koch([M, Q], n - 1))

    return points

# Points de départ
P0 = [0, 0]
P1 = [1, 0]

# Générer les points de la courbe de Koch
koch_points = von_koch([P0, P1], 4)  # Changer le niveau d'itération selon vos besoins (dernier chiffre)

# Affichage graphique de la courbe de Koch
x, y = zip(*koch_points)  # Séparation des coordonnées x et y
plt.plot(x, y)
plt.title("Koch")
plt.gca().set_aspect('equal')  # Assurez-vous que l'aspect est égal
plt.axis('off')  # Masquer les axes
plt.show()