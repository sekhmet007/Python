import numpy as np
import cv2


def gauss_pivot(A, b):
    n = len(A)

    for i in range(n):
        max_row = max(range(i, n), key=lambda r: abs(A[r][i]))
        if A[max_row][i] == 0:
            raise ValueError("Matrix is singular and cannot be solved.")

        A[i], A[max_row] = A[max_row], A[i]
        b[i], b[max_row] = b[max_row], b[i]

        for j in range(i + 1, n):
            ratio = A[j][i] / A[i][i]
            for k in range(i, n):
                A[j][k] -= ratio * A[i][k]
            b[j] -= ratio * b[i]

    return A, b


def solve_upper_triangular(A, b):

    n = len(A)
    x = [0 for _ in range(n)]

    for i in range(n - 1, -1, -1):
        x[i] = b[i]
        for j in range(i + 1, n):
            x[i] -= A[i][j] * x[j]
        x[i] = x[i] / A[i][i]

    return x


def are_points_colinear(p1, p2, p3):
    # Calcul de l'aire du triangle formé par p1, p2, p3
    # Aire = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
    area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) +
                     p3[0] * (p1[1] - p2[1]))
    return area < 1e-5  # Un seuil très petit pour considérer comme colinéaire


def calculate_intermediate_vertices(start_triangles, end_triangles, alpha,
                                    img_shape):
    intermediate_triangles = []
    h, w = img_shape
    for start_triangle, end_triangle in zip(start_triangles, end_triangles):
        intermediate_triangle = []
        for start_vertex, end_vertex in zip(start_triangle, end_triangle):
            intermediate_x = (1 - alpha) * start_vertex[0] + alpha * end_vertex[0]
            intermediate_y = (1 - alpha) * start_vertex[1] + alpha * end_vertex[1]

            # Vérification des limites des points
            intermediate_x = min(max(intermediate_x, 0), w - 1)
            intermediate_y = min(max(intermediate_y, 0), h - 1)

            intermediate_vertex = (intermediate_x, intermediate_y)
            intermediate_triangle.append(intermediate_vertex)

        # Vérification de la non-colinéarité
        if not are_points_colinear(*intermediate_triangle):
            intermediate_triangles.append(tuple(intermediate_triangle))

    return intermediate_triangles


def affine_transformation_matrix(from_points, to_points):
    A = []
    b = []
    for (x, y), (x_prime, y_prime) in zip(from_points, to_points):
        A.append([x, y, 1, 0, 0, 0])
        A.append([0, 0, 0, x, y, 1])
        b.append(x_prime)
        b.append(y_prime)

    A = np.array(A)
    b = np.array(b)

    A_pivot, b_pivot = gauss_pivot(A.tolist(), b.tolist())
    transform = solve_upper_triangular(A_pivot, b_pivot)

    transformation_matrix = np.array(transform).reshape(2, 3)

    return transformation_matrix


def calculate_affine_transformations(intermediate_triangles, start_triangles,
                                     end_triangles):

    transformations = []
    for intermediate, start, end in zip(intermediate_triangles,
                                        start_triangles, end_triangles):
        to_intermediate = affine_transformation_matrix(start, intermediate)
        from_intermediate = affine_transformation_matrix(intermediate, end)
        transformations.append((to_intermediate, from_intermediate))

    return transformations


def is_point_inside_triangle(pt, tri):

    (x, y) = pt
    (x1, y1), (x2, y2), (x3, y3) = tri

    total_area = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    area1 = abs((x1 - x) * (y2 - y) - (x2 - x) * (y1 - y))
    area2 = abs((x2 - x) * (y3 - y) - (x3 - x) * (y2 - y))
    area3 = abs((x3 - x) * (y1 - y) - (x1 - x) * (y3 - y))

    return area1 + area2 + area3 == total_area


def get_pixel_color(img, point):

    h, w, _ = img.shape
    x, y = point
    x = min(max(int(x), 0), w - 1)
    y = min(max(int(y), 0), h - 1)

    return img[y, x]


def interpolate_colors(color1, color2, alpha):
    """
    Interpole linéairement entre deux couleurs.

    :param color1: La première couleur (au début du morphing).
    :param color2: La deuxième couleur (à la fin du morphing).
    :param alpha: Le paramètre d'interpolation (0 <= alpha <= 1).
    :return: La couleur résultante.
    """
    if not (0 <= alpha <= 1):
        raise ValueError("Alpha doit être compris entre 0 et 1")

    # Conversion en float pour la précision de calcul
    color1 = np.array(color1, dtype=np.float32)
    color2 = np.array(color2, dtype=np.float32)

    # Interpolation linéaire
    interpolated_color = (1 - alpha) * color1 + alpha * color2

    # Clip pour s'assurer que la couleur reste valide
    return np.clip(interpolated_color, 0, 255).astype(np.uint8)


def calculate_pixel_color(img1, img2, point, transform1, transform2, alpha):
    # Transformer le point pour chaque image source
    point_start = np.dot(transform1, [point[0], point[1], 1])
    point_end = np.dot(transform2, [point[0], point[1], 1])

    # Arrondir les coordonnées des points transformés et s'assurer qu'elles sont dans les limites de l'image
    x_start, y_start = round(point_start[0]), round(point_start[1])
    x_end, y_end = round(point_end[0]), round(point_end[1])

    # S'assurer que les points transformés sont dans les limites de l'image
    h, w, _ = img1.shape
    x_start = min(max(int(x_start), 0), w - 1)
    y_start = min(max(int(y_start), 0), h - 1)
    x_end = min(max(int(x_end), 0), w - 1)
    y_end = min(max(int(y_end), 0), h - 1)

    # Obtenir la couleur dans chaque image source
    color_start = get_pixel_color(img1, (x_start, y_start))
    color_end = get_pixel_color(img2, (x_end, y_end))

    # Interpolation linéaire des couleurs
    return interpolate_colors(color_start, color_end, alpha)


def point_in_triangle(pt, tri):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    b1 = sign(pt, tri[0], tri[1]) < 0.0
    b2 = sign(pt, tri[1], tri[2]) < 0.0
    b3 = sign(pt, tri[2], tri[0]) < 0.0

    return ((b1 == b2) and (b2 == b3))


def morph_image(img1, img2, triangles_start, triangles_end,
                triangles_intermediate, transformations, alpha):
    morphed_image = np.zeros_like(img1)
    h, w, _ = img1.shape

    for tri_start, tri_end, tri_intermediate in zip(triangles_start, triangles_end, triangles_intermediate):
        # Calculer les matrices de transformation pour chaque triangle
        transform_to_start = affine_transformation_matrix(tri_intermediate, tri_start)
        transform_to_end = affine_transformation_matrix(tri_intermediate, tri_end)
        # Parcourir chaque pixel à l'intérieur du rectangle englobant du triangle intermédiaire
        h, w, _ = img1.shape
        for x in range(w):
            for y in range(h):
                if point_in_triangle((x, y), tri_intermediate):
                    color = calculate_pixel_color(img1, img2, (x, y), transform_to_start, transform_to_end, alpha)
                    morphed_image[y, x] = color.astype(np.uint16)

    return morphed_image


def load_image(image_path):
    return cv2.imread(image_path)


def read_points_from_file(file_path):
    points_koala, points_tigre = [], []
    with open(file_path, 'r') as file:
        next(file)
        next(file)
        for line in file:
            x_koala, y_koala, x_tigre, y_tigre = map(int, line.split())
            points_koala.append((x_koala, y_koala))
            points_tigre.append((x_tigre, y_tigre))
    return points_koala, points_tigre


def read_triangles(file_path, points_koala, points_tigre):
    triangles_koala, triangles_tigre = [], []
    with open(file_path, 'r') as file:
        next(file)
        for line in file:
            indices = [int(i) - 0 for i in line.strip().split()]

            if all(0 <= idx < len(points_koala) for idx in indices):
                triangle_koala = [points_koala[index] for index in indices]
                triangle_tigre = [points_tigre[index] for index in indices]
                triangles_koala.append(triangle_koala)
                triangles_tigre.append(triangle_tigre)
    return triangles_koala, triangles_tigre


def draw_triangles(image, triangles):
    h, w, _ = image.shape
    for triangle in triangles:
        if all(isinstance(point, tuple) and len(point) == 2 for point
               in triangle):
            pts = np.array([triangle[0], triangle[1], triangle[2]],
                           dtype=np.int32)
            cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0),
                          thickness=1)


def main():
    # Charger les images
    img_koala = load_image("koala.ppm")
    img_tigre = load_image("tigre.ppm")

    h, w = img_koala.shape[:2]
    # Lire les points de contrôle
    points_koala, points_tigre = read_points_from_file("points.txt")

    triangles_koala, triangles_tigre = read_triangles("triangles.txt", points_koala, points_tigre)

    for i in range(0, 101, 4):
        alpha = i / 100.0

        # Calculer les triangles intermédiaires
        triangles_intermediate = calculate_intermediate_vertices(triangles_koala, triangles_tigre, alpha, (h, w))

        # Calculer les transformations affines
        transformations = calculate_affine_transformations(triangles_intermediate, triangles_koala, triangles_tigre)
        # Morphing de l'image
        morphed_image = morph_image(img_koala, img_tigre, triangles_koala, triangles_tigre, triangles_intermediate, transformations, alpha)

        # Enregistrer l'image
        cv2.imwrite(f"morphing_{i}.ppm", morphed_image)


if __name__ == "__main__":
    main()
