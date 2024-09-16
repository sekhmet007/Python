import numpy as np
import cv2


def gauss_pivot(A, b):
    n = len(A)

    for i in range(n):
        # Find the maximum element for pivot
        max_row = max(range(i, n), key=lambda r: abs(A[r][i]))
        if A[max_row][i] == 0:
            raise ValueError("Matrix is singular and cannot be solved.")

        # Swap rows
        A[i], A[max_row] = A[max_row], A[i]
        b[i], b[max_row] = b[max_row], b[i]

        # Proceed with elimination
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


def validate_control_points(start_points, end_points):
    if len(start_points) < 3 or len(end_points) < 3:
        raise ValueError("Au moins trois points de contrôle sont nécessaires")


def is_resolvable(A):
    return np.linalg.det(A) != 0


def are_points_colinear(p1, p2, p3):
    # Calcul de l'aire du triangle formé par p1, p2, p3
    # Aire = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
    area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
    return area < 1e-5  # Un seuil très petit pour considérer comme colinéaire


def get_affine_transform(start, end):
    if len(start) < 3 or len(end) < 3:
        raise ValueError("Au moins trois points de contrôle sont nécessaires pour une transformation affine.")

    # Vérifier si les points sont colinéaires ou identiques
    for i in range(2):
        if are_points_colinear(start[i], start[i + 1], end[i]) or are_points_colinear(start[i], start[i + 1], end[i + 1]):
            raise ValueError("Les points de contrôle sont colinéaires ou identiques, ce qui rend la matrice singulière.")

    A = []
    b = []
    for (x, y), (x_prime, y_prime) in zip(start, end):
        A.append([x, y, 1, 0, 0, 0])
        A.append([0, 0, 0, x, y, 1])
        b.append(x_prime)
        b.append(y_prime)

    A = np.array(A)
    b = np.array(b)

    # Utilisation de numpy.linalg.solve pour résoudre le système d'équations
    transform = solve_upper_triangular(A, b)
    transform_matrix = transform.reshape(2, 3)

    return transform_matrix


def validate_triangle_points(triangles):
    for triangle in triangles:
        if are_points_colinear(*triangle):
            return False
    return True


def calculate_intermediate_vertices(start_triangles, end_triangles, alpha):
    intermediate_triangles = []
    for start_triangle, end_triangle in zip(start_triangles, end_triangles):
        intermediate_triangle = []
        for start_vertex, end_vertex in zip(start_triangle, end_triangle):
            intermediate_x = (1 - alpha) * start_vertex[0] + alpha * end_vertex[0]
            intermediate_y = (1 - alpha) * start_vertex[1] + alpha * end_vertex[1]
            intermediate_vertex = (intermediate_x, intermediate_y)
            intermediate_triangle.append(intermediate_vertex)
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


def display_points(image, points, window_name="Image"):
    for point in points:
        cv2.circle(image, point, radius=5, color=(0, 255, 0), thickness=-1)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def warp_and_blend_triangle(img1, img2, tri1, tri2, tri_intermediate, alpha):

    warp_mat1 = cv2.getAffineTransform(np.float32(tri1),
                                       np.float32(tri_intermediate))
    warp_mat2 = cv2.getAffineTransform(np.float32(tri2),
                                       np.float32(tri_intermediate))

    rows, cols, ch = img1.shape
    warped1 = cv2.warpAffine(img1, warp_mat1, (cols, rows))
    warped2 = cv2.warpAffine(img2, warp_mat2, (cols, rows))

    mask = np.zeros_like(img1, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(tri_intermediate), (255, 255, 255))

    warped1 = cv2.bitwise_and(warped1, mask)
    warped2 = cv2.bitwise_and(warped2, mask)
    result = cv2.addWeighted(warped1, alpha, warped2, 1 - alpha, 0)

    return result


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


def calculate_pixel_color(img1, img2, point, transform1, transform2, alpha):

    point_start = np.dot(transform1, [point[0], point[1], 1])
    point_end = np.dot(transform2, [point[0], point[1], 1])

    color_start = get_pixel_color(img1, point_start[:2])
    color_end = get_pixel_color(img2, point_end[:2])

    return (1 - alpha) * color_start + alpha * color_end


def morph_image(img1, img2, triangles_start, triangles_end, triangles_intermediate, transformations, alpha):
    morphed_image = np.zeros_like(img1)
    for tri_start, tri_end, tri_intermediate in zip(triangles_start, triangles_end, triangles_intermediate):

        transform_to_start = affine_transformation_matrix(tri_intermediate, tri_start)
        transform_to_end = affine_transformation_matrix(tri_intermediate, tri_end)

        min_x = int(max(min(v[0] for v in tri_intermediate), 0))
        max_x = int(min(max(v[0] for v in tri_intermediate), img1.shape[1] - 1))
        min_y = int(max(min(v[1] for v in tri_intermediate), 0))
        max_y = int(min(max(v[1] for v in tri_intermediate), img1.shape[0] - 1))

        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                if is_point_inside_triangle((x, y), tri_intermediate):
                    color = calculate_pixel_color(img1, img2, (x, y), transform_to_start, transform_to_end, alpha)
                    morphed_image[y, x] = color.astype(np.uint8)
    return morphed_image


def verify_points_within_bounds(points, img_shape):
    h, w = img_shape[:2]
    for x, y in points:
        if not (0 <= x < w and 0 <= y < h):
            raise ValueError(f"Le point ({x}, {y}) est hors des limites de l'image avec les dimensions {img_shape}")


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
        next(file)  # Skip the first line with the number of triangles
        for line in file:
            indices = [int(i) - 1 for i in line.strip().split()]  # Assurez-vous de retirer les espaces avec strip() et de séparer correctement avec split()
            triangle_koala = [points_koala[index] for index in indices]
            triangle_tigre = [points_tigre[index] for index in indices]
            triangles_koala.append(triangle_koala)
            triangles_tigre.append(triangle_tigre)
    return triangles_koala, triangles_tigre


def draw_triangles(image, triangles):
    for triangle in triangles:
        pts = np.array([triangle[0], triangle[1], triangle[2]], dtype=np.int32)
        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=1)


def main():
    # Charger les images
    img_koala = load_image("koala.ppm")
    img_tigre = load_image("tigre.ppm")

    # Lire les points de contrôle
    points_koala, points_tigre = read_points_from_file("points.txt")
    display_points(img_koala, points_koala, window_name="Image_koala")
    display_points(img_tigre, points_tigre, window_name="Image_tigre")

    triangles_koala, triangles_tigre = read_triangles("triangles.txt", points_koala, points_tigre)

    print("Triangles Koala:")
    for tri in triangles_koala:
        print(tri)

    print("Triangles Tigre:")
    for tri in triangles_tigre:
        print(tri)

    # Vérifier la validité des triangles
    if not validate_triangle_points(triangles_koala):
        raise ValueError("Les triangles de l'image Koala ne sont pas valides.")
    if not validate_triangle_points(triangles_tigre):
        raise ValueError("Les triangles de l'image Tigre ne sont pas valides.")

    img_koala_with_triangles = img_koala.copy()
    img_tigre_with_triangles = img_tigre.copy()
    draw_triangles(img_koala_with_triangles, triangles_koala)
    draw_triangles(img_tigre_with_triangles, triangles_tigre)

    cv2.imshow("Koala avec Triangles", img_koala_with_triangles)
    cv2.imshow("Tigre avec Triangles", img_tigre_with_triangles)
    cv2.waitKey(0)  # Attend jusqu'à ce qu'une touche soit pressée
    cv2.destroyAllWindows()

    # Boucle sur les valeurs alpha pour créer les images intermédiaires
    for i in range(0, 101, 4):
        alpha = i / 100.0

        # Calculer les triangles intermédiaires
        triangles_intermediate = calculate_intermediate_vertices(triangles_koala, triangles_tigre, alpha)

        # Calculer les transformations affines
        transformations = calculate_affine_transformations(triangles_intermediate, triangles_koala, triangles_tigre)

        # Morphing de l'image
        morphed_image = morph_image(img_koala, img_tigre, triangles_koala, triangles_tigre, triangles_intermediate, transformations, alpha)

        # Enregistrer l'image
        cv2.imwrite(f"morphing_{i}.ppm", morphed_image)


if __name__ == "__main__":
    main()
