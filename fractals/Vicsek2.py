import matplotlib.pyplot as plt

def draw_cross(x, y, length):
    x_points = [x - length / 6, x + length / 6, x, x, x]
    y_points = [y - length / 6, y - length / 6, y + length / 6, y - length / 6, y - length / 6]
    plt.plot(x_points, y_points, 'b')

def vicsek(x, y, length, n):
    if n == 0:
        draw_cross(x, y, length)
        return

    vicsek(x, y, length / 3, n - 1)
    vicsek(x + length / 3, y, length / 3, n - 1)
    vicsek(x - length / 3, y, length / 3, n - 1)
    vicsek(x, y + length / 3, length / 3, n - 1)
    vicsek(x, y - length / 3, length / 3, n - 1)

plt.figure()
vicsek(0, 0, 250, 3)

plt.title("Vicsek.2")
plt.gca().set_aspect('equal')
plt.axis('off')
plt.show()