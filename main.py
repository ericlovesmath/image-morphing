from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backend_bases import MouseButton
from PIL import Image
from scipy.spatial import Delaunay

# --------------------------------------------------------------------------------------


def read_points(path: str) -> np.ndarray:
    """Read array of points of an CSV, reading verticies"""
    return pd.read_csv(path, header=None).to_numpy()


def read_image(path: str) -> np.ndarray:
    """Read an image an array of linear RGB radiance values ∈ [0,1]"""
    return np.array(Image.open(path), dtype=np.float32) / 255


def write_image(path: str, img: np.ndarray) -> None:
    """Saves image from an array of linear RGB radiance values ∈ [0,1]"""
    numpy_to_image(img).save(path)


def write_gif(path: str, imgs: List[np.ndarray]) -> None:
    """Saves gif from an array of linear RGB radiance values ∈ [0,1]"""
    head, *tail = [numpy_to_image(img) for img in imgs]
    head.save(path, save_all=True, append_images=tail, duration=5)


def numpy_to_image(img: np.ndarray) -> Image.Image:
    return Image.fromarray(np.uint8(255 * img.clip(0, 1)))


def resize(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize image to (height, width) using bicubic interpolation."""

    return np.dstack(
        [
            np.array(Image.fromarray(chan).resize((size[1], size[0]), Image.BICUBIC))
            for chan in img.transpose(2, 0, 1)
        ]
    )


# --------------------------------------------------------------------------------------


def cross_dissolve(src: np.ndarray, dst: np.ndarray, alpha: float) -> np.ndarray:
    """
    Cross dissolves between `src` and `dst` with factor `alpha`.
    Assumes `src.shape == dst.shape`
    """
    return (1 - alpha) * src + alpha * dst


# --------------------------------------------------------------------------------------


def select_points(path: str, img: np.ndarray):

    points = []

    def on_click(event):
        if event.button is MouseButton.LEFT and event.inaxes:
            plt.scatter(event.xdata, event.ydata, color="red")
            points.append([event.xdata, event.ydata])
            plt.draw()

    plt.imshow(numpy_to_image(img))
    plt.connect("button_press_event", on_click)
    plt.show()

    np.savetxt(path, np.array(points), delimiter=",")


def extend_border(points: np.ndarray):
    border = []
    for i in np.linspace(0, 1000, 9):
        border.extend([[0, i], [1000, 1000 - i], [1000 - i, 0], [i, 1000]])
    return np.vstack([points, np.array(border)])


# --------------------------------------------------------------------------------------

MENZEL = resize(read_image("imgs/idina_menzel.png"), (1000, 1000))
MUSCATO = resize(read_image("imgs/jamie_muscato.png"), (1000, 1000))


def run_naive_cross_fade():
    frames = [cross_dissolve(MENZEL, MUSCATO, a) for a in np.linspace(0, 1)]
    write_gif("out/cross_fade.gif", frames)
    write_image("out/cross_fade.png", cross_dissolve(MENZEL, MUSCATO, 0.5))


def run_draw_points():
    select_points(f"points/idina_menzel.csv", MENZEL)
    select_points(f"points/jamie_muscato.csv", MUSCATO)


def run_draw_mesh(fname: str, img: np.ndarray):
    points = read_points(f"points/{fname}.csv")
    points = extend_border(points)

    triangles = Delaunay(points)

    plt.imshow(numpy_to_image(img))
    plt.triplot(points[:, 0], points[:, 1], triangles.simplices, linewidth=1)

    plt.axis((0, 1000, 1000, 0))
    plt.savefig(f"out/{fname}_mesh.png")
    plt.show()


def run_verify_points():
    plt.axis((0, 1000, 1000, 0))
    points1 = read_points("points/idina_menzel.csv")
    points2 = read_points("points/jamie_muscato.csv")
    for p1, p2 in zip(points1, points2):
        plt.plot(p1[0], p1[1], "bo", ms=2)
        plt.plot(p2[0], p2[1], "ro", ms=2)
        plt.pause(0.2)


def run_average_mesh():
    menzel = extend_border(read_points("points/idina_menzel.csv"))
    muscato = extend_border(read_points("points/jamie_muscato.csv"))
    points = (menzel + muscato) / 2

    triangles = Delaunay(points)

    plt.axis((0, 1000, 1000, 0))

    plt.triplot(points[:, 0], points[:, 1], triangles.simplices)
    plt.savefig(f"out/average_mesh.png")
    plt.show()

    plt.triplot(menzel[:, 0], menzel[:, 1], triangles.simplices)
    plt.imshow(numpy_to_image(MENZEL))
    plt.savefig(f"out/average_idina_menzel_mesh.png")
    plt.show()

    plt.triplot(muscato[:, 0], muscato[:, 1], triangles.simplices)
    plt.imshow(numpy_to_image(MUSCATO))
    plt.savefig(f"out/average_jamie_muscato_mesh.png")
    plt.show()


def run_triangle_transform():
    src = np.array([[1, 1], [5, 3], [3, 5]])
    dst = np.array([[6, 1], [7, 7], [3, 4]])

    M = np.vstack([src.T, np.ones(3)]) @ np.linalg.inv(np.vstack([dst.T, np.ones(3)]))

    # Generates 100 random points in `dst
    x = np.sort(np.random.rand(2, 100), axis=0)
    dst_points = np.column_stack([x[0], x[1] - x[0], 1.0 - x[1]]) @ dst

    dst_points = np.vstack([dst_points.T, np.ones(100)])
    src_points = M @ dst_points

    plt.axis((0, 8, 0, 8))

    for alpha in np.linspace(0, 1, 100, endpoint=True):
        points = (1 - alpha) * src_points + alpha * dst_points
        plt.clf()
        plt.triplot(src[:, 0], src[:, 1], [[0, 1, 2]])
        plt.triplot(dst[:, 0], dst[:, 1], [[0, 1, 2]])
        plt.scatter(points[0], points[1], color="r")
        plt.pause(0.5 if alpha == 0 or alpha == 1 else 0.02)


def run_calculate_simplex_per_pixel():

    # TODO: This is incorrect since it just doesn't use `muscato` for morphing
    # at all, as scipy's Delaunay doesn't have a `find_simplex` alternative

    menzel = extend_border(read_points("points/idina_menzel.csv"))
    muscato = extend_border(read_points("points/jamie_muscato.csv"))
    points = (menzel + muscato) / 2

    triangles = Delaunay(points)
    N = len(triangles.simplices)

    pixels = [[] for _ in range(N)]
    for i in range(1000):
        for j in range(1000):
            pixels[triangles.find_simplex([i, j])].append([i, j])

    dsts = [np.empty(1) for _ in range(N)]
    srcs = [np.empty(1) for _ in range(N)]
    for i in range(N):
        src = menzel[triangles.simplices[i]]
        dst = points[triangles.simplices[i]]

        M = np.vstack([src.T, np.ones(3)]) @ np.linalg.inv(np.vstack([dst.T, np.ones(3)]))

        dst = np.vstack([np.array(pixels[i]).T, np.ones(len(pixels[i]))])
        src = M @ dst

        dsts[i] = dst
        srcs[i] = src

    for alpha in np.linspace(0, 1, 20, endpoint=True):
        plt.clf()
        plt.axis((0, 1000, 1000, 0))
        for src, dst in zip(srcs, dsts):
            points = (1 - alpha) * src + alpha * dst
            plt.scatter(points[0], points[1], s=1)
        plt.pause(1 if alpha == 0 or alpha == 1 else 0.02)


if __name__ == "__main__":
    # run_naive_cross_fade()
    # run_draw_points()
    # run_verify_points()
    # run_draw_mesh("idina_menzel", MENZEL)
    # run_draw_mesh("jamie_muscato", MUSCATO)
    # run_average_mesh()
    # run_triangle_transform()
    run_calculate_simplex_per_pixel()
