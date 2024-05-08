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


# --------------------------------------------------------------------------------------


def run_naive_cross_fade():
    menzel = resize(read_image("imgs/idina_menzel.png"), (1000, 1000))
    muscato = resize(read_image("imgs/jamie_muscato.png"), (1000, 1000))

    frames = [cross_dissolve(menzel, muscato, a) for a in np.linspace(0, 1)]
    write_gif("out/cross_fade.gif", frames)
    write_image("out/cross_fade.png", cross_dissolve(menzel, muscato, 0.5))


def run_draw_points(fname: str):
    img = resize(read_image(f"imgs/{fname}.png"), (1000, 1000))
    select_points(f"points/{fname}.csv", img)


def run_draw_mesh(fname: str):
    img = resize(read_image(f"imgs/{fname}.png"), (1000, 1000))
    points = pd.read_csv(f"points/{fname}.csv", header=None).to_numpy()

    border = []
    for i in np.linspace(0, 1000, 9):
        border.extend([[0, i], [1000, 1000 - i], [1000 - i, 0], [i, 1000]])
    points = np.vstack([points, np.array(border)])

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
    menzel = resize(read_image(f"imgs/idina_menzel.png"), (1000, 1000))
    muscato = resize(read_image(f"imgs/jamie_muscato.png"), (1000, 1000))

    menzel_points = read_points("points/idina_menzel.csv")
    muscato_points = read_points("points/jamie_muscato.csv")
    points = (menzel_points + muscato_points) / 2

    border = []
    for i in np.linspace(0, 1000, 9):
        border.extend([[0, i], [1000, 1000 - i], [1000 - i, 0], [i, 1000]])
    points = np.vstack([points, np.array(border)])
    menzel_points = np.vstack([menzel_points, np.array(border)])
    muscato_points = np.vstack([muscato_points, np.array(border)])

    triangles = Delaunay(points)

    plt.axis((0, 1000, 1000, 0))

    plt.triplot(points[:, 0], points[:, 1], triangles.simplices)
    plt.savefig(f"out/average_mesh.png")
    plt.show()

    plt.triplot(menzel_points[:, 0], menzel_points[:, 1], triangles.simplices)
    plt.imshow(numpy_to_image(menzel))
    plt.savefig(f"out/average_idina_menzel_mesh.png")
    plt.show()

    plt.triplot(muscato_points[:, 0], muscato_points[:, 1], triangles.simplices)
    plt.imshow(numpy_to_image(muscato))
    plt.savefig(f"out/average_jamie_muscato_mesh.png")
    plt.show()

if __name__ == "__main__":
    # run_naive_cross_fade()
    # run_draw_points("idina_menzel")
    # run_draw_points("jamie_muscato")
    # run_verify_points()
    # run_draw_mesh("idina_menzel")
    # run_draw_mesh("jamie_muscato")
    run_average_mesh()
