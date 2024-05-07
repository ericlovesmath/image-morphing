from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backend_bases import MouseButton
from scipy.spatial import Delaunay
from PIL import Image

# --------------------------------------------------------------------------------------


def read_image(path: str) -> np.ndarray:
    """Read an image an array of linear RGB radiance values ∈ [0,1]"""
    return (np.array(Image.open(path), dtype=np.float32) / 255) ** 2.2


def write_image(path: str, img: np.ndarray) -> None:
    """Saves image from an array of linear RGB radiance values ∈ [0,1]"""
    numpy_to_image(img).save(path)


def write_gif(path: str, imgs: List[np.ndarray]) -> None:
    """Saves gif from an array of linear RGB radiance values ∈ [0,1]"""
    head, *tail = [numpy_to_image(img) for img in imgs]
    head.save(path, save_all=True, append_images=tail, duration=5)


def numpy_to_image(img: np.ndarray) -> Image.Image:
    return Image.fromarray(np.uint8(255 * img.clip(0, 1) ** (1 / 2.2)))


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


def run_draw_points():
    menzel = resize(read_image("imgs/idina_menzel.png"), (1000, 1000))
    select_points("points/idina_menzel.csv", menzel)


if __name__ == "__main__":
    # run_naive_cross_fade()
    run_draw_points()
