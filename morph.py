from multiprocessing import Pool
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import Delaunay


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
    """Converts RGB numpy image to PIL image"""
    return Image.fromarray(np.uint8(255 * img.clip(0, 1)))


def resize(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize image to (height, width) using bicubic interpolation."""
    return np.dstack(
        [
            np.array(Image.fromarray(chan).resize((size[1], size[0]), Image.BICUBIC))
            for chan in img.transpose(2, 0, 1)
        ]
    )


def extend_border(points: np.ndarray):
    border = []
    for i in np.linspace(0, 1000, 9):
        border.extend([[0, i], [1000, 1000 - i], [1000 - i, 0], [i, 1000]])
    return np.vstack([points, np.array(border)])


def point_in_simplex(p: np.ndarray, simplex: np.ndarray):
    """Returns if position `p` is in simplex with corners at `simplex`."""

    assert p.shape == (2,)
    assert simplex.shape == (3, 2)

    EPS = 1e-6
    a, b, c = simplex

    # Converts to barycentric coordinates
    A = (-b[1] * c[0] + a[1] * (-b[0] + c[0]) + a[0] * (b[1] - c[1]) + b[0] * c[1]) / 2
    sign = -1 if A < 0 else 1
    s = (a[1] * c[0] - a[0] * c[1] + (c[1] - a[1]) * p[0] + (a[0] - c[0]) * p[1]) * sign
    t = (a[0] * b[1] - a[1] * b[0] + (a[1] - b[1]) * p[0] + (b[0] - a[0]) * p[1]) * sign

    return s + EPS > 0 and t + EPS > 0 and (s + t) < 2 * A * sign + EPS


def identify_simplex(point: np.ndarray, points: np.ndarray, simplicies):
    """Utility function to check what simplex `point` lies in"""
    for i, simplex in enumerate(simplicies):
        if point_in_simplex(point, points[simplex]):
            return i
    raise SystemExit("Error: identify_simplex failed")


def interp_color(img: np.ndarray, point: np.ndarray):
    """
    Sample pixel color of `img` at `point`.
    Bilinear interpolation is used if `point` is non-integer.
    """
    x = np.clip(point[0], 0, img.shape[0] - 2)
    y = np.clip(point[1], 0, img.shape[1] - 2)

    x1, y1 = np.floor(x).astype(int), np.floor(y).astype(int)
    x2, y2 = x1 + 1, y1 + 1

    color = np.array([0, 0, 0], dtype=np.float32)
    color += (x2 - x) * (y2 - y) * img[x1, y1]
    color += (x - x1) * (y2 - y) * img[x2, y1]
    color += (x2 - x) * (y - y1) * img[x1, y2]
    color += (x - x1) * (y - y1) * img[x2, y2]

    return color


def generate_frame(
    src_img: np.ndarray,
    dst_img: np.ndarray,
    src_mesh: np.ndarray,
    dst_mesh: np.ndarray,
    alpha: float,
):
    src_mesh = extend_border(src_mesh)
    dst_mesh = extend_border(dst_mesh)
    target_mesh = (1 - alpha) * src_mesh + alpha * dst_mesh

    triangles = Delaunay((src_mesh + dst_mesh) / 2)
    N = len(triangles.simplices)

    with Pool() as p:
        labels = p.starmap(
            identify_simplex,
            [
                (np.array([i, j]), target_mesh, triangles.simplices)
                for i in range(1000)
                for j in range(1000)
            ],
        )
        labels = np.array(labels).reshape((1000, 1000))

    pixels = [[] for _ in range(N)]
    for x in range(1000):
        for y in range(1000):
            pixels[labels[x, y]].append([x, y])
    pixels = [np.array(row) for row in pixels]

    point_mat = lambda points: np.vstack([points.T, np.ones(points.shape[0])])
    img = np.zeros((1000, 1000, 3))
    for i in range(N):
        if len(pixels[i]) == 0:
            continue

        from_target = np.linalg.inv(point_mat(target_mesh[triangles.simplices[i]]))
        to_src = point_mat(src_mesh[triangles.simplices[i]])
        to_dst = point_mat(dst_mesh[triangles.simplices[i]])
        src = to_src @ from_target @ point_mat(pixels[i])
        dst = to_dst @ from_target @ point_mat(pixels[i])

        for j, pixel in enumerate(pixels[i]):
            src_color = interp_color(src_img, src[[1, 0], j])
            dst_color = interp_color(dst_img, dst[[1, 0], j])
            img[*pixel] = (1 - alpha) * src_color + alpha * dst_color

    return img.transpose((1, 0, 2))


if __name__ == "__main__":
    src_img = resize(read_image("imgs/idina_menzel.png"), (1000, 1000))
    dst_img = resize(read_image("imgs/jamie_muscato.png"), (1000, 1000))
    src_mesh = read_points("points/idina_menzel.csv")
    dst_mesh = read_points("points/jamie_muscato.csv")
    frame = generate_frame(src_img, dst_img, src_mesh, dst_mesh, 0.5)

    plt.clf()
    plt.axis((0, 1000, 1000, 0))
    plt.imshow(numpy_to_image(frame))
    plt.savefig("out/inv_sample_color.png")
    plt.show()
