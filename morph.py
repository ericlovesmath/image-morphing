#!/usr/bin/env python3

import argparse
import os
from multiprocessing import Pool
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backend_bases import MouseButton
from PIL import Image
from scipy.spatial import Delaunay


def read_mesh(path: str) -> np.ndarray:
    """Read array of points of an CSV, reading verticies"""
    return pd.read_csv(path, header=None).to_numpy()


def read_image(path: str) -> np.ndarray:
    """Read an image an array of linear RGB radiance values in [0,1]"""
    return np.array(Image.open(path), dtype=np.float32) / 255


def write_image(path: str, img: np.ndarray) -> None:
    """Saves image from an array of linear RGB radiance values in [0,1]"""
    numpy_to_image(img).save(path)


def write_gif(path: str, imgs: List[np.ndarray]) -> None:
    """Saves gif from an array of linear RGB radiance values in [0,1]"""
    head, *tail = [numpy_to_image(img) for img in imgs]
    head.save(path, save_all=True, append_images=tail, duration=5)


def numpy_to_image(img: np.ndarray) -> Image.Image:
    """Converts RGB numpy image to PIL image"""
    return Image.fromarray(np.uint8(255 * img.clip(0, 1)))


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


def extend_border(points: np.ndarray):
    border = []
    for i in np.linspace(0, 1000, 9):
        border.extend([[0, i], [1000, 1000 - i], [1000 - i, 0], [i, 1000]])
    return np.vstack([points, np.array(border)])


def generate_frame(
    src_img: np.ndarray,
    dst_img: np.ndarray,
    src_mesh: np.ndarray,
    dst_mesh: np.ndarray,
    alpha: float,
    sample: bool = True,
):
    assert np.shape(src_img) == np.shape(dst_img)
    assert np.shape(src_mesh) == np.shape(dst_mesh)

    H, W, _ = np.shape(src_img)
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
                for i in range(H)
                for j in range(W)
            ],
        )
        labels = np.array(labels).reshape((H, W))

    simplex_to_pixels = [[] for _ in range(N)]
    for x in range(H):
        for y in range(W):
            simplex_to_pixels[labels[x, y]].append([x, y])
    simplex_to_pixels = [np.array(row) for row in simplex_to_pixels]

    mat = lambda points: np.vstack([points.T, np.ones(points.shape[0])])
    img = np.zeros((H, W, 3))
    for i in range(N):
        if len(simplex_to_pixels[i]) == 0:
            continue

        from_target = np.linalg.inv(mat(target_mesh[triangles.simplices[i]]))
        to_src = mat(src_mesh[triangles.simplices[i]])
        to_dst = mat(dst_mesh[triangles.simplices[i]])
        src = to_src @ from_target @ mat(simplex_to_pixels[i])
        dst = to_dst @ from_target @ mat(simplex_to_pixels[i])

        for j, pixel in enumerate(simplex_to_pixels[i]):
            src_color = interp_color(src_img, src[[1, 0], j])
            if sample:
                dst_color = interp_color(dst_img, dst[[1, 0], j])
                img[*pixel] = (1 - alpha) * src_color + alpha * dst_color
            else:
                img[*pixel] = src_color

    return img.transpose((1, 0, 2))


def generate_gif(
    src_img: np.ndarray,
    dst_img: np.ndarray,
    src_mesh: np.ndarray,
    dst_mesh: np.ndarray,
    nframes: int,
    fname: str,
    sample: bool = True,
):
    if not os.path.exists("frames"):
        raise SystemExit("Error: local folder 'frames' missing")

    frames = []
    for i, alpha in enumerate(np.linspace(0, 1, nframes, endpoint=True)):
        print(f"Generating frame {i + 1}/{nframes}...")
        frame = generate_frame(src_img, dst_img, src_mesh, dst_mesh, alpha, sample)
        frames.append(frame)
        write_image(f"frames/{fname}_{i:03}.png", frame)
    print(f"Generating {fname}.gif...")
    write_gif(f"{fname}.gif", frames)


def select_mesh(img: np.ndarray, path: str):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Draw meshes and interpolate images.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "src_img",
        help="Path to image used as source",
    )
    parser.add_argument(
        "dst_img",
        help="Path to image used as destination",
    )
    parser.add_argument(
        "src_mesh",
        help="Path to mesh corresponding to src_img",
    )
    parser.add_argument(
        "dst_mesh",
        help="Path to mesh corresponding to dst_img",
    )
    parser.add_argument(
        "--task",
        choices=["interp_images", "interp_meshes", "draw_mesh"],
        default="interp_images",
        help="""interp_images: Interpolate between images (default).
interp_meshes: Ignore dst_img and sample interpolation from only
               src_img, effectively animating the image.
draw_mesh: Select points on image to draw mesh, ignores dst_* args. """,
    )
    parser.add_argument(
        "--nframes",
        default=10,
        type=int,
        help="Number of frames to generate (default: 10)",
    )
    parser.add_argument(
        "--fname",
        default="result",
        help="Name of generated gif and frames (default: 'result')",
    )

    args = parser.parse_args()

    if args.task == "draw_mesh":
        select_mesh(read_image(args.src_img), args.src_mesh)
    else:
        src_img = read_image(args.src_img)
        src_mesh = read_mesh(args.src_mesh)
        dst_mesh = read_mesh(args.dst_mesh)
        nframes = args.nframes
        fname = args.fname

        if args.task == "interp_images":
            dst_img = read_image(args.dst_img)
            generate_gif(src_img, dst_img, src_mesh, dst_mesh, nframes, fname)
        elif args.task == "interp_meshes":
            generate_gif(src_img, src_img, src_mesh, dst_mesh, nframes, fname, False)
