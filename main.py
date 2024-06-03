# The purpose of this script is to act as a testing ground and a place to
# Quickly generate visualizations for reference. This is not the final code.

from multiprocessing import Pool
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backend_bases import MouseButton
from PIL import Image
from scipy.ndimage import affine_transform
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
    menzel = extend_border(read_points("points/idina_menzel.csv"))
    muscato = extend_border(read_points("points/jamie_muscato.csv"))
    points = (menzel + muscato) / 2

    triangles = Delaunay(points)
    N = len(triangles.simplices)

    pixels = [[] for _ in range(N)]
    for i in range(1000):
        for j in range(1000):
            pixels[triangles.find_simplex([i, j])].append([i, j])
    pixels = [np.array(row) for row in pixels]

    dsts = [np.empty(1) for _ in range(N)]
    srcs = [np.empty(1) for _ in range(N)]
    for i in range(N):
        mid = points[triangles.simplices[i]]

        def point_mat(points: np.ndarray) -> np.ndarray:
            return np.vstack([points.T, np.ones(points.shape[0])])

        M = point_mat(menzel[triangles.simplices[i]]) @ np.linalg.inv(point_mat(mid))
        srcs[i] = M @ point_mat(pixels[i])

        M = point_mat(muscato[triangles.simplices[i]]) @ np.linalg.inv(point_mat(mid))
        dsts[i] = M @ point_mat(pixels[i])

    imgs = []
    for i, alpha in enumerate(np.linspace(0, 1, 40, endpoint=True)):
        plt.clf()
        plt.axis((0, 1000, 1000, 0))
        for src, dst in zip(srcs, dsts):
            points = (1 - alpha) * src + alpha * dst
            plt.scatter(points[0], points[1], s=1)
        fname = f"out/triangle_transform/{i:03}.png"
        plt.savefig(fname)
        imgs.append(read_image(fname))

    write_gif("out/triangle_transform.gif", imgs + imgs[::-1])


def run_transform_image():
    a = 2 * np.pi / 3
    M = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

    H, W, C = MUSCATO.shape

    dst = np.zeros((H, W, C), dtype=np.float32)
    center = np.array([H // 2, W // 2])
    offset = center - np.dot(center, M)
    for i in range(C):
        dst[:, :, i] = affine_transform(
            MUSCATO[:, :, i],
            matrix=M.T,
            order=1,
            offset=offset,
            cval=0.0,
            output=np.float32,
        )
    plt.imshow(numpy_to_image(dst))
    plt.show()


def point_in_simplex(p, simplex):
    # Uses Barycentric Coordinates to calculate
    p0, p1, p2 = simplex

    A = (
        1
        / 2
        * (
            -p1[1] * p2[0]
            + p0[1] * (-p1[0] + p2[0])
            + p0[0] * (p1[1] - p2[1])
            + p1[0] * p2[1]
        )
    )
    sign = -1 if A < 0 else 1
    s = (
        p0[1] * p2[0] - p0[0] * p2[1] + (p2[1] - p0[1]) * p[0] + (p0[0] - p2[0]) * p[1]
    ) * sign
    t = (
        p0[0] * p1[1] - p0[1] * p1[0] + (p0[1] - p1[1]) * p[0] + (p1[0] - p0[0]) * p[1]
    ) * sign

    EPS = 1e-6

    return s + EPS > 0 and t + EPS > 0 and (s + t) < 2 * A * sign + EPS


def identify_simplex(point, points, simplicies):
    for i, simplex in enumerate(simplicies):
        if point_in_simplex(point, points[simplex]):
            return i
    exit(-1)


def test_identify_simplex():
    menzel = extend_border(read_points("points/idina_menzel.csv"))
    muscato = extend_border(read_points("points/jamie_muscato.csv"))
    points = (menzel + muscato) / 2

    triangles = Delaunay(points)

    test_points = 1000 * np.random.rand(10000, 2)
    for point in test_points:
        alg = identify_simplex(point, points, triangles.simplices)
        delaunay = triangles.find_simplex(point)
        assert alg == delaunay

    print("identify_simplex works")


def run_inv_sample_frame():
    menzel = extend_border(read_points("points/idina_menzel.csv"))
    muscato = extend_border(read_points("points/jamie_muscato.csv"))
    points = (menzel + muscato) / 2
    alpha = 0.3
    target = (1 - alpha) * menzel + alpha * muscato

    triangles = Delaunay(points)
    N = len(triangles.simplices)

    pixels = [[] for _ in range(N)]
    for x in range(0, 1000, 10):
        for y in range(0, 1000, 10):
            pixels[
                identify_simplex(np.array([x, y]), target, triangles.simplices)
            ].append([x, y])
    pixels = [np.array(row) for row in pixels]

    srcs = [np.empty(2) for _ in range(N)]
    dsts = [np.empty(2) for _ in range(N)]
    for i in range(N):

        def point_mat(points: np.ndarray) -> np.ndarray:
            return np.vstack([points.T, np.ones(points.shape[0])])

        # mid = points[triangles.simplices[i]]
        # MID_TO_MENZEL = point_mat(menzel[triangles.simplices[i]]) @ np.linalg.inv(point_mat(mid))
        # MID_TO_MUSCATO = point_mat(muscato[triangles.simplices[i]]) @ np.linalg.inv(point_mat(mid))
        # TARGET_TO_MID = point_mat(mid) @ np.linalg.inv(point_mat(target[triangles.simplices[i]]))
        # srcs[i] = MID_TO_MENZEL @ TARGET_TO_MID @ point_mat(pixels[i])
        # dsts[i] = MID_TO_MUSCATO @ TARGET_TO_MID @ point_mat(pixels[i])

        TARGET_TO_MENZEL = point_mat(menzel[triangles.simplices[i]]) @ np.linalg.inv(
            point_mat(target[triangles.simplices[i]])
        )
        TARGET_TO_MUSCATO = point_mat(muscato[triangles.simplices[i]]) @ np.linalg.inv(
            point_mat(target[triangles.simplices[i]])
        )

        assert len(pixels[i]) != 0

        srcs[i] = TARGET_TO_MENZEL @ point_mat(pixels[i])
        dsts[i] = TARGET_TO_MUSCATO @ point_mat(pixels[i])

    plt.clf()
    plt.axis((0, 1000, 1000, 0))
    for src in srcs:
        plt.scatter(src[0], src[1], s=5)
    plt.savefig("out/inv_sample_menzel.png")
    plt.show()

    plt.clf()
    plt.axis((0, 1000, 1000, 0))
    for dst in dsts:
        plt.scatter(dst[0], dst[1], s=5)
    plt.savefig("out/inv_sample_muscato.png")
    plt.show()


def interpolate_pixel_color(img: np.ndarray, point: np.ndarray):
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


def generate_frame(alpha: float):
    menzel = extend_border(read_points("points/idina_menzel.csv"))
    muscato = extend_border(read_points("points/jamie_muscato.csv"))
    alpha = 0.5
    target = (1 - alpha) * menzel + alpha * muscato

    triangles = Delaunay((menzel + muscato) / 2)
    N = len(triangles.simplices)

    with Pool() as p:
        labels = p.starmap(
            identify_simplex,
            [
                (np.array([i, j]), target, triangles.simplices)
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

    img = np.ones((1000, 1000, 3))
    for i in range(N):

        def point_mat(points: np.ndarray) -> np.ndarray:
            return np.vstack([points.T, np.ones(points.shape[0])])

        TARGET_TO_MENZEL = point_mat(menzel[triangles.simplices[i]]) @ np.linalg.inv(
            point_mat(target[triangles.simplices[i]])
        )
        TARGET_TO_MUSCATO = point_mat(muscato[triangles.simplices[i]]) @ np.linalg.inv(
            point_mat(target[triangles.simplices[i]])
        )

        assert len(pixels[i]) != 0

        src = TARGET_TO_MENZEL @ point_mat(pixels[i])
        dst = TARGET_TO_MUSCATO @ point_mat(pixels[i])
        for j, pixel in enumerate(pixels[i]):
            img[pixel[0], pixel[1]] = (1 - alpha) * interpolate_pixel_color(
                MENZEL, src[[1, 0], j]
            )
            img[pixel[0], pixel[1]] += alpha * interpolate_pixel_color(
                MUSCATO, dst[[1, 0], j]
            )

    img = img.transpose((1, 0, 2))

    return img

def run_inv_color_sample_frame():
    plt.clf()
    plt.axis((0, 1000, 1000, 0))
    plt.imshow(numpy_to_image(generate_frame(0.5)))
    plt.savefig(f"out/inv_sample_color.png")
    plt.show()

def run_generate_gif():
    frames = []
    for i, alpha in enumerate(np.linspace(0, 1, 10, endpoint=True)):
        fname = f"out/final_transform/{i:03}.png"
        frame = generate_frame(alpha)
        frames.append(frame)
        write_image(fname, frame)

    write_gif("out/final_transform.gif", frames)


if __name__ == "__main__":
    # run_naive_cross_fade()
    # run_draw_points()
    # run_verify_points()
    # run_draw_mesh("idina_menzel", MENZEL)
    # run_draw_mesh("jamie_muscato", MUSCATO)
    # run_average_mesh()
    # run_triangle_transform()
    # run_calculate_simplex_per_pixel()
    # run_transform_image()
    # test_identify_simplex()
    # run_inv_sample_frame()
    # run_inv_color_sample_frame()
    run_generate_gif()
    print("EOF")
