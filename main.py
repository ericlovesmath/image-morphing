from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

# ------------------------------------------------------------------------------


def read_image(path: str) -> np.ndarray:
    """Read an image an array of linear RGB radiance values ∈ [0,1]."""
    return (np.array(Image.open(path), dtype=np.float32) / 255) ** 2.2


def write_image(path: str, img: np.ndarray) -> None:
    """Saves image from an array of linear RGB radiance values ∈ [0,1]"""
    _ndarray_to_image(img).save(path)


def write_gif(path: str, imgs: List[np.ndarray]) -> None:
    """Saves gif from an array of linear RGB radiance values ∈ [0,1]"""
    head, *tail = [_ndarray_to_image(img) for img in imgs]
    head.save(path, save_all=True, append_images=tail, duration=5)


def _ndarray_to_image(img: np.ndarray) -> Image.Image:
    return Image.fromarray(np.uint8(255 * img.clip(0, 1) ** (1 / 2.2)))


def resize(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize image to (height, width) using bicubic interpolation."""
    return (
        _resize_channel(img, size)
        if img.ndim == 2
        else np.dstack(
            [_resize_channel(chan, size) for chan in img.transpose(2, 0, 1)]
        )
    )


def _resize_channel(chan: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return np.asarray(
        Image.fromarray(chan).resize((size[1], size[0]), Image.BICUBIC)
    )


# ------------------------------------------------------------------------------


def cross_fade(src: np.ndarray, dst: np.ndarray, alpha: float) -> np.ndarray:
    """
    Cross fades between `src` and `dst` with factor `alpha`.
    Assumes `src.shape == dst.shape`
    """
    return (1 - alpha) * src + alpha * dst


# ------------------------------------------------------------------------------


def run_naive_cross_fade():
    menzel = resize(read_image("imgs/idina_menzel.png"), (1000, 1000))
    muscato = resize(read_image("imgs/jamie_muscato.png"), (1000, 1000))

    write_image("naive_cross_fade.png", cross_fade(menzel, muscato, 0.5))

    fades = [cross_fade(menzel, muscato, a) for a in np.linspace(0, 1, 20)]
    write_gif("naive_cross_fade.gif", fades)

if __name__ == "__main__":
    run_naive_cross_fade()
