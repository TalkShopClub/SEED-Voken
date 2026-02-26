"""
Smart resize utilities: area-based resize with dimensions rounded to multiples of ds_factor.
Used across IBQ data pipelines (PIL, numpy/albumentations).
"""
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

from albumentations.core.transforms_interface import DualTransform


def smart_resize_shape(
    width: int,
    height: int,
    area: int = 512 * 512,
    ds_factor: int = 16,
) -> Tuple[int, int]:
    """Compute (new_width, new_height) for area-based resize, divisible by ds_factor."""
    if width <= 0 or height <= 0 or (width * height) < area:
        return (width, height)
    aspect_ratio = width / height
    new_height = int((area / aspect_ratio) ** 0.5)
    new_width = int(new_height * aspect_ratio)
    new_height = ((new_height + ds_factor // 2) // ds_factor) * ds_factor
    new_width = ((new_width + ds_factor // 2) // ds_factor) * ds_factor
    # Ensure at least ds_factor to avoid degenerate size
    new_height = max(new_height, ds_factor)
    new_width = max(new_width, ds_factor)
    # If crop/resize size would be larger than image, do not crop — return original dimensions
    if new_width > width or new_height > height:
        return (width, height)
    return (new_width, new_height)


def smart_resize(
    image: Image.Image,
    area: int = 512 * 512,
    ds_factor: int = 16,
) -> Image.Image:
    """Resize PIL image to approximate area while keeping aspect ratio; dimensions are multiples of ds_factor."""
    width, height = image.size
    new_width, new_height = smart_resize_shape(width, height, area=area, ds_factor=ds_factor)
    return image.resize((new_width, new_height), Image.BICUBIC)


class SmartResize(DualTransform):
    """
    Albumentations transform: resize image (and optional targets) using smart_resize logic.
    Replaces SmallestMaxSize with area-based resize and ds_factor-aligned dimensions.
    """

    def __init__(
        self,
        area: int = 512 * 512,
        ds_factor: int = 16,
        interpolation: int = cv2.INTER_CUBIC,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.area = area
        self.ds_factor = ds_factor
        self.interpolation = interpolation

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        h, w = img.shape[:2]
        new_w, new_h = smart_resize_shape(w, h, area=self.area, ds_factor=self.ds_factor)
        return cv2.resize(img, (new_w, new_h), interpolation=self.interpolation)

    def apply_to_mask(self, mask: np.ndarray, **params) -> np.ndarray:
        h, w = mask.shape[:2]
        new_w, new_h = smart_resize_shape(w, h, area=self.area, ds_factor=self.ds_factor)
        return cv2.resize(
            mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST
        )

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("area", "ds_factor", "interpolation")
