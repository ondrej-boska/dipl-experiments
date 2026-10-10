"""
Training image augmentation, ported from the TensorFlow Zeus (zeus/model/construct_tf_dataset.py).

Uses the same `--augment` syntax as Zeus, e.g. "h:8,rotate:1,v:4,de,en3:0.2,n:0.01,c:-1:1,b:-0.5:0.2":
a comma-separated pipeline of `name:arg1:arg2` filters, each applied with a 50:50 chance, in the given order.
As in Zeus, augmentation is the very last step before the model, so pixel amounts refer to the model's input
image (the height-normalized grayscale stave for Zeus, the stave band for end-to-end MuSViT).

- `h:8` Horizontal shift by at most 8 pixels left or right
- `rotate:1` Rotation by at most 1 degree in either direction
- `v:4` Vertical shift by at most 4 pixels up or down
- `de` Dilatation/erosion in a random direction on an ellipse with x semi-axis 1 and y semi-axis 0.5
- `en3:0.2` For a random probability of up to 0.2, negate pixels whose 3x3 neighborhood is not uniformly
  white or black (boundary-sensitive noise)
- `n:0.01` Negate each pixel with a random probability of up to 0.01 (global noise)
- `c:-1:1` Adjust contrast by a factor of 2^u, u uniform in [-1, 1]
- `b:-0.5:0.2` Adjust brightness by adding u, uniform in [-0.5, 0.2]
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# Number of numeric parameters of each filter
FILTERS: dict[str, int] = {"h": 1, "v": 1, "rotate": 1, "de": 0, "en3": 1, "n": 1, "c": 2, "b": 2}


class ZeusAugmentation:
    """
    Applies a Zeus augmentation pipeline to a (C, H, W) image in [0, 1] with white = 1.
    Random per-pixel masks are shared by all channels, so a grayscale image in RGB stays gray.

    Zeus's horizontal shift pads the left side and crops the left side, so the width changes by up to the shift.
    With `keep_width` (for the fixed-width MuSViT stave bands), the content is shifted by the same amount,
    but the right side is padded or cropped instead, so that the width stays.
    """
    def __init__(self, spec: str, keep_width: bool = False):
        self.spec = spec
        self.keep_width = keep_width
        self.filters: list[tuple[str, list[float]]] = []
        for part in spec.split(","):
            if not part:
                continue
            name, *params = part.split(":")
            if name not in FILTERS:
                raise ValueError(f"The augmentation '{name}' is unknown. Known: {', '.join(FILTERS)}.")
            if len(params) < FILTERS[name]:
                raise ValueError(f"The augmentation '{name}' needs {FILTERS[name]} parameter(s), got '{part}'.")
            self.filters.append((name, [float(p) for p in params[:FILTERS[name]]]))

    def __bool__(self) -> bool:
        return bool(self.filters)

    def __call__(self, image: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        def uniform(low: float = 0.0, high: float = 1.0) -> float:
            return low + (high - low) * torch.rand((), generator=generator).item()

        def randint(high: int) -> int:
            """Uniform integer in [0, high)."""
            return int(torch.randint(high, (), generator=generator))

        for name, params in self.filters:
            if uniform() >= 0.5:
                continue
            if name == "h":
                a = int(params[0])
                s = randint(2 * a + 1)
                if self.keep_width:
                    image = F.pad(image, (a, a), value=1.0)[..., s:s + image.shape[-1]]
                else:
                    image = F.pad(image, (a, 0), value=1.0)[..., s:]
            elif name == "v":
                a = int(params[0])
                s = randint(2 * a + 1)
                image = F.pad(image, (0, 0, a, a), value=1.0)[..., s:s + image.shape[-2], :]
            elif name == "rotate":
                # Keras RandomRotation(degrees / 360) rotates by up to the given degrees about the image center
                angle = math.radians(uniform(-params[0], params[0]))
                image = _warp(image, angle=angle)
            elif name == "b":
                image = (image + uniform(params[0], params[1])).clamp(0.0, 1.0)
            elif name == "c":
                # tf.image.adjust_contrast: (x - mean) * factor + mean, with the mean of each channel
                factor = 2 ** uniform(params[0], params[1])
                mean = image.mean(dim=(-2, -1), keepdim=True)
                image = ((image - mean) * factor + mean).clamp(0.0, 1.0)
            elif name == "n":
                p = uniform(0.0, params[0])
                negate = torch.rand((1, *image.shape[-2:]), generator=generator) < p
                image = torch.where(negate, 1.0 - image, image)
            elif name == "en3":
                p = uniform(0.0, params[0])
                # 3x3 mean as tf.nn.avg_pool2d with SAME padding, which averages over the valid pixels only
                local = F.avg_pool2d(image.mean(dim=0, keepdim=True)[None], 3, 1, padding=1, count_include_pad=False)[0]
                uniform_area = (local <= 0.1) | (local >= 0.9)
                negate = ~uniform_area & (torch.rand(local.shape, generator=generator) < p)
                image = torch.where(negate, 1.0 - image, image)
            elif name == "de":
                d = uniform(-math.pi / 2, math.pi / 2)
                moved = _warp(image, shift=(math.cos(d), 0.5 * math.sin(d)))
                if uniform() >= 0.5:
                    image = torch.maximum(image, moved)  # erosion of the (black) ink
                else:
                    image = (image + moved - 1.0).clamp(0.0, 1.0)  # dilatation of the ink
        return image

    def __repr__(self) -> str:
        return f"ZeusAugmentation('{self.spec}', keep_width={self.keep_width})"


def _warp(image: torch.Tensor, angle: float = 0.0, shift: tuple[float, float] = (0.0, 0.0)) -> torch.Tensor:
    """
    Bilinear resampling of a (C, H, W) image, filled with white outside it:
    output(p) = input(R(angle) p + shift), with p in pixels relative to the image center.
    """
    H, W = image.shape[-2:]
    cos, sin = math.cos(angle), math.sin(angle)
    # The pixel-space transform expressed in the normalized [-1, 1] coordinates of affine_grid
    theta = torch.tensor(
        [[cos, -sin * H / W, 2.0 * shift[0] / W], [sin * W / H, cos, 2.0 * shift[1] / H]],
        dtype=image.dtype,
    )[None]
    grid = F.affine_grid(theta, [1, image.shape[0], H, W], align_corners=False)
    # Resampling the inverted image with zero padding fills with white, also in the bilinear border pixels
    inverted = F.grid_sample(1.0 - image[None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return 1.0 - inverted[0]
