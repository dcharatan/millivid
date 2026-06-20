"""This file contains useful layout utilities for images. They are:

- add_border: Add a border to an image.
- cat/hcat/vcat: Join images by arranging them in a line. If the images have different
  sizes, they are aligned as specified (start, end, center). Allows you to specify a gap
  between images.
- add_label: Add a label above an image.

Images are assumed to be float32 tensors with range 0 to 1 and shape
(*batch, channel, height, width).
"""

from pathlib import Path
from string import ascii_letters, digits, punctuation
from typing import Generator, Iterable, Literal, TypeVar

import numpy as np
import torch
from einops import reduce
from jaxtyping import Float, Int
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor

type Alignment = Literal["start", "center", "end"]
type Direction = Literal["horizontal", "vertical"]
type Color = Float[Tensor | list, "#channel"] | Int[
    Tensor | list, "#channel"
] | float | int


EXPECTED_CHARACTERS = digits + punctuation + ascii_letters

T = TypeVar("T")
D = TypeVar("D")


def _pad_color(color: Color, device: torch.device) -> Float[Tensor, "#channel 1 1"]:
    return torch.tensor(color, dtype=torch.float32, device=device).reshape((-1, 1, 1))


def intersperse(iterable: Iterable[T], delimiter: D) -> Generator[T | D, None, None]:
    try:
        it = iter(iterable)
        yield next(it)
        for item in it:
            yield delimiter
            yield item
    except StopIteration:
        return


def direction_to_axis(direction: Direction) -> Literal[-1, -2]:
    return {
        "horizontal": -1,
        "vertical": -2,
    }[direction]


def pad(
    image: Float[Tensor, "*batch channel original_height original_width"],
    direction: Direction,
    align: Alignment,
    target: int,
    color: Color,
) -> Float[Tensor, "*batch channel padded_height padded_width"]:
    """Pad the image to the desired length on the target axis."""
    axis = direction_to_axis(direction)
    delta = target - image.shape[axis]
    before = {
        "start": 0,
        "center": delta // 2,
        "end": delta,
    }[align]

    # Create an image with the padded shape and desired color.
    padded_shape = list(image.shape)
    padded_shape[axis] = target
    padded_image = torch.empty(padded_shape, dtype=image.dtype, device=image.device)
    padded_image[:] = _pad_color(color, image.device)

    # Insert the original image into the padded image at the correct location.
    selector = [slice(None, None) for _ in padded_shape]
    selector[axis] = slice(before, before + image.shape[axis])
    padded_image[tuple(selector)] = image

    return padded_image


def cat(
    images: Iterable[Float[Tensor, "*#batch #channel _ _"]],
    direction: Direction,
    align: Alignment = "center",
    gap: int = 8,
    color: Color = 1,
    border: int = 0,
) -> Float[Tensor, "*batch channel height width"]:
    """Arrange images in a line. The interface resembles a CSS div with flexbox."""

    # Ensure that there's at least one image.
    images = list(images)
    assert images
    device = images[0].device

    # Find the axis and cross axis.
    axis = direction_to_axis(direction)
    cross_direction = {
        "horizontal": "vertical",
        "vertical": "horizontal",
    }[direction]
    cross_axis = direction_to_axis(cross_direction)

    # Pad the images along the cross axis.
    target = max(image.shape[cross_axis] for image in images)
    images = [pad(image, cross_direction, align, target, color) for image in images]

    # Intersperse separators to create gaps.
    if gap > 0:
        # Create a separator with the desired size.
        *_, channel, _, _ = images[0].shape
        separator_shape = [channel, gap, gap]
        separator_shape[cross_axis] = target
        separator = torch.empty(separator_shape, dtype=images[0].dtype, device=device)
        separator[:] = _pad_color(color, device)

        # Insert the separator.
        images = list(intersperse(images, separator))

    # Broadcast and concatenate the images.
    broad = [image.shape[:-2] for image in images]
    broad = torch.broadcast_shapes(*broad)
    images = [torch.broadcast_to(im, (*broad, *im.shape[-2:])) for im in images]
    images = torch.cat(images, axis=axis)

    # Add a border if desired.
    if border > 0:
        images = add_border(images, border, color)

    return images


def hcat(
    images: Iterable[Float[Tensor, "*#batch #channel _ _"]],
    align: Literal["start", "center", "end", "top", "bottom"] = "start",
    gap: int = 8,
    color: Color = 1,
    border: int = 0,
) -> Float[Tensor, "*batch channel height width"]:
    """Shorthand for horizontal concatenation."""
    return cat(
        images,
        "horizontal",
        align={
            "start": "start",
            "center": "center",
            "end": "end",
            "top": "start",
            "bottom": "end",
        }[align],
        gap=gap,
        color=color,
        border=border,
    )


def vcat(
    images: Iterable[Float[Tensor, "*#batch #channel _ _"]],
    align: Literal["start", "center", "end", "left", "right"] = "start",
    gap: int = 8,
    color: Color = 1,
    border: int = 0,
) -> Float[Tensor, "*batch channel height width"]:
    """Shorthand for vertical concatenation."""
    return cat(
        images,
        "vertical",
        align={
            "start": "start",
            "center": "center",
            "end": "end",
            "left": "start",
            "right": "end",
        }[align],
        gap=gap,
        color=color,
        border=border,
    )


def add_border(
    image: Float[Tensor, "*batch channel height width"],
    border: int = 8,
    color: Color = 1,
) -> Float[Tensor, "*batch channel new_height new_width"]:
    """Add a border to the image."""
    *batch, channel, height, width = image.shape
    device = image.device

    # Create an empty larger image with the border color.
    padded_shape = (*batch, channel, height + 2 * border, width + 2 * border)
    padded_image = torch.empty(padded_shape, dtype=image.dtype, device=device)
    padded_image[:] = _pad_color(color, device)

    # Paste the original image intot he padded image.
    padded_image[..., border : border + height, border : border + width] = image

    return padded_image


def draw_label(
    text: str,
    font: Path,
    font_size: int,
    device: torch.device,
    font_color: Color = 0,
    background: Color = 1,
) -> Float[Tensor, "channel height width"]:
    """Draw a monochrome white label on a black background with no border."""
    try:
        font = ImageFont.truetype(str(font), font_size)
    except OSError:
        font = ImageFont.load_default()
    left, _, right, _ = font.getbbox(text)
    width = right - left
    _, top, _, bottom = font.getbbox(EXPECTED_CHARACTERS)
    height = bottom - top
    image = Image.new("RGB", (width, height), color="black")
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), text, font=font, fill="white")
    image = reduce(np.array(image, dtype=np.float32) / 255, "h w c -> h w", "mean")
    label = torch.tensor(image, dtype=torch.float32, device=device)
    label_color = _pad_color(font_color, device)
    background_color = _pad_color(background, device)
    return label * label_color + (1 - label) * background_color


def add_label(
    image: Float[Tensor, "*batch channel width height"],
    label: str,
    font: Path = Path("assets/Inter-Regular.otf"),
    font_size: int = 24,
    font_color: Color = 0,
    background: Color = 1,
    gap: int = 4,
    align: Alignment = "left",
    border: int = 0,
) -> Float[Tensor, "*batch channel width_with_label height_with_label"]:
    device = image.device
    label = draw_label(
        label,
        font,
        font_size,
        device,
        font_color=font_color,
        background=background,
    )
    return vcat((label, image), align=align, gap=gap, color=background, border=border)


def make_grid(
    images: Iterable[Float[Tensor, "*#batch #channel _ _"]],
    num_columns: int = 4,
    gap: int = 8,
    color: Color = 1,
    border: int = 0,
    align: Literal["start", "center", "end", "left", "right"] = "start",
) -> Float[Tensor, "*batch channel height width"]:
    images = list(images)
    chunks = []
    for i in range(0, len(images), num_columns):
        chunk = images[i : i + num_columns]
        chunk = hcat(chunk, gap=gap, color=color, align=align)
        chunks.append(chunk)
    return vcat(chunks, gap=gap, color=color, border=border)
