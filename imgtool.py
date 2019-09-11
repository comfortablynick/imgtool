#!/usr/bin/env python3
"""Compress/watermark images using vips."""

import pyvips
import logging
from dataclasses import dataclass
from typing import Tuple, NamedTuple, Any

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


class RGB(NamedTuple):
    """Red, green, blue colors."""

    r: int
    g: int
    b: int


class CMYK(NamedTuple):
    """Cyan, magenta, yellow, black colors."""

    c: int
    m: int
    y: int
    k: int


@dataclass
class TextWatermark:
    """Specify settings of text watermark."""

    text: str
    fg_color: RGB = RGB(0, 0, 0)
    x_margin: int = 25
    y_margin: int = 25
    font: str = "sans 12"
    rotate: int = 0
    opacity: float = 0.7
    replicate: bool = False

    def add_to_image(self, im: pyvips.Image) -> pyvips.Image:
        """Add watermark to supplied image."""
        text = pyvips.Image.text(
            self.text, width=im.width, dpi=300, font=f"{self.font}"
        )
        text = text.rotate(self.rotate)
        text = (text * self.opacity).cast("uchar")
        text = text.embed(
            self.x_margin,
            (im.height - text.height) - self.y_margin,
            im.width,
            im.height,
        )

        if self.replicate:
            text = text.replicate(
                1 + im.width / text.width, 1 + im.height / text.height
            )
            text = text.crop(0, 0, im.width, im.height)

        # we want to blend into the visible part of the image and leave any alpha
        # channels untouched ... we need to split im into two parts
        # guess how many bands from the start of im contain visible colour information
        if im.bands >= 4 and im.interpretation == pyvips.Interpretation.CMYK:
            # cmyk image
            n_visible_bands = 4
            text_color: Any = list(rgb_to_cmyk(self.fg_color))
        elif im.bands < 4:
            # rgb image
            n_visible_bands = 3
            text_color = list(self.fg_color)
        else:
            # mono image
            n_visible_bands = 1
            text_color = rgb_to_grayscale(self.fg_color)
        LOG.info("Watermark fg_color: %s (original: %s)", text_color, self.fg_color)

        # split into image and alpha
        if im.bands > n_visible_bands:
            alpha = im.extract_band(n_visible_bands, n=im.bands - n_visible_bands)
            im = im.extract_band(0, n=n_visible_bands)
        else:
            alpha = None

        im = text.ifthenelse(text_color, im, blend=True)

        # reattach alpha
        if alpha:
            im = im.bandjoin(alpha)

        return im


def rgb_to_cmyk(rgb: RGB) -> CMYK:
    """Convert rgb color to cmyk color."""
    float2int = lambda n: int(round(n))

    # Convert from 0..255 to 0..1
    LOG.info("RBG value: %s", rgb)
    rgb_adj = (rgb.r / 255, rgb.g / 255, rgb.b / 255)
    k = float2int(1 - max(rgb_adj))
    c = float2int((1 - rgb_adj[0] - k) / (1 - k))
    m = float2int((1 - rgb_adj[1] - k) / (1 - k))
    y = float2int((1 - rgb_adj[2] - k) / (1 - k))
    cmyk = CMYK(c, m, y, k)
    LOG.info("CMYK value: %s", cmyk)
    return cmyk


def rgb_to_grayscale(rgb: RGB) -> int:
    """Convert rgb to grayscale with weighted method."""
    LOG.info("RGB value: %s", rgb)
    gray = int(round((0.299 * rgb.r) + (0.587 * rgb.g) + (0.114 * rgb.b)))
    LOG.info("Grayscale value: %d", gray)
    return gray


def main():
    """Entry point."""
    in_filename = "./test/chin_class.jpg"
    out_filename = "./test/chin_class_edited.jpg"
    watermark_text = "© 2019 Nick Murphy | murphpix.com"
    LOG.info("Adding watermark '%s'", watermark_text)

    im = pyvips.Image.new_from_file(in_filename, access=pyvips.Access.SEQUENTIAL)
    watermark = TextWatermark(watermark_text)
    watermark.font = "Source Sans Pro 16"
    watermark.rotate = -90
    watermark.opacity = 1

    im = watermark.add_to_image(im)
    im.write_to_file(out_filename)


if __name__ == "__main__":
    main()
