#!/usr/bin/env python3
"""Compress/watermark images using vips."""

import pyvips
import logging
from dataclasses import dataclass
from typing import NamedTuple, Any
import argparse
import sys
import os

logging.basicConfig(level=logging.WARNING)
LOG = logging.getLogger(__name__)

# test defaults
in_filename = os.path.expanduser("~/git/imgtool/test/sunset.jpg")
out_filename = os.path.expanduser("~/git/imgtool/test/sunset_edited.jpg")
watermark_text = "© 2019 Nick Murphy | murphpix.com"
suffix = "_edited"


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
    """Text watermark that can be blended with an image."""

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
        LOG.info("Adding watermark '%s'", self.text)
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
    # Convert float to rounded int
    def float2int(n: float):
        return int(round(n))

    LOG.info("RBG value: %s", rgb)
    # Convert from 0..255 to 0..1
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


def resize(im: pyvips.Image, width: int, height: int) -> pyvips.Image:
    """Resize image."""
    resized = im.thumbnail_image(width, height=height)
    return resized


def parse_args(args: list):
    """Parse command line arguments."""
    # flags
    desc = (
        "A command-line utility which uses the vips library to manipulate "
        "images for web vewing. Images can be resampled, resized, and "
        "compressed at custom quality levels. Watermarking can also be added."
    )
    parser = argparse.ArgumentParser(prog="imgtool", description=desc, add_help=False)
    parser.add_argument(
        "-v",
        help="increase logging output to console",
        action="count",
        dest="verbosity",
        default=0,
    )
    parser.add_argument("-h", action="help", help="show this help message and exit")
    parser.add_argument("--help", action="help", help=argparse.SUPPRESS)
    parser.add_argument("-V", action="version", version="%(prog)s 0.0.1")
    parser.add_argument(
        "-s",
        nargs=1,
        help="text suffix appended to INPUT path if no OUTPUT file given",
        metavar="TEXT",
        default=suffix,
    )

    # Positionals
    parser.add_argument(
        "infile",
        help="image file to process",
        nargs=1,
        metavar="INPUT",
        type=argparse.FileType("r"),
        #  default=sys.stdin,
        default=in_filename,
    )
    parser.add_argument(
        "outfile",
        help="file to save processed image",
        nargs="?",
        metavar="OUTPUT",
        type=argparse.FileType("w"),
        #  default=sys.stdout,
        default=out_filename,
    )

    # Image group
    image_group = parser.add_argument_group("General image options")
    image_group.add_argument(
        "-wt",
        help="text to display in watermark",
        default=watermark_text,
        dest="watermark_text",
        metavar="TEXT",
    ),

    # Jpg group
    jpg_group = parser.add_argument_group("Jpeg options")
    jpg_group.add_argument(
        "-q",
        help="Quality setting for jpeg files (an integer between 1 and 100; default: 75)",
        type=int,
        dest="jpg_quality",
        default=75,
        metavar="QUALITY",
    )

    if parser._positionals.title is not None:
        parser._positionals.title = "Arguments"
    if parser._optionals.title is not None:
        parser._optionals.title = "Options"
    parsed = parser.parse_args(args)

    # do rudimentary checks
    if not 0 <= parsed.jpg_quality <= 100:
        parser.exit(
            1, f"Quality (-q) must be within 0-100; found: {parsed.jpg_quality}\n"
        )
    return parsed


def main():
    """Entry point."""
    args = parse_args(sys.argv)
    log_level = 0
    try:
        log_level = (0, 20, 10)[args.verbosity]
    except IndexError:
        log_level = 10
    LOG.setLevel(log_level)
    logging.getLogger("pyvips").setLevel(log_level)
    LOG.debug(args)
    quality = args.jpg_quality
    progressive = True
    no_subsample = True if quality > 75 else False
    strip_exif = False

    im = pyvips.Image.new_from_file(in_filename, access=pyvips.Access.SEQUENTIAL)
    watermark = TextWatermark(args.watermark_text)
    watermark.font = "Source Sans Pro 14"
    watermark.fg_color = RGB(255, 255, 255)
    watermark.rotate = -90
    watermark.opacity = 0.9

    im = watermark.add_to_image(im)
    resize_opts = {"width": 2000, "height": 2000}
    im = resize(im, **resize_opts)
    write_opts = {
        "Q": quality,
        "no_subsample": no_subsample,
        "interlace": progressive,
        "strip": strip_exif,
    }
    LOG.info("Writing '%s' with options: %s", out_filename, write_opts)
    im.write_to_file(out_filename, **write_opts)


if __name__ == "__main__":
    main()
