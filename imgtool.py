#!/usr/bin/env python3
"""Compress/watermark images using vips."""

import pyvips
import logging
from dataclasses import dataclass
from typing import NamedTuple, Any
import argparse
import sys
from enum import Enum

logging.basicConfig()
LOG = logging.getLogger(__name__)


def parse_args(args: list):
    """Parse command line arguments."""
    # flags
    desc = (
        "A command-line utility which uses the vips library to manipulate "
        "images for web vewing. Images can be resampled, resized, and "
        "compressed at custom quality levels. Watermarking can also be added."
    )
    parser = argparse.ArgumentParser(prog="imgtool", description=desc, add_help=False)

    # Positionals
    parser.add_argument("input", help="image file to process", metavar="INPUT")
    parser.add_argument("output", help="file to save processed image", metavar="OUTPUT")

    # Flags
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
        default="_edited",
    )
    parser.add_argument(
        "-n",
        help="display results only; don't save file",
        dest="no_op",
        action="store_true",
    )

    # Image group
    image_group = parser.add_argument_group("General image options")
    image_group.add_argument(
        "-ke", help="keep exif data", dest="keep_exif", action="store_true"
    )
    image_group.add_argument(
        "-mw",
        help="maximum width of output",
        dest="width",
        metavar="WIDTH",
        type=int,
        default=0,
    )
    image_group.add_argument(
        "-mh",
        help="maximum height of output",
        dest="height",
        metavar="HEIGHT",
        default=0,
        type=int,
    )
    image_group.add_argument(
        "-wt",
        help="text to display in watermark",
        type=str,
        dest="watermark_text",
        metavar="TEXT",
    )
    image_group.add_argument(
        "-wr",
        help="angle of watermark rotation",
        dest="watermark_rotation",
        metavar="ANGLE",
        type=int,
        default=0,
    )
    image_group.add_argument(
        "-wp",
        help="watermark position",
        dest="watermark_position",
        metavar="POS",
        default=Position.BOTTOM_RIGHT,
    )

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
    parsed = parser.parse_intermixed_args(args)

    # do rudimentary checks
    if not 0 <= parsed.jpg_quality <= 100:
        parser.exit(
            1, f"Quality (-q) must be within 0-100; found: {parsed.jpg_quality}\n"
        )
    return parsed


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


class Position(Enum):
    """Predefined locations of text watermark."""

    TOP_LEFT = "top-left"
    TOP_RIGHT = "top-right"
    BOTTOM_RIGHT = "bottom-right"
    BOTTOM_LEFT = "bottom-left"


def avgze(image):
    """Zero-excluding average of an image.

    Returns an array, with the average for each band
    """
    # since we use histograms
    if image.format != "uchar" and image.format != "ushort":
        raise Exception("uchar and ushort images only")

    # take the histogram, and set the count for 0 pixels to 0, removing them
    histze = image.hist_find().insert(pyvips.Image.black(1, 1), 0, 0)

    # number of non-zero pixels in each band
    nnz = [histze[i].avg() * histze.width * histze.height for i in range(histze.bands)]

    # multiply by the identity function and we get the sum of non-zero
    # pixels ... for 16-bit images, we need a larger identity
    # function
    totalze = histze * pyvips.Image.identity(ushort=histze.width > 256)

    # find average value in each band
    avgze = [
        totalze[i].avg() * histze.width * histze.height / nnz[i]
        for i in range(totalze.bands)
    ]

    return avgze


def oppose(value, mx):
    """Find an opposing color for watermark.

    we split the density range (0 .. mx) into three:
      - values in the bottom third move to the top
      - values in the top third move to the bottom
      - values in the middle also move to the bottom
    """
    if value < mx / 3:
        # bottom goes up
        return mx / 3 - value + 2 * mx / 3
    elif value < 2 * mx / 3:
        # middle goes down
        return 2 * mx / 3 - value
    else:
        # top goes down
        return mx - value


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
    position: Position = Position.BOTTOM_LEFT

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
        if im.hasalpha():
            alpha = im.extract_band(im.bands - 1)
            im = im.extract_band(0, n=im.bands - 1)
        else:
            alpha = None

        if im.bands == 4:
            # cmyk
            text_color: Any = list(rgb_to_cmyk(self.fg_color))
        elif im.bands == 3:
            # rgb
            text_color = list(self.fg_color)
        else:
            # mono
            text_color = rgb_to_grayscale(self.fg_color)
        LOG.info("Watermark fg_color: %s (original: %s)", text_color, self.fg_color)
        im = text.ifthenelse(text_color, im, blend=True)

        # reattach alpha
        if alpha:
            im = im.bandjoin(alpha)
        return im

    def add(self, im: pyvips.Image) -> pyvips.Image:
        """Use method which finds complementary color.

        Returns image
        """
        text = pyvips.Image.text(
            self.text, width=im.width, dpi=300, font=f"{self.font}"
        )
        text = text.rotate(self.rotate)

        # the position of the overlay in the image
        margin = 25
        position = self.position

        if position == "top-left":
            x_pos = margin
            y_pos = margin
        elif position == "top-right":
            x_pos = im.width - text.width - margin
            y_pos = margin
        elif position == "bottom-right":
            x_pos = im.width - text.width - margin
            y_pos = im.height - text.height - margin
        elif position == "bottom-left":
            x_pos = margin
            y_pos = im.height - text.height - margin
        else:
            print(f"Incorrect watermark position: {position}")
            sys.exit(1)

        # find the non-alpha image bands
        if im.hasalpha():
            no_alpha = im.extract_band(0, n=im.bands - 1)
        else:
            no_alpha = im

        # the pixels we will render the overlay on top of
        bg = no_alpha.crop(x_pos, y_pos, text.width, text.height)

        # mask the background with the text, so all non-text areas become zero, and find
        # the zero-excluding average
        avg = avgze(text.ifthenelse(bg, 0))

        # for each band, find the opposing value
        mx = 255 if im.format == "uchar" else 65535
        text_colour = [oppose(avg[i], mx) for i in range(len(avg))]

        # make an overlay ... we put solid colour into the image and set a faded version
        # of the text mask as the alpha
        overlay = bg.new_from_image(text_colour)
        overlay = overlay.bandjoin((text * self.opacity).cast("uchar"))

        # and composite that on to the original image
        im = im.composite(overlay, "over", x=x_pos, y=y_pos)
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


def main():
    """Entry point."""
    args = parse_args(sys.argv[1:])
    log_level = 0
    try:
        log_level = (0, 20, 10)[args.verbosity]
    except IndexError:
        log_level = 10
    LOG.setLevel(log_level)
    logging.getLogger("pyvips").setLevel(log_level)
    LOG.debug(args)
    progressive = True
    no_subsample = True if args.jpg_quality > 75 else False

    im = pyvips.Image.new_from_file(args.input)
    if args.watermark_text:
        watermark = TextWatermark(args.watermark_text)
        watermark.font = "Source Sans Pro 16"
        watermark.fg_color = RGB(255, 255, 255)
        watermark.rotate = args.watermark_rotation
        watermark.opacity = 0.9
        watermark.position = args.watermark_position
        im = watermark.add_to_image(im)

    if args.width or args.height:
        im = resize(im, width=args.width, height=args.height)

    if args.no_op:
        LOG.info("Displaying results only due to -n")
    else:
        write_opts = {
            "Q": args.jpg_quality,
            "no_subsample": no_subsample,
            "interlace": progressive,
            "strip": not args.keep_exif,
        }
        LOG.info("Writing '%s' with options: %s", args.input, write_opts)
        im.write_to_file(args.output, **write_opts)


if __name__ == "__main__":
    main()
