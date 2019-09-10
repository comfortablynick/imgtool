#!/usr/bin/env python3
"""Compress/watermark images using vips."""

import pyvips
import logging

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

in_filename = "./test/chin_class.jpg"
out_filename = "./test/chin_class_edited.jpg"
watermark_opacity = 1
watermark_text = "© 2019 Nick Murphy | murphpix.com"
font = "sans 12"
replicate = False
rotate_degrees = 0
x_margin = 25
y_margin = 25

im = pyvips.Image.new_from_file(in_filename, access=pyvips.Access.SEQUENTIAL)

text = pyvips.Image.text(watermark_text, width=2000, dpi=300, font=f"{font}")
text = text.rotate(rotate_degrees)
text = (text * watermark_opacity).cast("uchar")
text = text.embed(x_margin, (im.height - text.height) - y_margin, im.width, im.height)

if replicate:
    text = text.replicate(1 + im.width / text.width, 1 + im.height / text.height)
    text = text.crop(0, 0, im.width, im.height)

# TODO: convert RGB to CMYK/mono on the fly
colors = {"cmyk": [0, 0, 0, 0], "rgb": [255, 255, 255], "mono": 255}

# we want to blend into the visible part of the image and leave any alpha
# channels untouched ... we need to split im into two parts
# guess how many bands from the start of im contain visible colour information
if im.bands >= 4 and im.interpretation == pyvips.Interpretation.CMYK:
    # cmyk image
    n_visible_bands = 4
    text_color = colors["cmyk"]
elif im.bands < 4:
    # rgb image
    n_visible_bands = 3
    text_color = colors["rgb"]
else:
    # mono image
    n_visible_bands = 1
    text_color = colors["mono"]

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

im.write_to_file(out_filename)
