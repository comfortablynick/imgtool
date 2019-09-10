#!/usr/bin/env python3
"""Compress/watermark images using vips."""

import pyvips
import logging

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

in_filename = "./test/chin_class.jpg"
out_filename = "./test/chin_class_edited.jpg"

image = pyvips.Image.new_from_file(in_filename, access=pyvips.Access.SEQUENTIAL)
LOG.debug("Image: %s; Dims: %dx%d", in_filename, image.width, image.height)
#  LOG.debug(image.get_fields())
resized = image.thumbnail_image(2000, height=2000)
resized.write_to_file(out_filename, Q=85)
