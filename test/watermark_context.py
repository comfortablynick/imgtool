#!/usr/bin/env python3

"""Proof of concept for watermarking an image in a context-sensitive way.

Write black if the area we are overlaying the text on is white:
usage: ./watermark_context.py ~/pics/k2.jpg x.jpg 'hello there!'
"""
import sys
import pyvips


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


im = pyvips.Image.new_from_file(sys.argv[1])

text = pyvips.Image.text(
    sys.argv[3], width=im.width, dpi=300, align="centre", font="sans 16"
)
#  text = text.rotate(45)

# the position of the overlay in the image
#  x_pos = im.width - text.width - 25
#  top = 25
#  y_pos = im.height - text.height - 25
margin = 25
position = sys.argv[4]

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
overlay = overlay.bandjoin((text * 0.5).cast("uchar"))

# and composite that on to the original image
im = im.composite(overlay, "over", x=x_pos, y=y_pos)

im.write_to_file(sys.argv[2])
