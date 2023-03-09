# Copyright (c) 2022-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import json
import hashlib
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from optparse import OptionParser


"""
Takes in the data from a specific label id and maps it to the proper color for the bounding box
"""


def data_to_colour(data):
    if isinstance(data, str):
        data = bytes(data, "utf-8")
    else:
        data = bytes(data)
    m = hashlib.sha256()
    m.update(data)
    key = int(m.hexdigest()[:8], 16)
    r = ((((key >> 0) & 0xFF) + 1) * 33) % 255
    g = ((((key >> 8) & 0xFF) + 1) * 33) % 255
    b = ((((key >> 16) & 0xFF) + 1) * 33) % 255

    # illumination normalization to 128
    inv_norm_i = 128 * (3.0 / (r + g + b))

    return (int(r * inv_norm_i) / 255, int(g * inv_norm_i) / 255, int(b * inv_norm_i) / 255)


"""
Takes in the path to the rgb image for the background, then it takes bounding box data, the labels and the place to store the visualization. It outputs a colorized bounding box.
"""


def colorize_bbox_2d(rgb_path, data, id_to_labels, file_path):

    rgb_img = Image.open(rgb_path)
    colors = [data_to_colour(bbox["semanticId"]) for bbox in data]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb_img)
    for bbox_2d, color, index in zip(data, colors, range(len(data))):
        labels = id_to_labels[str(index)]
        rect = patches.Rectangle(
            xy=(bbox_2d["x_min"], bbox_2d["y_min"]),
            width=bbox_2d["x_max"] - bbox_2d["x_min"],
            height=bbox_2d["y_max"] - bbox_2d["y_min"],
            edgecolor=color,
            linewidth=2,
            label=labels,
            fill=False,
        )
        ax.add_patch(rect)

    plt.legend(loc="upper left")

    plt.savefig(file_path)


"""
Parses command line options. Requires input directory, output directory, and number for image to use.
"""


def parse_input():
    usage = "usage: visualize.py [options] arg1 arg2 arg3"
    parser = OptionParser(usage)
    parser.add_option("-d", "--data_dir", dest="data_dir", help="Directory location for Omniverse synthetic data")
    parser.add_option("-o", "--out_dir", dest="out_dir", help="Directory location for output image")
    parser.add_option("-n", "--number", dest="number", help="Number of image to use for visualization")
    (options, args) = parser.parse_args()
    return options, args


def main():
    options, args = parse_input()
    out_dir = options.data_dir
    rgb = "png/rgb_" + options.number + ".png"
    rgb_path = os.path.join(out_dir, rgb)
    bbox2d_tight_file_name = "npy/bounding_box_2d_tight_" + options.number + ".npy"
    data = np.load(os.path.join(options.data_dir, bbox2d_tight_file_name))

    # Check for labels
    bbox2d_tight_labels_file_name = "json/bounding_box_2d_tight_labels_" + options.number + ".json"
    with open(os.path.join(options.data_dir, bbox2d_tight_labels_file_name), "r") as json_data:
        bbox2d_tight_id_to_labels = json.load(json_data)

    # colorize and save image
    colorize_bbox_2d(rgb_path, data, bbox2d_tight_id_to_labels, os.path.join(options.out_dir, "bbox2d_tight.png"))


if __name__ == "__main__":
    main()
