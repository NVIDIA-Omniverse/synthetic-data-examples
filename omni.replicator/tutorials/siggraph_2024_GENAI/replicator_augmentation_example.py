# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
#     2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
#     3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from typing import Dict

import omni.replicator.core as rep
import omni.replicator.core.functional as F
import numpy as np
import warp as wp
from datetime import datetime

now = datetime.now()
time = now.strftime("%H-%M-%S")
output_directory = "_sdg_dataset_" + time


def random_colours(N, enable_random=True, num_channels=4):
    """
    Generate random colors.
    Generate visually distinct colours by linearly spacing the hue
    channel in HSV space and then convert to RGB space.
    """
    import random
    import colorsys

    start = 0
    if enable_random:
        random.seed(10)
        start = random.random()
    hues = [(start + i / N) % 1.0 for i in range(N)]
    colours = [list(colorsys.hsv_to_rgb(h, 0.9, 1.0)) for i, h in enumerate(hues)]
    if num_channels == 4:
        for color in colours:
            color.append(1.0)
    if enable_random:
        random.shuffle(colours)

    colours = (np.array(colours) * 255).astype(np.uint8)
    return colours


@wp.kernel
def remap_depth(
    data_in: wp.array2d(dtype=wp.float32),
    data_out: wp.array2d(dtype=wp.float32),
    z_far: float = 100.0,
):
    """Remap depth to range expected by controlNet

    Args:
        depth: Depth array representing distance from image plane in float32
    Returns:
        Depth array normalized to format expected by controlNet in
    """
    i, j = wp.tid()
    data_out[i, j] = 1.0 - (min(z_far, data_in[i, j]) / z_far)


def _remap_segmentation(segmentation_data: np.ndarray, mapping: Dict) -> np.ndarray:
    """Remap semantic IDs to predefined colours

    Args:
        segmentation_image: data returned by the annotator.
    Return:
        Data converted to uint8 RGBA image
    """
    segmentation = segmentation_data["data"]
    mapping = {
        int(k): mapping.get(v.get("class"), (0, 0, 0, 0))
        for k, v in segmentation_data["info"]["idToLabels"].items()
    }
    segmentation_ids = np.unique(segmentation)
    num_colours = len(segmentation_ids)

    # This is to avoid generating lots of colours for semantic classes not in frame
    lut = np.array([segmentation_ids, list(range(num_colours))])

    new_segmentation_image = lut[1, np.searchsorted(lut[0, :], segmentation)]

    colours = np.array([[0.0] * 4] * (num_colours + 1))
    for idx in range(lut.shape[1]):
        semantic_id, lut_idx = lut[:, idx]
        colours[lut_idx] = mapping.get(semantic_id)

    segmentation_image_rgba = np.array(colours[new_segmentation_image], dtype=np.uint8)
    return segmentation_image_rgba


class GenAIWriter(rep.writers.Writer):
    def __init__(self, output_dir):
        self.frame_id = 0
        self.class_mapping = {
            "floor": (255, 25, 171, 255),
            "forklift": (149, 255, 25, 255),
            "walking_worker": (25, 240, 255, 255),
            "warehouse_pallets": (255, 197, 25, 255),
            "warehouse_bin": (255, 249, 25, 255),
        }
        depth_custom = rep.annotators.get(
            "distance_to_image_plane", device="cuda"
        ).augment(remap_depth)
        self.annotators = [
            "LdrColor",
            "normals",
            depth_custom,
            "semantic_segmentation",
        ]
        self.backend = rep.backends.get(
            "DiskBackend", init_params={"output_dir": output_dir}
        )

    def write(self, data):
        # write colour
        self.backend.schedule(
            F.write_image, data=data["LdrColor"], path=f"ldr_color_{self.frame_id}.png"
        )

        # write normals
        colourize_normals = rep.backends.Sequential(
            lambda x: ((x * 0.5 + 0.5) * 255).astype(np.uint8)
        )
        self.backend.schedule(
            F.write_image,
            data=colourize_normals(data["normals"]),
            path=f"normals_{self.frame_id}.png",
        )

        # write depth
        depth_float_to_uint16 = rep.backends.Sequential(
            lambda x: x.numpy(),
            lambda x: x * 65535,
            lambda x: x.astype(np.uint16),
        )
        self.backend.schedule(
            F.write_image,
            data=depth_float_to_uint16(data["distance_to_image_plane"]),
            path=f"depth_{self.frame_id}.png",
        )

        # write segmentation
        remap_segmentation = rep.backends.Sequential(
            lambda x: _remap_segmentation(x, self.class_mapping)
        )
        self.backend.schedule(
            F.write_image,
            data=remap_segmentation(data["semantic_segmentation"]),
            path=f"semantic_{self.frame_id}.png",
        )

        self.frame_id += 1


# Create a camera in the scene, create render product and attach writer
camera = rep.create.camera()
rp = rep.create.render_product(camera, (1920, 1080))
writer = GenAIWriter(output_directory)
writer.attach(rp)

# Set camera azimuth angles
camera_azimuth = [-105, -95, -85, -75, -65, -55, -45, -35]

# The camera does a short arc around a central barycenter
# 8 Images are generated with RGB, Semantic Segmentation, Normals and Depth.
with rep.trigger.on_frame(max_execs=8, rt_subframes=20):
    with camera:
        rep.modify.pose_orbit(
            barycentre=(-7.5, 6.2, 2),
            distance=rep.distribution.uniform(12, 12),
            azimuth=rep.distribution.sequence(camera_azimuth),
            elevation=rep.distribution.uniform(20, 20),
        )
rep.orchestrator.run()
