# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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
import numpy as np
import omni.replicator.core as rep
import omni.replicator.core.functional as F
import omni.usd
import pxr
import json
import warp as wp
from video_encoding import get_video_encoding_interface

#### SETUP REQUIRED!
# You need to enable the "Video Encoding Extension"
# Search for "omni.videoencoding" in the Extensions tab of your KIT app, and enable it.

OUTPUT_DIR = "C:/YOURFOLDER/shaded_segmentation.mp4"
start_frame = 0
end_frame = 120

MAPPING = {
    "class:background": (255, 36, 66, 255),
    "class:palette": (255, 237, 218, 255),
    "class:floor": (100, 100, 100, 255),
    "class:forklift": (61, 178, 255, 255),
}


class CosmosWriter(rep.Writer):
    """Writer class for generating shaded semantic segmentation videos.

    Args:
        video_filepath: Path where the output video will be saved (e.g. "/home/user/Videos/my_video.mp4").
    """

    def __init__(self, video_filepath: str, num_frames: int = 0):
        self._video_filepath = OUTPUT_DIR
        self.annotators = [
            rep.annotators.get(
                "semantic_segmentation", init_params={"colorize": True, "mapping": json.dumps(MAPPING)}, device="cuda"
            ),
            rep.annotators.get("normals", device="cuda"),
        ]
        self._frame_id = 0
        self._light_source = None

        self._video_encoding = get_video_encoding_interface()
        self._video_encoding.start_encoding(
            video_filename=self._video_filepath,
            framerate=40.0,
            nframes=num_frames,  # 0 for unlimited
            overwrite_video=True,
        )

    @staticmethod
    @wp.kernel
    def shade_segmentation(
        segmentation: wp.array3d(dtype=wp.uint8),
        normals: wp.array3d(dtype=wp.float32),
        shading_out: wp.array3d(dtype=wp.uint8),
        light_source: wp.array(dtype=wp.vec3f),
    ):
        """Apply shading to semantic segmentation using surface normals.

        Args:
            segmentation: Input semantic segmentation image (H,W,C)
            normals: Surface normal vectors (H,W,3)
            shading_out: Output shaded segmentation image (H,W,C)
            light_source: Position of light source
        """
        i, j = wp.tid()
        normal = normals[i, j]
        normals_normalized = wp.vec3f(normal[0], normal[1], normal[2]) * 0.5 + wp.vec3f(0.5, 0.5, 0.5)
        light_source_vec = wp.normalize(light_source[0])
        shade = 0.5 + wp.dot(normals_normalized, light_source_vec) * 0.5
        shading_out[i, j, 0] = wp.uint8(wp.float32(segmentation[i, j, 0]) * shade)
        shading_out[i, j, 1] = wp.uint8(wp.float32(segmentation[i, j, 1]) * shade)
        shading_out[i, j, 2] = wp.uint8(wp.float32(segmentation[i, j, 2]) * shade)

    def write(self, data):
        """Write a frame of shaded semantic segmentation to the video.

        Args:
            data: Dictionary containing frame data with keys:
                - normals: Surface normal vectors array (H,W,3)
                - semantic_segmentation: Dictionary containing:
                    - data: Semantic segmentation image array (H,W,C)

        Note:
            The method applies shading to the semantic segmentation using surface normals
            and encodes the resulting frame to video.
        """
        normals = data["normals"][:, :, :3]
        segmentation = data["semantic_segmentation"]["data"]
        if self._light_source is None:
            self._light_source = wp.array([0.0, 0.0, 1.0], dtype=wp.vec3f, device=normals.device)
        shaded_seg = wp.empty_like(segmentation)
        wp.launch(
            self.shade_segmentation,
            dim=normals.shape[:2],
            inputs=[segmentation, normals, shaded_seg, self._light_source],
        )

        height, width = shaded_seg.shape[:2]
        self._video_encoding.encode_next_frame_from_buffer(shaded_seg.numpy().tobytes(), width=width, height=height)
        self._frame_id += 1

    def on_final_frame(self):
        if self._frame_id > 0:
            self._video_encoding.finalize_encoding()


async def capture(render_product):
    """Capture frames from the scene.

    Args:
        render_product: List of render products to capture from
        idx: Demo index
        dataset: Dataset index
    """
    print("Capturing...")
    timeline = omni.timeline.get_timeline_interface()
    timeline.set_end_time(1000)
    timeline.set_time_codes_per_second(60)
    timeline.commit()

    rep.utils.send_og_event("foo")
    output_dir = f"{OUTPUT_DIR}"

    cosmos_writer = CosmosWriter(output_dir, num_frames=200)
    cosmos_writer.attach(render_product[0])

    timeline.set_current_time(start_frame / timeline.get_time_codes_per_seconds())
    timeline.commit()

    await rep.orchestrator.run_async(num_frames=end_frame, start_timeline=True)
    timeline.stop()
    await rep.run_until_complete_async()
    cosmos_writer.detach()


async def main():
    """Main function to run the stacking demo scene generation."""
    rep.settings.set_stage_up_axis("Z")
    rep.settings.set_stage_meters_per_unit(1.0)

    # Make sure to assign which camera in your seen is assigned to the render product
    camera = "/World/PickUpCam"
    render_product = rep.create.render_product(camera, resolution=(1280, 704))

    await capture([render_product])

    print("DONE!")


import asyncio

asyncio.ensure_future(main())
