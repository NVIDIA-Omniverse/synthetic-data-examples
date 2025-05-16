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

from semantics.schema.editor import add_prim_semantics, LabelWriteType
import omni
from pxr import Gf, UsdGeom
import asyncio
import omni.replicator.core as rep
import numpy as np
import random
import json
from typing import Optional
import omni.timeline
import warp as wp
from video_encoding import get_video_encoding_interface
from omni.replicator.core import functional as F
from omni.replicator.core.annotators import AnnotatorRegistry
from omni.replicator.core.backends import io_queue
from omni.replicator.core.writers import Writer

#### SETUP REQUIRED!
# 1. You need to enable the "Video Encoding Extension"
# 2. Search for "omni.videoencoding" in the Extensions tab of your KIT app, and enable it.
# 3. Set the output_path below to a local directory
start_frame = 0
end_frame = 121
output_path = "C:/your_output_folder"  # See setup item 1 & 2 as well

__version__ = "0.0.1"


@wp.kernel
def rgb_to_grey_and_blur(
    data_in: wp.array3d(dtype=wp.uint8), data_out: wp.array2d(dtype=wp.uint8)
):
    i, j = wp.tid()
    height = data_in.shape[0]
    width = data_in.shape[1]

    # Only process non-border pixels for the blur
    if i > 0 and i < height - 1 and j > 0 and j < width - 1:
        # Gaussian kernel 3x3
        kernel = wp.mat33f(1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0) / 16.0

        sum = 0.0
        # Apply convolution with RGB to grayscale conversion in a single pass
        for ki in range(-1, 2):
            for kj in range(-1, 2):
                # Convert RGB to grayscale using standard weights
                gray_val = (
                    wp.float(data_in[i + ki, j + kj, 0]) * 0.299  # Red
                    + wp.float(data_in[i + ki, j + kj, 1]) * 0.587  # Green
                    + wp.float(data_in[i + ki, j + kj, 2]) * 0.114  # Blue
                )
                sum += gray_val * kernel[ki + 1, kj + 1]

        data_out[i, j] = wp.uint8(wp.clamp(sum, 0.0, 255.0))
    else:
        # For border pixels, just convert to grayscale without blur
        data_out[i, j] = wp.uint8(
            wp.float(data_in[i, j, 0]) * 0.299
            + wp.float(data_in[i, j, 1]) * 0.587
            + wp.float(data_in[i, j, 2]) * 0.114
        )


@wp.kernel
def sobel_and_suppress(
    data_in: wp.array2d(dtype=wp.uint8),
    data_out: wp.array3d(dtype=wp.uint8),
    low_threshold: float,
    high_threshold: float,
):
    i, j = wp.tid()
    height = data_in.shape[0]
    width = data_in.shape[1]

    if i > 0 and i < height - 2 and j > 0 and j < width - 2:
        # Compute Sobel gradients with proper kernel weights
        gx = (
            -1.0 * wp.float(data_in[i - 1, j - 1])
            + -2.0 * wp.float(data_in[i, j - 1])
            + -1.0 * wp.float(data_in[i + 1, j - 1])
            + 1.0 * wp.float(data_in[i - 1, j + 1])
            + 2.0 * wp.float(data_in[i, j + 1])
            + 1.0 * wp.float(data_in[i + 1, j + 1])
        )

        gy = (
            -1.0 * wp.float(data_in[i - 1, j - 1])
            + -2.0 * wp.float(data_in[i - 1, j])
            + -1.0 * wp.float(data_in[i - 1, j + 1])
            + 1.0 * wp.float(data_in[i + 1, j - 1])
            + 2.0 * wp.float(data_in[i + 1, j])
            + 1.0 * wp.float(data_in[i + 1, j + 1])
        )

        # Compute gradient magnitude
        magnitude = wp.sqrt(gx * gx + gy * gy)

        # Calculate gradient direction and handle edge cases
        angle = wp.atan2(gy, gx) * 180.0 / 3.14159
        if angle < 0:
            angle += 180.0

        # Get interpolated magnitudes along gradient direction
        g00 = wp.float(0.0)  # First interpolated point
        g01 = wp.float(0.0)  # Second interpolated point
        xstep = wp.float(0.0)
        ystep = wp.float(0.0)

        # Determine interpolation direction based on angle
        if angle <= 22.5 or angle > 157.5:  # ~0 degrees
            xstep = wp.float(1.0)
            ystep = wp.float(0.0)
        elif angle > 22.5 and angle <= 67.5:  # ~45 degrees
            xstep = wp.float(1.0)
            ystep = wp.float(1.0)
        elif angle > 67.5 and angle <= 112.5:  # ~90 degrees
            xstep = wp.float(0.0)
            ystep = wp.float(1.0)
        else:  # ~135 degrees
            xstep = wp.float(-1.0)
            ystep = wp.float(1.0)

        # Interpolate gradient magnitudes
        # Forward direction
        x1 = wp.float(j) + xstep
        y1 = wp.float(i) + ystep
        if x1 >= 0 and x1 < width and y1 >= 0 and y1 < height:
            x1_floor = wp.int32(x1)
            y1_floor = wp.int32(y1)
            if (
                x1_floor >= 0
                and x1_floor < width - 1
                and y1_floor >= 0
                and y1_floor < height - 1
            ):
                gx1 = (
                    -1.0 * wp.float(data_in[y1_floor - 1, x1_floor - 1])
                    + -2.0 * wp.float(data_in[y1_floor, x1_floor - 1])
                    + -1.0 * wp.float(data_in[y1_floor + 1, x1_floor - 1])
                    + 1.0 * wp.float(data_in[y1_floor - 1, x1_floor + 1])
                    + 2.0 * wp.float(data_in[y1_floor, x1_floor + 1])
                    + 1.0 * wp.float(data_in[y1_floor + 1, x1_floor + 1])
                )
                gy1 = (
                    -1.0 * wp.float(data_in[y1_floor - 1, x1_floor - 1])
                    + -2.0 * wp.float(data_in[y1_floor - 1, x1_floor])
                    + -1.0 * wp.float(data_in[y1_floor - 1, x1_floor + 1])
                    + 1.0 * wp.float(data_in[y1_floor + 1, x1_floor - 1])
                    + 2.0 * wp.float(data_in[y1_floor + 1, x1_floor])
                    + 1.0 * wp.float(data_in[y1_floor + 1, x1_floor + 1])
                )
                g00 = wp.sqrt(gx1 * gx1 + gy1 * gy1)

        # Backward direction
        x2 = wp.float(j) - xstep
        y2 = wp.float(i) - ystep
        if x2 >= 0 and x2 < width and y2 >= 0 and y2 < height:
            x2_floor = wp.int32(x2)
            y2_floor = wp.int32(y2)
            if (
                x2_floor >= 0
                and x2_floor < width - 1
                and y2_floor >= 0
                and y2_floor < height - 1
            ):
                gx2 = (
                    -1.0 * wp.float(data_in[y2_floor - 1, x2_floor - 1])
                    + -2.0 * wp.float(data_in[y2_floor, x2_floor - 1])
                    + -1.0 * wp.float(data_in[y2_floor + 1, x2_floor - 1])
                    + 1.0 * wp.float(data_in[y2_floor - 1, x2_floor + 1])
                    + 2.0 * wp.float(data_in[y2_floor, x2_floor + 1])
                    + 1.0 * wp.float(data_in[y2_floor + 1, x2_floor + 1])
                )
                gy2 = (
                    -1.0 * wp.float(data_in[y2_floor - 1, x2_floor - 1])
                    + -2.0 * wp.float(data_in[y2_floor - 1, x2_floor])
                    + -1.0 * wp.float(data_in[y2_floor - 1, x2_floor + 1])
                    + 1.0 * wp.float(data_in[y2_floor + 1, x2_floor - 1])
                    + 2.0 * wp.float(data_in[y2_floor + 1, x2_floor])
                    + 1.0 * wp.float(data_in[y2_floor + 1, x2_floor + 1])
                )
                g01 = wp.sqrt(gx2 * gx2 + gy2 * gy2)

        # Strict non-maximum suppression with interpolation
        if magnitude > g00 and magnitude > g01:
            # Scale magnitude to match OpenCV's scaling
            scaled_magnitude = magnitude * 3.0

            if scaled_magnitude >= high_threshold:
                data_out[i, j, 0] = wp.uint8(255)  # Strong edge
            elif scaled_magnitude >= low_threshold:
                data_out[i, j, 0] = wp.uint8(127)  # Weak edge
            else:
                data_out[i, j, 0] = wp.uint8(0)  # Non-edge
        else:
            data_out[i, j, 0] = wp.uint8(0)
    else:
        data_out[i, j, 0] = wp.uint8(0)


@wp.kernel
def hysteresis_thresholding(data_inout: wp.array3d(dtype=wp.uint8)):
    i, j = wp.tid()
    height = data_inout.shape[0]
    width = data_inout.shape[1]

    if i > 0 and i < height - 1 and j > 0 and j < width - 1:
        # Only process weak edges
        if data_inout[i, j, 0] == 127:
            # Check 8-connected neighbors
            has_strong_neighbor = float(0.0)

            for di in range(-1, 2):
                if i + di < 0 or i + di >= height:
                    continue
                for dj in range(-1, 2):
                    if j + dj < 0 or j + dj >= width:
                        continue
                    if di == 0 and dj == 0:
                        continue
                    if data_inout[i + di, j + dj, 0] == 255:
                        has_strong_neighbor += 1.0
                        break
                if has_strong_neighbor >= 1.0:
                    break

            # Convert weak edge to strong if connected to strong edge, otherwise suppress
            if has_strong_neighbor >= 1.0:
                data_inout[i, j, 0] = wp.uint8(255)
            else:
                data_inout[i, j, 0] = wp.uint8(0)

    # Populate B and G channels for video encoder
    data_inout[i, j, 1] = data_inout[i, j, 0]
    data_inout[i, j, 2] = data_inout[i, j, 0]


@wp.kernel
def rand_colours(
    data_in: wp.array2d(dtype=wp.uint32), data_out: wp.array3d(dtype=wp.uint8)
):
    i, j = wp.tid()
    instance_id = data_in[i, j]

    # Check if this is background (index 0)
    if instance_id == 0:
        # Set background to black
        data_out[i, j, 0] = wp.uint8(0)
        data_out[i, j, 1] = wp.uint8(0)
        data_out[i, j, 2] = wp.uint8(0)

    # Convert instance_id to color using HSV with pastel parameters
    state_h = wp.rand_init(wp.int32(instance_id))
    state_s = wp.rand_init(wp.int32(instance_id), 1)
    state_v = wp.rand_init(wp.int32(instance_id), 2)

    # Pastel colors have moderate saturation and high value/brightness
    h = wp.randf(state_h)
    s = 0.3 + wp.randf(state_s) * 0.3  # Lower saturation (0.3-0.6) for pastel effect
    v = 0.9 + wp.randf(state_v) * 0.1  # High value/brightness (0.9-1.0)

    # HSV to RGB conversion
    K = wp.vec4f(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0)
    p = wp.vec3f(
        wp.abs(wp.frac(h + K.x) * 6.0 - K.w),
        wp.abs(wp.frac(h + K.y) * 6.0 - K.w),
        wp.abs(wp.frac(h + K.z) * 6.0 - K.w),
    )
    clamped = wp.vec3f(
        wp.clamp(p[0] - K.x, 0.0, 1.0),
        wp.clamp(p[1] - K.x, 0.0, 1.0),
        wp.clamp(p[2] - K.x, 0.0, 1.0),
    )
    rgb = v * wp.lerp(wp.vec3f(K.x), clamped, s)

    data_out[i, j, 0] = wp.uint8(rgb[0] * 255.0)
    data_out[i, j, 1] = wp.uint8(rgb[1] * 255.0)
    data_out[i, j, 2] = wp.uint8(rgb[2] * 255.0)


@wp.kernel
def shade_segmentation(
    segmentation: wp.array3d(dtype=wp.uint8),
    normals: wp.array3d(dtype=wp.float32),
    shading_out: wp.array3d(dtype=wp.uint8),
    light_source: wp.array(dtype=wp.vec3f),
):
    """Apply pastel-like colorization to semantic segmentation and shading using surface normals.

    Args:
        segmentation: Input semantic segmentation image with instance IDs (H,W)
        normals: Surface normal vectors (H,W,3)
        shading_out: Output colorized segmentation image (H,W,3)
        light_source: Position of light source
    """
    i, j = wp.tid()

    normal = normals[i, j]
    normals_normalized = wp.normalize(wp.vec3f(normal[0], normal[1], normal[2]))
    light_source_vec = wp.normalize(light_source[0])

    # Calculate base shading from dot product (ranges from -1 to 1)
    base_shade = wp.dot(normals_normalized, light_source_vec)
    # Remap from [-1, 1] to desired shading range [min_shade, 1.0]
    min_shade = 0.5  # Adjusted for pastel effect
    shade = wp.clamp(wp.lerp(min_shade, 1.0, (base_shade + 1.0) * 0.5), 0.0, 1.0)

    # Apply shading and convert to uint8
    shading_out[i, j, 0] = wp.uint8(wp.float(segmentation[i, j, 0]) * shade)
    shading_out[i, j, 1] = wp.uint8(wp.float(segmentation[i, j, 1]) * shade)
    shading_out[i, j, 2] = wp.uint8(wp.float(segmentation[i, j, 2]) * shade)


@wp.kernel
def colorize_depth(
    data_in: wp.array2d(dtype=wp.float32),
    data_out: wp.array3d(dtype=wp.uint8),
    near: float,
    far: float,
):
    """Apply colorization to depth data.

    Args:
        distance_data: Input depth data (H,W)
        output: Output colorized depth image (H,W,3)
        near: Minimum depth value to consider
        far: Maximum depth value to consider
    """
    i, j = wp.tid()

    # Get the depth value
    depth = data_in[i, j]

    # Skip invalid values
    if depth != wp.inf and depth != -wp.inf:
        # Clip depth to range [near, far]
        clipped_depth = wp.clamp(depth, near, far) + 1e-5

        # Apply log normalization
        normalized = (wp.log(clipped_depth) - wp.log(near)) / (
            wp.log(far) - wp.log(near)
        )

        # Invert and scale to 0-255 range
        color_value = wp.uint8((1.0 - normalized) * 255.0)

        # Set RGB channels to the same value (grayscale)
        data_out[i, j, 0] = color_value
        data_out[i, j, 1] = color_value
        data_out[i, j, 2] = color_value
    else:
        # For invalid depth values, set to black
        data_out[i, j, 0] = wp.uint8(0)
        data_out[i, j, 1] = wp.uint8(0)
        data_out[i, j, 2] = wp.uint8(0)
    # Set alpha channel to 0
    data_out[i, j, 3] = wp.uint8(255)


class CosmosWriter(Writer):
    """Writer class for generating input to Cosmos Transfer1.

    This writer generates videos for various modalities that can be used as input to Cosmos Transfer1. The included
    modalities are:
    - RGB
    - Shaded Instance Segmentation
    - Instance Segmentation
    - Distance to camera (Depth)
    - Edges

    If using a ``trigger.on_time`` node, the writer will automatically increment the clip index when the trigger fires.
    Otherwise, the clip index can be incremented manually by calling the ``next_clip`` method.

    Args:
        backend: The backend to use for writing the video.
        video_filepath: Path where the output video will be saved (e.g. "/home/user/Videos/my_video.mp4").
        segmentation_mapping: An optional dictionary mapping semantic labels to specific colors.
        use_instance_id: Whether to use instance id segmentation instead of instance segmentation. Instance ID
            segmentation does not require assets be semantically annotated.
        canny_threshold_low: The lower threshold for the Canny edge detector.
        canny_threshold_high: The higher threshold for the Canny edge detector.
    """

    def __init__(
        self,
        backend,
        segmentation_mapping: Optional[dict] = None,
        use_instance_id: bool = False,
        canny_threshold_low: int = 10,
        canny_threshold_high: int = 100,
    ):
        self._backend = backend
        self.version = __version__

        semantic_params = {"colorize": False}
        if segmentation_mapping:
            semantic_params["mapping"] = json.dumps(segmentation_mapping)

        segmentation_annotator = (
            "instance_id_segmentation_fast"
            if use_instance_id
            else "instance_segmentation_fast"
        )
        self.annotators = [
            AnnotatorRegistry.get_annotator(
                segmentation_annotator, init_params=semantic_params, device="cuda"
            ),
            AnnotatorRegistry.get_annotator("normals", device="cuda"),
            AnnotatorRegistry.get_annotator("distance_to_camera").augment(
                colorize_depth,
                name="depth",
                near=0.1,
                far=100,
                data_out_shape=(-1, -1, 4),
            ),
            AnnotatorRegistry.get_annotator("rgb", device="cuda"),
        ]
        self._canny_threshold_low = canny_threshold_low
        self._canny_threshold_high = canny_threshold_high
        self._frame_id = 0
        self._clip_idx = 0
        self._frame_rate = None
        self._light_source = None
        self._cached_buffers = {}

    def _get_shaded_segmentation(self, normals, segmentation):
        """Get shaded segmentation from the segmentation and normals.

        Args:
            normals: Normals array (H,W,3)
            segmentation: Segmentation array (H,W)
        """
        height, width = segmentation.shape[:2]
        device = segmentation.device
        if self._cached_buffers.get("light_source") is None:
            self._cached_buffers["light_source"] = wp.array(
                [0.0, 0.0, 1.0], dtype=wp.vec3f, device=device
            )
        light_source = self._cached_buffers["light_source"]
        shaded_segmentation_out = wp.empty(
            dtype=wp.uint8,
            shape=(height, width, 3),
            device=device,
            owner=False,
            requires_grad=False,
        )
        wp.launch(
            kernel=shade_segmentation,
            dim=(height, width),
            inputs=[segmentation, normals, shaded_segmentation_out, light_source],
            device=device,
        )
        return shaded_segmentation_out

    def _get_segmentation(self, instance_segmentation):
        """Get segmentation from the instance segmentation.

        Args:
            instance_segmentation: Instance segmentation array (H,W,C)
        """
        height, width = instance_segmentation.shape[:2]
        device = instance_segmentation.device
        segmentation_out = wp.empty(
            dtype=wp.uint8,
            shape=(height, width, 3),
            device=device,
        )
        wp.launch(
            kernel=rand_colours,
            dim=(height, width),
            inputs=[instance_segmentation, segmentation_out],
            device=device,
        )
        return segmentation_out

    def _get_canny_edges(self, shaded_segmentation):
        """Get canny edges from the shaded segmentation.

        Args:
            shaded_segmentation: Shaded segmentation array (H,W,3)
        """
        height, width = shaded_segmentation.shape[:2]
        device = shaded_segmentation.device

        # Allocate output buffer
        # Encoder expects 3 channels
        canny_edges_out = wp.empty(
            dtype=wp.uint8,
            shape=(height, width, 3),
            device=device,
        )

        # Cache this buffer to avoid reallocating it on each call
        if self._cached_buffers.get("greyscale") is None:
            self._cached_buffers["greyscale"] = wp.empty(
                dtype=wp.uint8,
                shape=(height, width),
                device=device,
                owner=False,
                requires_grad=False,
            )

        # Step 1: Convert to greyscale and blur
        wp.launch(
            kernel=rgb_to_grey_and_blur,
            dim=(height, width),
            inputs=[shaded_segmentation, self._cached_buffers["greyscale"]],
            device=device,
        )

        # Step 2: Apply sobel operator
        wp.launch(
            kernel=sobel_and_suppress,
            dim=(height, width),
            inputs=[
                self._cached_buffers["greyscale"],
                canny_edges_out,
                self._canny_threshold_low,
                self._canny_threshold_high,
            ],
            device=device,
        )

        # Step 3: Apply hysteresis thresholding
        wp.launch(
            kernel=hysteresis_thresholding,
            dim=(height, width),
            inputs=[canny_edges_out],
            device=device,
        )

        return canny_edges_out

    def write(self, data):
        """Write video data optimized for Cosmos Transfer1.

        Args:
            data: Dictionary containing frame data with keys:
                - rgb: RGB image array (H,W,3)
                - instance_segmentation_fast: Instance segmentation image array (H,W,C)
                - shaded_instance_segmentation: Shaded instance segmentation image array (H,W,C)
                - depth: Depth image array (H,W,1)
                - edges: Edge image array (H,W,1)
        """
        sequence_id = 0
        for trigger_name, call_count in data["trigger_outputs"].items():
            if "on_time" in trigger_name:
                sequence_id = call_count
        if sequence_id != self._clip_idx:
            self.next_clip()

        if self._frame_rate is None:
            timeline_iface = omni.timeline.get_timeline_interface()
            self._frame_rate = timeline_iface.get_time_codes_per_seconds()
        instance_segmentation = data["instance_segmentation_fast"]["data"]
        segmentation = self._get_segmentation(instance_segmentation)
        normals = data["normals"]
        depth = data["depth"]
        rgb = data["rgb"]
        shaded_seg = self._get_shaded_segmentation(normals, segmentation)
        edges = self._get_canny_edges(shaded_seg)

        self._backend.schedule(
            F.write_image,
            data=rgb,
            path=f"clip_{self._clip_idx:04}/rgb/rgb_{self._frame_id:04}.png",
        )
        self._backend.schedule(
            F.write_image,
            data=shaded_seg,
            path=f"clip_{self._clip_idx:04}/shaded_seg/shaded_seg_{self._frame_id:04}.png",
        )
        self._backend.schedule(
            F.write_image,
            data=segmentation,
            path=f"clip_{self._clip_idx:04}/instance_segmentation/instance_segmentation_{self._frame_id:04}.png",
        )
        self._backend.schedule(
            F.write_image,
            data=depth,
            path=f"clip_{self._clip_idx:04}/depth/depth_{self._frame_id:04}.png",
        )
        self._backend.schedule(
            F.write_image,
            data=edges,
            path=f"clip_{self._clip_idx:04}/edges/edges_{self._frame_id:04}.png",
        )
        self._frame_id += 1

    def on_final_frame(self):
        if self._frame_id == 0:
            return

        io_queue.wait_until_done()

        for key in ["rgb", "instance_segmentation", "edges", "depth", "shaded_seg"]:
            video_encoding = get_video_encoding_interface()
            video_encoding.start_encoding(
                video_filename=f"{self._backend.output_dir}/clip_{self._clip_idx:04}/{key}.mp4",
                framerate=self._frame_rate,
                nframes=self._frame_id,
                overwrite_video=True,
            )
            for i in range(self._frame_id):
                path = f"{self._backend.output_dir}/clip_{self._clip_idx:04}/{key}/{key}_{i:04}.png"
                video_encoding.encode_next_frame_from_file(path)
            video_encoding.finalize_encoding()

        self._frame_id = 0

    def next_clip(self):
        """Finalize current clip and update parameters for the next one.

        - Combines generated frames into videos
        - Resets frame counter
        - Increments output directory
        """
        self.on_final_frame()
        self._clip_idx += 1


def set_posrot(mesh_prim, prim_pos, prim_rot):
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xformable(mesh_prim)
    xform_ops = {op.GetBaseName(): op for op in xform.GetOrderedXformOps()}

    for op in xform.GetOrderedXformOps():
        if "translate" in op.GetName():
            translate_op = xform_ops["translate"]
            translate_op.Set(Gf.Vec3d(prim_pos[0], prim_pos[1], prim_pos[2]))
        elif "rotate" in op.GetName():
            rotate_op = xform_ops["rotateXYZ"]
            rotate_op.Set(Gf.Vec3d(prim_rot[0], prim_rot[1], prim_rot[2]))


def interpolate(v0, t0, v1, t1, t):
    u = (t - t0) / (t1 - t0)
    u = np.clip(u, 0.0, 1.0)

    v0_arr = np.array(v0, dtype=float)
    v1_arr = np.array(v1, dtype=float)
    return (1 - u) * v0_arr + u * v1_arr


async def run():

    # Get the stage
    stage = omni.usd.get_context().get_stage()

    # Set timeline
    print("Capturing...")
    timeline = omni.timeline.get_timeline_interface()
    timeline.set_end_time(end_frame)
    timeline.set_time_codes_per_second(60)
    timeline.commit()

    # Forklift Animation Transform settings
    forklift_start = (0.63876, 5.33176, 0)
    forklift_end = (-1.98992, 4.52809, 0)
    forklift_rot = (0, 0, -73)

    # Set up render products and annotators
    # These are the resolutions for Cosmos Transfer1 inputs
    render_product = rep.create.render_product("/Root/Worker/Camera", (1280, 704))

    # Initialize and attach writer
    rep.WriterRegistry.register(CosmosWriter)
    backend = rep.backends.get("DiskBackend")
    backend.initialize(output_dir=output_path)
    cosmos_writer = rep.writers.get("CosmosWriter")
    cosmos_writer.initialize(backend=backend, use_instance_id=False)
    cosmos_writer.attach(render_product)

    # Set up the scene ONCE
    forklift = stage.GetPrimAtPath("/Root/forklift")
    index = 0
    timeline.set_current_time(start_frame / timeline.get_time_codes_per_seconds())
    timeline.commit()

    # Iterate through the frames, moving the forklift
    for frame_id in range(end_frame):
        # Move Forklift
        newpos = interpolate(
            forklift_start, start_frame, forklift_end, end_frame, index
        )
        set_posrot(forklift, newpos, forklift_rot)
        index += 1

        await rep.orchestrator.step_async()
    rep.orchestrator.stop()


asyncio.ensure_future(run())
