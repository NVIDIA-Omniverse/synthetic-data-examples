# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import einops
import tensorrt as trt
import torch2trt
from typing import Sequence


BOX_EDGES = [
    [0, 1],
    [1, 5],
    [5, 4],
    [4, 0],
    [2, 3],
    [3, 7],
    [7, 6],
    [6, 2],
    [0, 2],
    [1, 3],
    [4, 6],
    [5, 7]
]


def make_offset_grid(
        size,
        stride=(1, 1)
    ):

    grid = torch.stack(
        torch.meshgrid(
            stride[0] * (torch.arange(size[0]) + 0.5),
            stride[1] * (torch.arange(size[1]) + 0.5)
        ),
        dim=-1
    )

    return grid


def vectormap_to_keypointmap(
        offset_grid, 
        vector_map, 
        vector_scale: float = 1./256.
    ):

    vector_map = vector_map / vector_scale
    keypoint_map = einops.rearrange(vector_map, "b (k d) h w -> b h w k d", d=2) 
    keypoint_map = keypoint_map + offset_grid[:, :, None, :]

    # yx -> xy
    keypoint_map = keypoint_map[..., [1, 0]]

    return keypoint_map


def find_heatmap_peak_mask(heatmap, window=3, threshold=0.5):

    all_indices = torch.arange(
        heatmap.numel(), 
        device=heatmap.device
    )

    all_indices = all_indices.reshape(heatmap.shape)

    if isinstance(window, int):
        window = (window, window)

    values, max_indices = F.max_pool2d_with_indices(
        heatmap,
        kernel_size=window,
        stride=1,
        padding=(window[0] // 2, window[1] // 2)
    )

    is_above_threshold = heatmap >= threshold
    is_max = max_indices == all_indices
    is_peak = is_above_threshold & is_max

    return is_peak


def draw_box(image_bgr, keypoints, color=(118, 186, 0), thickness=1):

    num_objects = int(keypoints.shape[0])
    for i in range(num_objects):
        keypoints_i = keypoints[i]
        kps_i = [(int(x), int(y)) for x, y in keypoints_i]

        edges = BOX_EDGES
        for e in edges:
            cv2.line(
                image_bgr, 
                kps_i[e[0]], 
                kps_i[e[1]], 
                (118, 186, 0), 
                thickness=thickness
            )
    
    return image_bgr


def pad_resize(image, output_shape):

    ar_i = image.shape[1] / image.shape[0]
    ar_o = output_shape[1] / output_shape[0]

    # resize
    if ar_i > ar_o:
        w_i = output_shape[1]
        h_i = min(int(w_i / ar_i), output_shape[0])
    else:
        h_i = output_shape[0]
        w_i = min(int(h_i * ar_i), output_shape[1])

    # paste
    pad_left = (output_shape[1] - w_i) // 2
    pad_top = (output_shape[0] - h_i) // 2

    image_resize = cv2.resize(image, (w_i, h_i))

    out = np.zeros_like(
        image, 
        shape=(output_shape[0], output_shape[1], image.shape[2])
    )

    out[pad_top:pad_top + h_i, pad_left:pad_left + w_i] = image_resize

    pad = (pad_top, pad_left)
    scale = (image.shape[0] / h_i, image.shape[1] / w_i)

    return out, pad, scale


def load_trt_engine(path: str):

    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        with open(path, 'rb') as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)

    return engine


def load_trt_engine_wrapper(
        path: str, 
        input_names: Sequence, 
        output_names: Sequence
    ):

    engine = load_trt_engine(path)

    wrapper = torch2trt.TRTModule(
        engine=engine,
        input_names=input_names,
        output_names=output_names
    )

    return wrapper


def format_bgr8_image(image, device="cuda"):

    x = torch.from_numpy(image)
    x = x.permute(2, 0, 1)[None, ...]
    x = (x / 255 - 0.45) / 0.25

    return x