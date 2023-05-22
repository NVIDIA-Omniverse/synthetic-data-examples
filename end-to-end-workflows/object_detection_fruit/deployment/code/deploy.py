# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import tritonclient.grpc as grpcclient
from optparse import OptionParser

# load image data
import cv2
import numpy as np
from matplotlib import pyplot as plt

import subprocess


def install(name):
    subprocess.call(["pip", "install", name])


"""
Parses command line options. Requires input sample png
"""


def parse_input():
    usage = "usage: deploy.py [options] arg1 "
    parser = OptionParser(usage)
    parser.add_option(
        "-p", "--png", dest="png", help="Directory location for single sample image."
    )
    (options, args) = parser.parse_args()
    return options, args


def main():
    options, args = parse_input()
    target_width, target_height = 1024, 1024

    # add path to test image
    image_sample = options.png
    image_bgr = cv2.imread(image_sample)
    image_bgr
    image_bgr = cv2.resize(image_bgr, (target_width, target_height))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = np.float32(image_rgb)

    # preprocessing
    image = image / 255
    image = np.moveaxis(image, -1, 0)  # HWC to CHW

    image = image[np.newaxis, :]  # add batch dimension
    image = np.float32(image)

    plt.imshow(image_rgb)

    inference_server_url = "0.0.0.0:9001"
    triton_client = grpcclient.InferenceServerClient(url=inference_server_url)

    # find out info about model
    model_name = "fasterrcnn_resnet50"
    triton_client.get_model_config(model_name)

    # create input
    input_name = "input"
    inputs = [grpcclient.InferInput(input_name, image.shape, "FP32")]
    inputs[0].set_data_from_numpy(image)

    output_name = "output"
    outputs = [grpcclient.InferRequestedOutput("output")]

    results = triton_client.infer(model_name, inputs, outputs=outputs)

    output = results.as_numpy("output")

    # annotate
    annotated_image = image_bgr.copy()

    if output.size > 0:  # ensure something is found
        for box in output:
            box_top_left = int(box[0]), int(box[1])
            box_bottom_right = int(box[2]), int(box[3])
            text_origin = int(box[0]), int(box[3])

            border_color = (50, 0, 100)
            text_color = (255, 255, 255)

            font_scale = 0.9
            thickness = 1

            # bounding box
            cv2.rectangle(
                annotated_image,
                box_top_left,
                box_bottom_right,
                border_color,
                thickness=5,
                lineType=cv2.LINE_8,
            )

    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))


if __name__ == "__main__":
    main()
