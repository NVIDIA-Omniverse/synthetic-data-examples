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
"""
A snippet showing to how create a custom writer to output specific colors
in the semantic annotator output image.
"""

import omni.replicator.core as rep
from omni.replicator.core import Writer, BackendDispatch, WriterRegistry


class MyWriter(Writer):
    def __init__(self, output_dir: str):
        self._frame_id = 0
        self.backend = BackendDispatch({"paths": {"out_dir": output_dir}})
        self.annotators = ["rgb", "semantic_segmentation"]
        # Dictionary mapping of label to RGBA color
        self.CUSTOM_LABELS = {
            "unlabelled": (0, 0, 0, 0),
            "sphere": (128, 64, 128, 255),
            "cube": (244, 35, 232, 255),
            "plane": (102, 102, 156, 255),
        }

    def write(self, data):
        render_products = [k for k in data.keys() if k.startswith("rp_")]
        self._write_rgb(data, "rgb")
        self._write_segmentation(data, "semantic_segmentation")
        self._frame_id += 1

    def _write_rgb(self, data, annotator: str):
        # Save the rgb data under the correct path
        rgb_file_path = f"rgb_{self._frame_id}.png"
        self.backend.write_image(rgb_file_path, data[annotator])

    def _write_segmentation(self, data, annotator: str):
        seg_filepath = f"seg_{self._frame_id}.png"
        semantic_seg_data_colorized = rep.tools.colorize_segmentation(
            data[annotator]["data"],
            data[annotator]["info"]["idToLabels"],
            mapping=self.CUSTOM_LABELS,
        )
        self.backend.write_image(seg_filepath, semantic_seg_data_colorized)

    def on_final_frame(self):
        self.backend.sync_pending_paths()


# Register new writer
WriterRegistry.register(MyWriter)

# Create a new layer for our work to be performed in.
# This is a good habit to develop for later when working on existing Usd scenes
with rep.new_layer():
    light = rep.create.light(light_type="dome")
    # Create a simple camera with a position and a point to look at
    camera = rep.create.camera(position=(0, 500, 1000), look_at=(0, 0, 0))

    # Create some simple shapes to manipulate
    plane = rep.create.plane(
        semantics=[("class", "plane")], position=(0, -100, 0), scale=(100, 1, 100)
    )
    torus = rep.create.torus(position=(200, 0, 100))  # Torus will be unlabeled
    sphere = rep.create.sphere(semantics=[("class", "sphere")], position=(0, 0, 100))
    cube = rep.create.cube(semantics=[("class", "cube")], position=(-200, 0, 100))

    # Randomize position and scale of each object on each frame
    with rep.trigger.on_frame(num_frames=10):
        # Creating a group so that our modify.pose operation works on all the shapes at once
        with rep.create.group([torus, sphere, cube]):
            rep.modify.pose(
                position=rep.distribution.uniform((-300, 0, -300), (300, 0, 300)),
                scale=rep.distribution.uniform(0.1, 2),
            )

# Initialize render product and attach a writer
render_product = rep.create.render_product(camera, (1024, 1024))
writer = rep.WriterRegistry.get("MyWriter")
writer.initialize(output_dir="myWriter_output")
writer.attach([render_product])
rep.orchestrator.run()  # Run the simulation
