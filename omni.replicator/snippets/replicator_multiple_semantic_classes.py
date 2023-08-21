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

import omni.replicator.core as rep

with rep.new_layer():
    sphere = rep.create.sphere(semantics=[("class", "sphere")], position=(0, 100, 100))
    cube = rep.create.cube(semantics=[("class2", "cube")], position=(200, 200, 100))
    plane = rep.create.plane(semantics=[("class3", "plane")], scale=10)

    def get_shapes():
        shapes = rep.get.prims(semantics=[("class", "cube"), ("class", "sphere")])
        with shapes:
            rep.modify.pose(
                position=rep.distribution.uniform((-500, 50, -500), (500, 50, 500)),
                rotation=rep.distribution.uniform((0, -180, 0), (0, 180, 0)),
                scale=rep.distribution.normal(1, 0.5),
            )
        return shapes.node

    with rep.trigger.on_frame(num_frames=2):
        rep.randomizer.register(get_shapes)

    # Setup Camera
    camera = rep.create.camera(position=(500, 500, 500), look_at=(0, 0, 0))

render_product = rep.create.render_product(camera, (512, 512))


writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(
    output_dir="semantics_classes",
    rgb=True,
    semantic_segmentation=True,
    colorize_semantic_segmentation=True,
    semantic_types=["class", "class2", "class3"],
)

writer.attach([render_product])
rep.orchestrator.run()
