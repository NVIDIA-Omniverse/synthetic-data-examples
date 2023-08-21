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

# create new objects to be used in the dataset
with rep.new_layer():
    sphere = rep.create.sphere(
        semantics=[("class", "sphere")], position=(0, 100, 100), count=5
    )
    cube = rep.create.cube(
        semantics=[("class", "cube")], position=(200, 200, 100), count=5
    )
    cone = rep.create.cone(
        semantics=[("class", "cone")], position=(200, 400, 200), count=10
    )
    cylinder = rep.create.cylinder(
        semantics=[("class", "cylinder")], position=(200, 100, 200), count=5
    )

    # create new camera & render product and attach to camera
    camera = rep.create.camera(position=(0, 0, 1000))
    render_product = rep.create.render_product(camera, (1024, 1024))

    # create plane if needed (but unused here)
    plane = rep.create.plane(scale=10)

    # function to get shapes that you've created above, via their semantic labels
    def get_shapes():
        shapes = rep.get.prims(
            semantics=[
                ("class", "cube"),
                ("class", "sphere"),
                ("class", "cone"),
                ("class", "cylinder"),
            ]
        )
        with shapes:
            # assign textures to the different objects
            rep.randomizer.texture(
                textures=[
                    "omniverse://localhost/NVIDIA/Materials/vMaterials_2/Ground/textures/aggregate_exposed_diff.jpg",
                    "omniverse://localhost/NVIDIA/Materials/vMaterials_2/Ground/textures/gravel_track_ballast_diff.jpg",
                    "omniverse://localhost/NVIDIA/Materials/vMaterials_2/Ground/textures/gravel_track_ballast_multi_R_rough_G_ao.jpg",
                    "omniverse://localhost/NVIDIA/Materials/vMaterials_2/Ground/textures/rough_gravel_rough.jpg",
                ]
            )

            # modify pose and distribution
            rep.modify.pose(
                position=rep.distribution.uniform((-500, 50, -500), (500, 50, 500)),
                rotation=rep.distribution.uniform((0, -180, 0), (0, 180, 0)),
                scale=rep.distribution.normal(1, 0.5),
            )
        return shapes.node

    # register the get shapes function as a randomizer function
    rep.randomizer.register(get_shapes)
    # Setup randomization. 100 variations here from 'num_frames'
    with rep.trigger.on_frame(num_frames=100):
        rep.randomizer.get_shapes()

    # Initialize and attach writer
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir="~/replicator_examples/dli_example_22", rgb=True)
    writer.attach([render_product])

    rep.orchestrator.run()
