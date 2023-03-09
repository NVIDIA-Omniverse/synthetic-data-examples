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

import omni.replicator.core as rep

# Create a new layer for our work to be performed in.
# This is a good habit to develop for later when working on existing Usd scenes
with rep.new_layer():

    # Create a simple camera with a position and a point to look at
    camera = rep.create.camera(position=(0, 500, 1000), look_at=(0, 0, 0))

    # Create some simple shapes to manipulate
    plane = rep.create.plane(semantics=[("class", "plane")], position=(0, -100, 0), scale=(100, 1, 100))
    torus = rep.create.torus(semantics=[("class", "torus")], position=(200, 0, 100))
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
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(
    output_dir="~/replicator_examples/dli_hello_replicator/",
    rgb=True,
    semantic_segmentation=True,
    bounding_box_2d_tight=True,
)
writer.attach([render_product])
rep.orchestrator.run()
