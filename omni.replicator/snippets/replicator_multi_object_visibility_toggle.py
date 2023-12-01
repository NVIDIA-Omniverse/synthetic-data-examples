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

"""This will create a group from a list of objects and
1. Render all the objects together
2. Toggle sole visiblity for each object & render
3. Randomize pose for all objects, repeat

This can be useful for training on object occlusions.
"""
import omni.replicator.core as rep

NUM_POSE_RANDOMIZATIONS = 10


# Make a list-of-lists of True/False for each object
# In this example of 3 objects:
# [[True, True, True]
# [True, False, False]
# [False, True, False]
# [False, False, True]]
def make_visibility_lists(num_objects):
    visib = []
    # Make an all-visible first pass
    visib.append(tuple([True for x in range(num_objects)]))

    # List to toggle one object visible at a time
    for x in range(num_objects):
        sub_vis = []
        for i in range(num_objects):
            if x == i:
                sub_vis.append(True)
            else:
                sub_vis.append(False)
        visib.append(tuple(sub_vis))
    return visib


with rep.new_layer():
    # Setup camera and simple light
    camera = rep.create.camera(position=(0, 500, 1000), look_at=(0, 0, 0))
    light = rep.create.light(rotation=(-45, 45, 0))

    # Create simple shapes to manipulate
    plane = rep.create.plane(
        semantics=[("class", "plane")], position=(0, -100, 0), scale=(100, 1, 100)
    )
    torus = rep.create.torus(semantics=[("class", "torus")], position=(200, 0, 100))
    sphere = rep.create.sphere(semantics=[("class", "sphere")], position=(0, 0, 100))
    cube = rep.create.cube(semantics=[("class", "cube")], position=(-200, 0, 100))

    # Create a group of the objects we will be manipulating
    # Leaving-out camera, light, and plane from visibility toggling and pose randomization
    object_group = rep.create.group([torus, sphere, cube])

    # Get the number of objects to toggle, can work with any number of objects
    num_objects_to_toggle = len(object_group.get_output_prims()["prims"])
    # Create our lists-of-lists for visibility
    visibility_sequence = make_visibility_lists(num_objects_to_toggle)

    # Trigger to toggle visibility one at a time
    with rep.trigger.on_frame(
        max_execs=(num_objects_to_toggle + 1) * NUM_POSE_RANDOMIZATIONS
    ):
        with object_group:
            rep.modify.visibility(rep.distribution.sequence(visibility_sequence))
    # Trigger to randomize position and scale, interval set to number of objects +1(1 extra for the "all visible" frame)
    with rep.trigger.on_frame(
        max_execs=NUM_POSE_RANDOMIZATIONS, interval=num_objects_to_toggle + 1
    ):
        with object_group:
            rep.modify.pose(
                position=rep.distribution.uniform((-300, 0, -300), (300, 0, 300)),
                scale=rep.distribution.uniform(0.1, 2),
            )

# Initialize render product and attach writer
render_product = rep.create.render_product(camera, (512, 512))
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(
    output_dir="toggle_multi_visibility",
    rgb=True,
    semantic_segmentation=True,
)
writer.attach([render_product])
rep.orchestrator.run()
