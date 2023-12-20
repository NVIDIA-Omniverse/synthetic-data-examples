# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
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
#
"""
This snippet shows how to setup multiple independent triggers that happen
at different intervals in the simulation.
"""
import omni.graph.core as og
import omni.replicator.core as rep

# A light to see
distance_light = rep.create.light(rotation=(-45, 0, 0), light_type="distant")

# Create a plane to sample on
plane_samp = rep.create.plane(scale=4, rotation=(20, 0, 0))

# Create a larger sphere to sample on the surface of
sphere_samp = rep.create.sphere(scale=2.4, position=(0, 100, -180))

# Create a larger cylinder we do not want to collide with
cylinder = rep.create.cylinder(semantics=[("class", "cylinder")], scale=(2, 1, 2))


def randomize_spheres():
    # create small spheres to sample inside the plane
    spheres = rep.create.sphere(scale=0.4, count=60)

    # scatter small spheres
    with spheres:
        rep.randomizer.scatter_2d(
            surface_prims=[plane_samp, sphere_samp],
            no_coll_prims=[cylinder],
            check_for_collisions=True,
        )
        # Add color to small spheres
        rep.randomizer.color(
            colors=rep.distribution.uniform((0.2, 0.2, 0.2), (1, 1, 1))
        )
    return spheres.node


rep.randomizer.register(randomize_spheres)

# Trigger will execute 5 times, every-other-frame (interval=2)
with rep.trigger.on_frame(num_frames=5, interval=2):
    rep.randomizer.randomize_spheres()
# Trigger will execute 10 times, once every frame
with rep.trigger.on_frame(num_frames=10):
    with cylinder:
        rep.modify.visibility(rep.distribution.sequence([True, False]))

og.Controller.evaluate_sync()  # Only for snippet demonstration preview, not needed for production
rep.orchestrator.preview()  # Only for snippet demonstration preview, not needed for production

rp = rep.create.render_product("/OmniverseKit_Persp", (1024, 768))

# Initialize and attach writer
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(output_dir="scatter_example", rgb=True)
writer.attach([rp])

rep.orchestrator.run()
