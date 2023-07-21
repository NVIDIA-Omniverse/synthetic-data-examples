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

import omni.replicator.core as rep

# A light to see
distance_light = rep.create.light(rotation=(-45, 0, 0), light_type="distant")

# Create a plane to sample on
plane_samp = rep.create.plane(scale=4, rotation=(20, 0, 0))

# create a larger sphere to sample on the surface of
sphere_samp = rep.create.sphere(scale=2.4, position=(0, 100, -180))


def randomize_spheres():
    # create small spheres to sample inside the plane
    spheres = rep.create.sphere(scale=0.4, count=60)

    # scatter small spheres
    with spheres:
        rep.randomizer.scatter_2d(
            [plane_samp, sphere_samp],
            min_samp=(None, None, None),
            # Small spheres will not go beyond 0 in X, 110 in Y, 30 in Z world space
            max_samp=(0, 110, 30),
            check_for_collisions=False,
        )
        # Add color to small spheres
        rep.randomizer.color(
            colors=rep.distribution.uniform((0.2, 0.2, 0.2), (1, 1, 1))
        )
    return spheres.node


rep.randomizer.register(randomize_spheres)

with rep.trigger.on_frame(num_frames=10):
    rep.randomizer.randomize_spheres()

og.Controller.evaluate_sync()
rep.orchestrator.preview()
