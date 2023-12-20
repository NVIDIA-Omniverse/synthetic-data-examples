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
This snippet shows how to modify attributes on prims that Replicator
may not have a direct functional mapping for.
"""
import omni.replicator.core as rep


with rep.new_layer():
    camera = rep.create.camera(position=(0, 500, 1000), look_at=(0, 0, 0))

    # Create simple shapes to manipulate
    plane = rep.create.plane(
        semantics=[("class", "plane")], position=(0, -100, 0), scale=(100, 1, 100)
    )
    cubes = rep.create.cube(
        semantics=[("class", "cube")],
        position=rep.distribution.uniform((-300, 0, -300), (300, 0, 300)),
        count=6,
    )
    spheres = rep.create.sphere(
        semantics=[("class", "sphere")],
        position=rep.distribution.uniform((-300, 0, -300), (300, 0, 300)),
        count=6,
    )
    lights = rep.create.light(
        light_type="Sphere",
        intensity=rep.distribution.normal(500, 35000),
        position=rep.distribution.uniform((-300, 300, -300), (300, 1000, 300)),
        scale=rep.distribution.uniform(50, 100),
        count=3,
    )
    with rep.trigger.on_frame(num_frames=10):
        with lights:
            rep.modify.pose(
                position=rep.distribution.uniform((-300, 300, -300), (300, 1000, 300))
            )
            rep.modify.attribute("intensity", rep.distribution.uniform(1.0, 50000.0))
            rep.modify.attribute(
                "color", rep.distribution.normal((0.2, 0.2, 0.2), (1.0, 1.0, 1.0))
            )
