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

    def dome_lights():
        lights = rep.create.light(
            light_type="Dome",
            rotation=(270, 0, 0),
            texture=rep.distribution.choice(
                [
                    "omniverse://localhost/NVIDIA/Assets/Skies/Cloudy/champagne_castle_1_4k.hdr",
                    "omniverse://localhost/NVIDIA/Assets/Skies/Clear/evening_road_01_4k.hdr",
                    "omniverse://localhost/NVIDIA/Assets/Skies/Cloudy/kloofendal_48d_partly_cloudy_4k.hdr",
                    "omniverse://localhost/NVIDIA/Assets/Skies/Clear/qwantani_4k.hdr",
                ]
            ),
        )
        return lights.node

    rep.randomizer.register(dome_lights)

    torus = rep.create.torus(semantics=[("class", "torus")], position=(0, -200, 100))

    # create surface
    surface = rep.create.disk(scale=5, visible=False)

    # create camera & render product for the scene
    camera = rep.create.camera()
    render_product = rep.create.render_product(camera, resolution=(1024, 1024))

    with rep.trigger.on_frame(num_frames=10, interval=10):
        rep.randomizer.dome_lights()
        with rep.create.group([torus]):
            rep.modify.pose(
                position=rep.distribution.uniform((-100, -100, -100), (200, 200, 200)),
                scale=rep.distribution.uniform(0.1, 2),
            )
        with camera:
            rep.modify.pose(
                position=rep.distribution.uniform((-500, 200, 1000), (500, 500, 1500)),
                look_at=surface,
            )

    # Initialize and attach writer
    writer = rep.WriterRegistry.get("BasicWriter")

    writer.initialize(output_dir="~/replicator_examples/dli_example_02", rgb=True)
    writer.attach([render_product])

# Run Replicator
# rep.orchestrator.run()
