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
    # Define paths for the character, the props, the environment and the surface where the assets will be scattered in.
    PROPS = "omniverse://localhost/NVIDIA/Assets/Isaac/2022.1/Isaac/Props/YCB/Axis_Aligned_Physics"
    SURFACE = (
        "omniverse://localhost/NVIDIA/Assets/Scenes/Templates/Basic/display_riser.usd"
    )
    ENVS = "omniverse://localhost/NVIDIA/Assets/Scenes/Templates/Interior/ZetCG_ExhibitionHall.usd"

    # Define randomizer function for Base assets. This randomization includes placement and rotation of the assets on the surface.
    def env_props(size=50):
        instances = rep.randomizer.instantiate(
            rep.utils.get_usd_files(PROPS, recursive=True),
            size=size,
            mode="scene_instance",
        )
        with instances:
            rep.modify.pose(
                position=rep.distribution.uniform((-50, 5, -50), (50, 20, 50)),
                rotation=rep.distribution.uniform((0, -180, 0), (0, 180, 0)),
                scale=100,
            )

            rep.physics.rigid_body(
                velocity=rep.distribution.uniform((-0, 0, -0), (0, 0, 0)),
                angular_velocity=rep.distribution.uniform((-0, 0, -100), (0, 0, 0)),
            )
        return instances.node

    # Register randomization
    rep.randomizer.register(env_props)
    # Setup the static elements
    env = rep.create.from_usd(ENVS)
    surface = rep.create.from_usd(SURFACE)
    with surface:
        rep.physics.collider()

    # Setup camera and attach it to render product
    camera = rep.create.camera()
    render_product = rep.create.render_product(camera, resolution=(1024, 1024))

    # sphere lights for extra randomization
    def sphere_lights(num):
        lights = rep.create.light(
            light_type="Sphere",
            temperature=rep.distribution.normal(6500, 500),
            intensity=rep.distribution.normal(35000, 5000),
            position=rep.distribution.uniform((-300, -300, -300), (300, 300, 300)),
            scale=rep.distribution.uniform(50, 100),
            count=num,
        )
        return lights.node

    rep.randomizer.register(sphere_lights)
    # trigger on frame for an interval

    with rep.trigger.on_time(interval=2, num=10):
        rep.randomizer.env_props(10)
        rep.randomizer.sphere_lights(10)
        with camera:
            rep.modify.pose(
                position=rep.distribution.uniform((-50, 20, 100), (50, 50, 150)),
                look_at=surface,
            )
    # Initialize and attach writer
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir="~/replicator_examples/dli_physics",
        rgb=True,
        bounding_box_2d_tight=True,
    )
    writer.attach([render_product])
