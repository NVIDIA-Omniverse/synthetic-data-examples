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

import datetime

now = datetime.datetime.now()
from functools import partial
import omni.replicator.core as rep

with rep.new_layer():
    # Define paths for the character, the props, the environment and the surface where the assets will be scattered in.
    CRATE = "omniverse://localhost/NVIDIA/Samples/Marbles/assets/standalone/SM_room_crate_3/SM_room_crate_3.usd"
    SURFACE = "omniverse://localhost/NVIDIA/Assets/Scenes/Templates/Basic/display_riser.usd"
    ENVS = "omniverse://localhost/NVIDIA/Assets/Scenes/Templates/Interior/ZetCG_ExhibitionHall.usd"
    FRUIT_PROPS = {
        "apple": "omniverse://localhost/NVIDIA/Assets/ArchVis/Residential/Food/Fruit/Apple.usd",
        "avocado": "omniverse://localhost/NVIDIA/Assets/ArchVis/Residential/Food/Fruit/Avocado01.usd",
        "kiwi": "omniverse://localhost/NVIDIA/Assets/ArchVis/Residential/Food/Fruit/Kiwi01.usd",
        "lime": "omniverse://localhost/NVIDIA/Assets/ArchVis/Residential/Food/Fruit/Lime01.usd",
        "lychee": "omniverse://localhost/NVIDIA/Assets/ArchVis/Residential/Food/Fruit/Lychee01.usd",
        "pomegranate": "omniverse://localhost/NVIDIA/Assets/ArchVis/Residential/Food/Fruit/Pomegranate01.usd",
        "onion": "omniverse://localhost/NVIDIA/Assets/ArchVis/Residential/Food/Vegetables/RedOnion.usd",
        "strawberry": "omniverse://localhost/NVIDIA/Assets/ArchVis/Residential/Food/Berries/strawberry.usd",
        "lemon": "omniverse://localhost/NVIDIA/Assets/ArchVis/Residential/Decor/Tchotchkes/Lemon_01.usd",
        "orange": "omniverse://localhost/NVIDIA/Assets/ArchVis/Residential/Decor/Tchotchkes/Orange_01.usd",
    }

    # Define randomizer function for Base assets. This randomization includes placement and rotation of the assets on the surface.
    def random_props(file_name, class_name, max_number=1, one_in_n_chance=3):
        instances = rep.randomizer.instantiate(file_name, size=max_number, mode="scene_instance")
        print(file_name)
        with instances:
            rep.modify.semantics([("class", class_name)])
            rep.modify.pose(
                position=rep.distribution.uniform((-8, 5, -25), (8, 30, 25)),
                rotation=rep.distribution.uniform((-180, -180, -180), (180, 180, 180)),
                scale=rep.distribution.uniform((0.8), (1.2)),
            )
            rep.modify.visibility(rep.distribution.choice([True], [False] * (one_in_n_chance)))
        return instances.node

    # Define randomizer function for sphere lights.
    def sphere_lights(num):
        lights = rep.create.light(
            light_type="Sphere",
            temperature=rep.distribution.normal(6500, 500),
            intensity=rep.distribution.normal(30000, 5000),
            position=rep.distribution.uniform((-300, -300, -300), (300, 300, 300)),
            scale=rep.distribution.uniform(50, 100),
            count=num,
        )
        return lights.node

    rep.randomizer.register(random_props)

    # Setup the static elements
    env = rep.create.from_usd(ENVS)
    surface = rep.create.from_usd(SURFACE)
    with surface:
        rep.physics.collider()
    crate = rep.create.from_usd(CRATE)
    with crate:
        rep.physics.collider("none")
        rep.physics.mass(mass=10000)
        rep.modify.pose(position=(0, 20, 0), rotation=(0, 0, 90))

    # Setup camera and attach it to render product
    camera = rep.create.camera()
    render_product = rep.create.render_product(camera, resolution=(1024, 1024))

    rep.randomizer.register(sphere_lights)
    # trigger on frame for an interval
    with rep.trigger.on_frame(num_frames=100):
        for n, f in FRUIT_PROPS.items():
            random_props(f, n)
        rep.randomizer.sphere_lights(5)
        with camera:
            rep.modify.pose(position=rep.distribution.uniform((-3, 114, -17), (-1, 116, -15)), look_at=(0, 20, 0))

    # Initialize and attach writer
    writer = rep.WriterRegistry.get("BasicWriter")
    now = now.strftime("%Y-%m-%d")
    output_dir = "fruit_data_" + now
    writer.initialize(output_dir=output_dir, rgb=True, bounding_box_2d_tight=True)
    writer.attach([render_product])
