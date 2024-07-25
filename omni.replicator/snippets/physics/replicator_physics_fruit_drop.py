# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# This example demonstrates using Omniverse Replicator with physics and
# multiple objects. A number of fruit objects are chosen, have physics applied
# and are sequentially dropped to fill a fruit crate.

import datetime
import omni.replicator.core as rep

# IsaacSim default is "Z", Other application might have "Y"-up
rep.settings.set_stage_up_axis("Z")
rep.settings.set_stage_meters_per_unit(0.01)  # Set the correct units


# Asset URLs for the props we want to use
CRATE = "omniverse://localhost/NVIDIA/Samples/Marbles/assets/standalone/SM_room_crate_3/SM_room_crate_3.usd"
SURFACE = "omniverse://localhost/NVIDIA/Assets/Scenes/Templates/Basic/display_riser.usd"
ENVS = "omniverse://localhost/NVIDIA/Assets/Scenes/Templates/Interior/ZetCG_ExhibitionHall.usd"

# A dictionary mapping fruit label to the asset URL, if we wished to semantically label the object
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
FRUIT_USD_LIST = list(FRUIT_PROPS.values())


def create_props():
    # Instantiate fruit objects
    # Choice - 500 samples of 1 item from FRUIT_USD_LIST
    # Without Sequence, it would choose 500 samples from all of the fruits - making a mix of fruits
    # for each drop
    instances = rep.randomizer.instantiate(
        rep.distribution.choice(
            rep.distribution.sequence(FRUIT_USD_LIST), num_samples=500
        ),
        size=1,
        mode="scene_instance",
    )
    # Randomize the position, rotation, and scale of the fruit objects slightly
    with instances:
        rep.modify.pose(
            position=rep.distribution.uniform((-8, 5, 100), (8, 30, 300)),
            rotation=rep.distribution.uniform((-180, -180, -180), (180, 180, 180)),
            scale=rep.distribution.uniform((0.8), (1.2)),
        )
        rep.physics.rigid_body()
    return instances.node


# Register our custom function with Replicator
rep.randomizer.register(create_props)

with rep.new_layer():
    # Setup surface object and make it a collider
    surface = rep.create.from_usd(SURFACE)
    with surface:
        rep.modify.pose(rotation=(90, 0, 0))
        rep.physics.collider()

    # Setup crate, orientation, size, and give it a large mass so it does not move
    # from collisions
    crate = rep.create.from_usd(CRATE)
    with crate:
        rep.modify.pose(position=(0, 0, 20.5), rotation=(0, -90, 0), scale=(1, 3, 2))
        rep.physics.collider("none")
        rep.physics.mass(mass=10000)

    # Setup camera and attach it to render product
    camera = rep.create.camera(position=(120, 160, 200), look_at=(0, 0, 0))
    render_product = rep.create.render_product(camera, resolution=(1024, 1024))
    wide_camera = rep.create.camera(
        name="fruit_camera", position=(300, 300, 300), look_at=(0, 0, 0)
    )

    # The on_time trigger is needed to run physics
    # The interval of 2 means every 2 seconds, a new batch of fruit will be dropped.
    with rep.trigger.on_time(max_execs=10, interval=2):
        rep.randomizer.create_props()

    # Getthing the date as a string, so this won't overwrite images generated on different generation days
    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d")
    output_dir = "fruit_data_" + now

    # Initialize and attach writer.  A BasicWriter that outputs RGB images and 2D Bounding box labels
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=output_dir, rgb=True, bounding_box_2d_tight=False)
    writer.attach([render_product])
