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


with rep.new_layer():

    def dome_lights():
        lights = rep.create.light(
            light_type="Dome",
            rotation=(270, 0, 0),
            texture=rep.distribution.choice(
                [
                    "omniverse://localhost/NVIDIA/Assets/Skies/Indoor/ZetoCGcom_ExhibitionHall_Interior1.hdr",
                    "omniverse://localhost/NVIDIA/Assets/Skies/Indoor/ZetoCG_com_WarehouseInterior2b.hdr",
                ]
            ),
        )
        return lights.node

    rep.randomizer.register(dome_lights)

    bar_stools = "omniverse://localhost/NVIDIA/Assets/ArchVis/Residential/Furniture/BarStools/"
    reception_tables = "omniverse://localhost/NVIDIA/Assets/ArchVis/Commercial/Reception/"
    chairs = "omniverse://localhost/NVIDIA/Assets/ArchVis/Commercial/Seating/"
    normal_tables = "omniverse://localhost/NVIDIA/Assets/ArchVis/Commercial/Tables/"
    sofa_set = "omniverse://localhost/NVIDIA/Assets/ArchVis/Residential/Furniture/FurnitureSets/Dutchtown/"

    # create randomizer function conference table assets.
    # This randomization includes placement and rotation of the assets on the surface.
    def env_bar_stools(size=5):
        barstool = rep.randomizer.instantiate(
            rep.utils.get_usd_files(bar_stools, recursive=False),
            size=size,
            mode="scene_instance",
        )
        rep.modify.semantics([("class", "chair_barstool")])
        with barstool:
            rep.modify.pose(
                position=rep.distribution.uniform((-500, 0, -500), (500, 0, 500)),
                rotation=rep.distribution.uniform((-90, -180, 0), (-90, 180, 0)),
            )
        return barstool.node

    # Register randomization
    rep.randomizer.register(env_bar_stools)

    # create randomizer function conference table assets.
    # This randomization includes placement and rotation of the assets on the surface.
    def env_reception_table(size=5):
        receptTable = rep.randomizer.instantiate(
            rep.utils.get_usd_files(reception_tables, recursive=False),
            size=size,
            mode="scene_instance",
        )
        rep.modify.semantics([("class", "reception_table")])
        with receptTable:
            rep.modify.pose(
                position=rep.distribution.uniform((-500, 0, -500), (500, 0, 500)),
                rotation=rep.distribution.uniform((-90, -180, 0), (-90, 180, 0)),
            )
        return receptTable.node

    # Register randomization
    rep.randomizer.register(env_reception_table)

    # create randomizer function conference table assets.
    # This randomization includes placement and rotation of the assets on the surface.
    def env_chairs_notable(size=10):
        chairsEnv = rep.randomizer.instantiate(
            rep.utils.get_usd_files(chairs, recursive=False),
            size=size,
            mode="scene_instance",
        )
        rep.modify.semantics([("class", "office_chairs")])
        with chairsEnv:
            rep.modify.pose(
                position=rep.distribution.uniform((-500, 0, -500), (500, 0, 500)),
                rotation=rep.distribution.uniform((-90, -180, 0), (-90, 180, 0)),
            )
        return chairsEnv.node

    # Register randomization
    rep.randomizer.register(env_chairs_notable)

    # create randomizer function conference table assets.
    # This randomization includes placement and rotation of the assets on the surface.
    def env_normal_table(size=5):
        normTable = rep.randomizer.instantiate(
            rep.utils.get_usd_files(normal_tables, recursive=False),
            size=size,
            mode="scene_instance",
        )
        rep.modify.semantics([("class", "normal_table")])
        with normTable:
            rep.modify.pose(
                position=rep.distribution.uniform((-500, 0, -500), (500, 0, 500)),
                rotation=rep.distribution.uniform((-90, -180, 0), (-90, 180, 0)),
            )
        return normTable.node

    # Register randomization
    rep.randomizer.register(env_normal_table)

    # create randomizer function conference table assets.
    # This randomization includes placement and rotation of the assets on the surface.
    def env_sofaset(size=5):
        sofaset = rep.randomizer.instantiate(
            rep.utils.get_usd_files(sofa_set, recursive=False),
            size=size,
            mode="scene_instance",
        )
        rep.modify.semantics([("class", "sofa")])
        with sofaset:
            rep.modify.pose(
                position=rep.distribution.uniform((-500, 0, -500), (500, 0, 500)),
                rotation=rep.distribution.uniform((-90, -180, 0), (-90, 180, 0)),
            )
        return sofaset.node

    # Register randomization
    rep.randomizer.register(env_sofaset)

    # Setup camera and attach it to render product
    camera = rep.create.camera()
    render_product = rep.create.render_product(camera, resolution=(1024, 1024))

    # sphere lights for bonus challenge randomization
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

    # create the surface for the camera to focus on
    surface = rep.create.disk(scale=100, visible=False)

    # trigger on frame for an interval
    with rep.trigger.on_frame(5):
        rep.randomizer.env_bar_stools(2)
        rep.randomizer.env_reception_table(2)
        rep.randomizer.env_chairs_notable(2)
        rep.randomizer.env_normal_table(2)
        rep.randomizer.env_sofaset(2)

        # rep.randomizer.sphere_lights(10)
        rep.randomizer.dome_lights()
        with camera:
            rep.modify.pose(
                position=rep.distribution.uniform((-500, 200, 1000), (500, 500, 1500)),
                look_at=surface,
            )

    # Initialize and attach writer
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir="~/replicator_examples/dli_example_33", rgb=True)
    writer.attach([render_product])
