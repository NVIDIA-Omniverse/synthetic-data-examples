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

    conference_tables = "omniverse://localhost/NVIDIA/Assets/ArchVis/Commercial/Conference/"

    # create randomizer function conference table assets.
    # This randomization includes placement and rotation of the assets on the surface.
    def env_conference_table(size=5):
        confTable = rep.randomizer.instantiate(
            rep.utils.get_usd_files(conference_tables, recursive=False),
            size=size,
            mode="scene_instance",
        )
        with confTable:
            rep.modify.pose(
                position=rep.distribution.uniform((-500, 0, -500), (500, 0, 500)),
                rotation=rep.distribution.uniform((-90, -180, 0), (-90, 180, 0)),
            )
        return confTable.node

    # Register randomization
    rep.randomizer.register(env_conference_table)

    # Setup camera and attach it to render product
    camera = rep.create.camera()
    render_product = rep.create.render_product(camera, resolution=(1024, 1024))

    surface = rep.create.disk(scale=100, visible=False)

    # trigger on frame for an interval
    with rep.trigger.on_frame(5):
        rep.randomizer.env_conference_table(2)
        rep.randomizer.dome_lights()
        with camera:
            rep.modify.pose(
                position=rep.distribution.uniform((-500, 200, 1000), (500, 500, 1500)),
                look_at=surface,
            )

    # Initialize and attach writer
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir="~/replicator_examples/dli_example_3", rgb=True)
    writer.attach([render_product])
