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

from pathlib import Path

import carb
import omni.replicator.core as rep
import omni.usd
from pxr import Sdf, UsdGeom

"""
Instructions:
Open the example scene file "scratches_randomization.usda",
located adjacent to this script, in Omniverse prior to using this script
"""

# Get the current Usd "stage".  This is where all the scene objects live
stage = omni.usd.get_context().get_stage()

with rep.new_layer():
    camera = rep.create.camera(position=(-30, 38, 60), look_at=(0, 0, 0))
    render_product = rep.create.render_product(camera, (1280, 720))

    # Get Scene cube
    cube_prim = stage.GetPrimAtPath("/World/RoundedCube2/Cube/Cube")

    # Set the primvars on the cubes once
    primvars_api = UsdGeom.PrimvarsAPI(cube_prim)
    primvars_api.CreatePrimvar("random_color", Sdf.ValueTypeNames.Float3).Set(
        (1.0, 1.0, 1.0)
    )
    primvars_api.CreatePrimvar("random_intensity", Sdf.ValueTypeNames.Float3).Set(
        (1.0, 1.0, 1.0)
    )

    def change_colors():
        # Change color primvars

        cubes = rep.get.prims(
            path_pattern="/World/RoundedCube2/Cube/Cube", prim_types=["Mesh"]
        )
        with cubes:
            rep.modify.attribute(
                "primvars:random_color",
                rep.distribution.uniform((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                attribute_type="float3",
            )
            rep.modify.attribute(
                "primvars:random_intensity",
                rep.distribution.uniform((0.0, 0.0, 0.0), (10.0, 10.0, 10.0)),
                attribute_type="float3",
            )
        return cubes.node

    rep.randomizer.register(change_colors)

    # Setup randomization of colors, different each frame
    with rep.trigger.on_frame(num_frames=10):
        rep.randomizer.change_colors()

    # (optional) Write output images to disk
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir="~/replicator_examples/box_scratches",
        rgb=True,
        bounding_box_2d_tight=True,
        semantic_segmentation=True,
        distance_to_image_plane=True,
    )
    writer.attach([render_product])

    carb.log_info("scratches randomization complete")
