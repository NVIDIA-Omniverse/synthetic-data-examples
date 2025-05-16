# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

from semantics.schema.editor import add_prim_semantics, LabelWriteType
import omni
from pxr import Gf, UsdGeom
import asyncio
import omni.replicator.core as rep
import numpy as np
import random
import json
from typing import Optional
import omni.timeline

#### SETUP REQUIRED!
# 1. You need to enable the "Video Encoding Extension"
# 2. Search for "omni.videoencoding" in the Extensions tab of your KIT app, and enable it.
# 3. Set the output_path below to a local directory
start_frame = 0
end_frame = 121
output_path = "C:/your_output_folder"  # See setup item 1 & 2 as well


def set_posrot(mesh_prim, prim_pos, prim_rot):
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xformable(mesh_prim)
    xform_ops = {op.GetBaseName(): op for op in xform.GetOrderedXformOps()}

    for op in xform.GetOrderedXformOps():
        if "translate" in op.GetName():
            translate_op = xform_ops["translate"]
            translate_op.Set(Gf.Vec3d(prim_pos[0], prim_pos[1], prim_pos[2]))
        elif "rotate" in op.GetName():
            rotate_op = xform_ops["rotateXYZ"]
            rotate_op.Set(Gf.Vec3d(prim_rot[0], prim_rot[1], prim_rot[2]))


def interpolate(v0, t0, v1, t1, t):
    u = (t - t0) / (t1 - t0)
    u = np.clip(u, 0.0, 1.0)

    v0_arr = np.array(v0, dtype=float)
    v1_arr = np.array(v1, dtype=float)
    return (1 - u) * v0_arr + u * v1_arr


async def run():

    # Get the stage
    stage = omni.usd.get_context().get_stage()

    # Set timeline
    print("Capturing...")
    timeline = omni.timeline.get_timeline_interface()
    timeline.set_end_time(end_frame)
    timeline.set_time_codes_per_second(60)
    timeline.commit()

    # Forklift Animation Transform settings
    forklift_start = (0.63876, 5.33176, 0)
    forklift_end = (-1.98992, 4.52809, 0)
    forklift_rot = (0, 0, -73)

    # Set up render products and annotators
    # These are the resolutions for Cosmos Transfer1 inputs
    render_product = rep.create.render_product("/Root/Worker/Camera", (1280, 704))

    # Initialize and attach writer
    backend = rep.backends.get("DiskBackend")
    backend.initialize(output_dir=output_path)
    cosmos_writer = rep.writers.get("CosmosWriter")
    cosmos_writer.initialize(backend=backend, use_instance_id=False)
    cosmos_writer.attach(render_product)

    # Set up the scene ONCE
    forklift = stage.GetPrimAtPath("/Root/forklift")
    index = 0
    timeline.set_current_time(start_frame / timeline.get_time_codes_per_seconds())
    timeline.commit()

    # Iterate through the frames, moving the forklift
    for frame_id in range(end_frame):
        # Move Forklift
        newpos = interpolate(
            forklift_start, start_frame, forklift_end, end_frame, index
        )
        set_posrot(forklift, newpos, forklift_rot)
        index += 1

        await rep.orchestrator.step_async()
    rep.orchestrator.stop()


asyncio.ensure_future(run())
