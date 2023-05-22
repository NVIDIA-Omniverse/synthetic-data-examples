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

import omni.graph.core as og
import omni.replicator.core as rep
from omni.usd._impl.utils import get_prim_at_path
from pxr import Semantics
from semantics.schema.editor import remove_prim_semantics

# Setup simple scene
with rep.new_layer():
    # Simple scene setup
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


# Get prims to remove semantics on - Execute this first by itself
my_spheres = rep.get.prims(semantics=[("class", "sphere")])

og.Controller.evaluate_sync()  # Trigger an OmniGraph evaluation of the graph to set the values
get_targets = rep.utils.get_node_targets(my_spheres.node, "outputs_prims")
print(get_targets)
# [Sdf.Path('/Replicator/Sphere_Xform'), Sdf.Path('/Replicator/Sphere_Xform_01'), Sdf.Path('/Replicator/Sphere_Xform_02'), Sdf.Path('/Replicator/Sphere_Xform_03'), Sdf.Path('/Replicator/Sphere_Xform_04'), Sdf.Path('/Replicator/Sphere_Xform_05')]

# Loop through each prim_path and remove all semantic data
for prim_path in get_targets:
    prim = get_prim_at_path(prim_path)
    # print(prim.HasAPI(Semantics.SemanticsAPI))
    result = remove_prim_semantics(prim)  # To remove all semantics
    # result = remove_prim_semantics(prim, label_type='class') # To remove only 'class' semantics
    print(result)
