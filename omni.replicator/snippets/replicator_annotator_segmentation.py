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
This is an example of how to view annotator data if needed.
"""
import asyncio
import omni.replicator.core as rep
import omni.syntheticdata as sd


async def test_semantics():
    cone = rep.create.cone(semantics=[("prim", "cone")], position=(100, 0, 0))
    sphere = rep.create.sphere(semantics=[("prim", "sphere")], position=(-100, 0, 0))
    invalid_type = rep.create.cube(semantics=[("shape", "boxy")], position=(0, 100, 0))

    # Setup semantic filter
    # sd.SyntheticData.Get().set_instance_mapping_semantic_filter("prim:*")

    cam = rep.create.camera(position=(500, 500, 500), look_at=(0, 0, 0))
    rp = rep.create.render_product(cam, (1024, 512))

    segmentation = rep.AnnotatorRegistry.get_annotator("semantic_segmentation")
    segmentation.attach(rp)

    # step_async() tells Omniverse to update, otherwise the annoation buffer could be empty
    await rep.orchestrator.step_async()
    data = segmentation.get_data()
    print(data)


# Example Output:
# {
#     "data": array(
#         [
#             [0, 0, 0, ..., 0, 0, 0],
#             [0, 0, 0, ..., 0, 0, 0],
#             [0, 0, 0, ..., 0, 0, 0],
#             ...,
#             [0, 0, 0, ..., 0, 0, 0],
#             [0, 0, 0, ..., 0, 0, 0],
#             [0, 0, 0, ..., 0, 0, 0],
#         ],
#         dtype=uint32,
#     ),
#     "info": {
#         "_uniqueInstanceIDs": array([1, 1, 1], dtype=uint8),
#         "idToLabels": {
#             "0": {"class": "BACKGROUND"},
#             "2": {"prim": "cone"},
#             "3": {"prim": "sphere"},
#             "4": {"shape": "boxy"},
#         },
#     },
# }

asyncio.ensure_future(test_semantics())
