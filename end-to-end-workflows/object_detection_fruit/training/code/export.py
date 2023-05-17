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

import os
import torch
import torchvision
from optparse import OptionParser


def parse_input():
    usage = "usage: export.py [options] arg1 "
    parser = OptionParser(usage)
    parser.add_option(
        "-d",
        "--pytorch_dir",
        dest="pytorch_dir",
        help="Location of output PyTorch model",
    )
    parser.add_option(
        "-o",
        "--output_dir",
        dest="output_dir",
        help="Export and save ONNX model to this path",
    )
    (options, args) = parser.parse_args()
    return options, args


def main():
    torch.manual_seed(0)
    options, args = parse_input()
    model = torch.load(options.pytorch_dir)
    model.eval()
    OUTPUT_DIR = options.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT", num_classes=91
    )
    model.eval()

    dummy_input = torch.rand(1, 3, 1024, 1024)

    torch.onnx.export(
        model,
        dummy_input,
        os.path.join(OUTPUT_DIR, "model.onnx"),
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
    )


if __name__ == "__main__":
    main()
