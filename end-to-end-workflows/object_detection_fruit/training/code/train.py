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

from PIL import Image
import os
import numpy as np
import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
import json
import shutil
from optparse import OptionParser
from torch.utils.tensorboard import SummaryWriter


class FruitDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms

        list_ = os.listdir(root)
        for file_ in list_:
            name, ext = os.path.splitext(file_)
            ext = ext[1:]
            if ext == "":
                continue

            if os.path.exists(root + "/" + ext):
                shutil.move(root + "/" + file_, root + "/" + ext + "/" + file_)

            else:
                os.makedirs(root + "/" + ext)
                shutil.move(root + "/" + file_, root + "/" + ext + "/" + file_)

        self.imgs = list(sorted(os.listdir(os.path.join(root, "png"))))
        self.label = list(sorted(os.listdir(os.path.join(root, "json"))))
        self.box = list(sorted(os.listdir(os.path.join(root, "npy"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "png", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        label_path = os.path.join(self.root, "json", self.label[idx])

        with open(os.path.join("root", label_path), "r") as json_data:
            json_labels = json.load(json_data)

        box_path = os.path.join(self.root, "npy", self.box[idx])
        dat = np.load(str(box_path))

        boxes = []
        labels = []
        for i in dat:
            obj_val = i[0]
            xmin = torch.as_tensor(np.min(i[1]), dtype=torch.float32)
            xmax = torch.as_tensor(np.max(i[3]), dtype=torch.float32)
            ymin = torch.as_tensor(np.min(i[2]), dtype=torch.float32)
            ymax = torch.as_tensor(np.max(i[4]), dtype=torch.float32)
            if (ymax > ymin) & (xmax > xmin):
                boxes.append([xmin, ymin, xmax, ymax])
                area = (xmax - xmin) * (ymax - ymin)
            labels += [json_labels.get(str(obj_val)).get("class")]

        label_dict = {}

        static_labels = {
            "apple": 0,
            "avocado": 1,
            "kiwi": 2,
            "lime": 3,
            "lychee": 4,
            "pomegranate": 5,
            "onion": 6,
            "strawberry": 7,
            "lemon": 8,
            "orange": 9,
        }

        labels_out = []

        for i in range(len(labels)):
            label_dict[i] = labels[i]

        for i in label_dict:
            fruit = label_dict[i]
            final_fruit_label = static_labels[fruit]
            labels_out += [final_fruit_label]

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels_out, dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])
        target["area"] = area

        if self.transforms is not None:
            img = self.transforms(img)
        return img, target

    def __len__(self):
        return len(self.imgs)


"""
Parses command line options. Requires input data directory, output torch file, and number epochs used to train.
"""


def parse_input():
    usage = "usage: train.py [options] arg1 arg2 "
    parser = OptionParser(usage)
    parser.add_option(
        "-d",
        "--data_dir",
        dest="data_dir",
        help="Directory location for Omniverse synthetic data.",
    )
    parser.add_option(
        "-o",
        "--output_file",
        dest="output_file",
        help="Save torch model to this file and location (file ending in .pth)",
    )
    parser.add_option(
        "-e",
        "--epochs",
        dest="epochs",
        help="Give number of epochs to be used for training",
    )
    (options, args) = parser.parse_args()
    return options, args


def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    return T.Compose(transforms)


def collate_fn(batch):
    return tuple(zip(*batch))


def create_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def main():
    writer = SummaryWriter()
    options, args = parse_input()
    dataset = FruitDataset(options.data_dir, get_transform(train=True))

    train_size = int(len(dataset) * 0.7)
    valid_size = int(len(dataset) * 0.2)
    test_size = len(dataset) - valid_size - train_size

    train, valid, test = torch.utils.data.random_split(
        dataset, [train_size, valid_size, test_size]
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=collate_fn
    )
    validloader = torch.utils.data.DataLoader(
        valid, batch_size=16, shuffle=True, num_workers=4, collate_fn=collate_fn
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    num_classes = 10
    num_epochs = int(options.epochs)
    model = create_model(num_classes)

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001)
    len_dataloader = len(data_loader)
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        i = 0
        for imgs, annotations in data_loader:
            i += 1
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            loss_dict = model(imgs, annotations)
            losses = sum(loss for loss in loss_dict.values())
            writer.add_scalar("Loss/train", losses, epoch)

            losses.backward()
            optimizer.step()

            print(f"Iteration: {i}/{len_dataloader}, Loss: {losses}")

    writer.close()
    torch.save(model, options.output_file)


if __name__ == "__main__":
    main()
