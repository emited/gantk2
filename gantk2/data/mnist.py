# Copyright 2021 Ibrahim Ayed, Emmanuel de Bézenac, Mickaël Chen, Jean-Yves Franceschi, Sylvain Lamprier, Patrick Gallinari

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import functools
import torch

import numpy as np

from torchvision import datasets, transforms


def collate_fn(transform, batch):
    batch = transform.apply(1.999 * (torch.stack([batch[i][0].permute(1, 2, 0) for i in range(len(batch))]).numpy() - 0.5))
    return torch.from_numpy(np.asarray(batch))


def get_dataset(data_path, transform, train):
    dataset_transform = transforms.Compose([transforms.CenterCrop(32), transforms.ToTensor()])
    transform_collate_fn = functools.partial(collate_fn, transform)
    return datasets.MNIST(data_path, train=train, download=True, transform=dataset_transform), transform_collate_fn
