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
import math
import os
import torch

import numpy as np

from PIL import Image

from torch import distributions as D
from torch.utils.data import IterableDataset, TensorDataset


fixed_datasets = ['two_couples_2d', 'two_singles_1d', 'modes_1d', 'modes_1d_overlap']
sampled_distributions = ['gaussian', 'image']
distributions = sampled_distributions + fixed_datasets


class SampleDataset(IterableDataset):
    def __init__(self, distribution, transform):
        super(SampleDataset).__init__()
        self.distribution = distribution
        self.transform = transform

    def __iter__(self):
        return self

    def __next__(self):
        return self.transform.apply(self.distribution.sample())


def distribution_factory(config, config_attr, source):
    distribution = config_attr('distribution')
    n = config_attr('nb_components')
    if not source or not config.generator:
        dim = config.data_dim
    else:
        dim = [1] * (len(config.data_dim) - 1) + [config.gen_in_dim]
    assert distribution in distributions and n > 0
    mixture_distribution = D.Categorical(torch.ones(n))

    if distribution == 'gaussian':
        mean = torch.tensor(config_attr('loc'))
        std = torch.tensor(config_attr('scale'))
        assert (len(std) == 1 or len(std) == dim[-1]) and (len(mean) == 1 or len(mean) == dim[-1])
        if n == 1:
            component_distribution = D.Normal(mean * torch.ones(dim), std)
        else:
            assert len(dim) == 1 and dim[0] <= 2
            radius = config_attr('mix_dev')
            if dim[0] == 1:
                centers = (torch.arange(n) - (n - 1) / 2).unsqueeze(1)
            else:
                radius = config_attr('mix_dev')
                delta_theta = 2 * math.pi / n
                centers_x = torch.cos(delta_theta * torch.arange(n))
                centers_y = torch.sqrt(1 - centers_x ** 2) * torch.sign(torch.arange(n) - n / 2)
                centers = torch.stack([centers_x, centers_y], dim=1)
            component_distribution = D.Independent(D.Normal(mean + radius * centers, std), 1)

    elif distribution == 'image':
        assert config.generator or (len(dim) == 1 and dim[0] == 2)
        loc = config_attr('loc')
        scale = config_attr('scale')
        assert len(scale) <= 2 and len(loc) <= 2
        if len(scale) == 1:
            scale *= 2
        if len(loc) == 1:
            loc *= 2
        img_path = os.path.join(config.data_path, config_attr('img_name'))
        image = torch.from_numpy(np.array((Image.open(img_path).convert('L'))))  # To greyscale
        h, w = image.size()
        xx = loc[0] - scale[0] * torch.linspace(-1, 1, w)
        yy = loc[1] + scale[1] * torch.linspace(-1, 1, h)
        xx, yy = torch.meshgrid(xx, yy)
        means = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        std = torch.tensor([scale[0] / w, scale[1] / h])
        image = (image.max() - image).T.flip(1).flipud()  # White is zero probability
        probs = image.flatten()
        # Build mixture distribution representing the image
        n = len(probs)
        assert n > 1
        mixture_distribution = D.Categorical(probs)
        component_distribution = D.Independent(D.Normal(means, std), 1)

    else:
        raise ValueError(f'No distribution named `{distribution}`')

    if n == 1:
        return component_distribution
    else:
        return D.MixtureSameFamily(mixture_distribution, component_distribution)


def transform_collate_fn(transform, batch):
    return torch.from_numpy(np.asarray(transform.apply(torch.stack([batch[i] for i in range(len(batch))]))))


def tensor_dataset_collate_fn(batch):
    return torch.stack([batch[i][0] for i in range(len(batch))])


def get_dataset(config, transform, source):
    def config_attr(attr):
        return getattr(config, prefix + attr, None)

    prefix = 'in_' if source else 'out_'
    nb_samples = config_attr('nb_samples')
    assert nb_samples >= 0
    assert ((config_attr('batch_size') > 0 and config.gen_nb_z > 0)
            or config_attr('nb_samples') > 0)

    if config_attr('distribution') in sampled_distributions:
        distribution = distribution_factory(config, config_attr, source)
        if nb_samples == 0:
            return SampleDataset(distribution, transform), functools.partial(transform_collate_fn, transform)
        else:
            transformed_samples = torch.from_numpy(np.asarray(transform.apply(distribution.sample([nb_samples]))))
            return TensorDataset(transformed_samples), tensor_dataset_collate_fn
    elif config_attr('distribution') in fixed_datasets:
        data = create_fixed_dataset(config, config_attr, source)
        transformed_samples = torch.from_numpy(np.asarray(transform.apply(data)))
        return TensorDataset(transformed_samples), tensor_dataset_collate_fn

    raise NotImplementedError(config_attr('distribution'))


def create_fixed_dataset(config, config_attr, source):
    if config_attr('distribution') == 'two_couples_2d':
        if source:
            return np.array([[1., 1], [1, -1.]])
        else:
            return np.array([[-1., -1], [-1, 1]])

    if config_attr('distribution') == 'two_singles_1d':
        if source:
            return np.array([[-1.,],])
        else:
            return np.array([[1.,],])
    if config_attr('distribution') == 'modes_1d':
        if source:
            return np.array([[-1.2,], [-1.34,], [-0.8,],  [-0.7,], [-0.3], ]) * 1.5
        else:
            return np.array([[0.6,], [1.,], [1.23,], [1.32,], [0.2,]]) * 1.5
    if config_attr('distribution') == 'modes_1d_overlap':
        if source:
            return np.array([[-1.2,], [-1.34,], [-0.8,],  [-0.7,], [-0.2], ]) * 1.5
        else:
            return np.array([[0.6,], [1.,], [1.23,], [1.32,], [-1.1,], [-1.,], [-1.8,]]) * 1.5
    raise NotImplementedError(config_attr('distribution'))

