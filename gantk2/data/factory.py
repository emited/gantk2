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

from torch.utils.data import Dataset, Subset

from gantk2.data.transforms import transform_factory


datasets = ['simple', 'mnist', 'celeba']


class IndicesDataset(Dataset):
    def __init__(self, dataset):
        super(IndicesDataset).__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index], index


def source_collate_fn(collate_fn, batch):
    indices = torch.tensor([batch[i][1] for i in range(len(batch))])
    return collate_fn([batch[i][0] for i in range(len(batch))]), indices


def dataset_dimensionality(dataset):
    assert dataset in datasets
    if dataset == 'mnist':
        return [32, 32, 1]
    elif dataset == 'celeba':
        return [32, 32, 3]
    else:
        raise ValueError(f'No defined dimensionality for `{dataset}`')


def dataset_factory(config, source):
    if source:
        dataset_name = config.in_dataset
        opposite_dataset = config.out_dataset
        transform = transform_factory(config.in_transform)
    else:
        dataset_name = config.out_dataset
        opposite_dataset = config.in_dataset
        transform = transform_factory(config.out_transform)

    assert dataset_name in datasets
    if dataset_name == 'simple':
        if opposite_dataset != 'simple' and not config.generator:
            config.data_dim = dataset_dimensionality(opposite_dataset)
        from gantk2.data.simple import get_dataset
        dataset, collate_fn = get_dataset(config, transform, source)
    else:
        assert not source or not config.generator
        config.data_dim = dataset_dimensionality(dataset_name)
        if dataset_name == 'mnist':
            from gantk2.data.mnist import get_dataset
            dataset, collate_fn = get_dataset(config.data_path, transform, True)
        elif dataset_name == 'celeba':
            from gantk2.data.celeba import get_dataset
            dataset, collate_fn = get_dataset(config.data_path, transform, True)
        else:
            raise ValueError(f'No dataset named `{dataset_name}`')
        if source:
            nb_samples = config.in_nb_samples
        else:
            nb_samples = config.out_nb_samples
        dataset_size = len(dataset)
        assert nb_samples <= dataset_size
        if nb_samples > 0:
            chosen_indices = np.arange(dataset_size)
            np.random.shuffle(chosen_indices)
            dataset = Subset(dataset, chosen_indices[:nb_samples])

    if source and not config.generator:
        if collate_fn is not None:
            collate_fn = functools.partial(source_collate_fn, collate_fn)
        return IndicesDataset(dataset), collate_fn
    else:
        return dataset, collate_fn
