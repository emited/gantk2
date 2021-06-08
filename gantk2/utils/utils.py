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


import os
import shutil
import sys

import jax
import yaml

import numpy as np
from jax import tree_map

from torch.utils.data import DataLoader, IterableDataset

from gantk2.utils.math import to_distributed



@jax.jit
def average_grads_params(grads):
    return tree_map(lambda g: g.mean(0), grads)

def cycle_iterator(iterable):
    while True:
        for x in iterable:
            if isinstance(x, tuple) or isinstance(x, list):
                yield [xt.numpy() for xt in x]
            else:
                yield x.numpy()


def dataset_to_array(dataset, collate_fn):
    assert not isinstance(dataset, IterableDataset)
    loader = cycle_iterator(DataLoader(dataset, batch_size=len(dataset), collate_fn=collate_fn, drop_last=False))
    x = next(loader)
    return x


def save_config(config, path):
    with open(os.path.join(path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)


def create_dir(directory, erase):
    if os.path.isdir(directory):
        if not erase and input(f'Experiment directory `{directory}` already exists. Remove? (y|n) ') != 'y':
            sys.exit()
        shutil.rmtree(directory)
    img_save_dir = os.path.join(directory, 'img')
    chkpt_save_dir = os.path.join(directory, 'chkpt')
    os.makedirs(img_save_dir)
    os.makedirs(chkpt_save_dir)
    return img_save_dir, chkpt_save_dir


def save_array(x, directory, step, name):
    np.savez_compressed(os.path.join(directory,
                                     f'{name}_{step:08d}.npz' if step is not None else f'{name}.npz'), name=x)


def append_dict_to_csv(csv_fn, d):
    import csv
    write_header = not os.path.isfile(csv_fn)
    with open(csv_fn, "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sorted(d.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(d)


def discr_train_step(opt, discr_init_fn, discr_apply_fn, discr_opt,
                     discr_loss_grad, jax_rng_key_discr,
                     x, x_batch_discr, y_batch, n_gpu,
                     discr_opt_state=None, step=0,
                     discr_params=None,
                     ):
    discr_opt_init_fn, discr_opt_update_fn, discr_opt_get_params = discr_opt

    if discr_opt_state is None:
        if discr_params is None:
            _, discr_params = discr_init_fn(jax_rng_key_discr, x.shape)
        discr_opt_state = discr_opt_init_fn(discr_params)

        # compute nb_d_steps from integration time
        opt.nb_d_steps = int(opt.discr_integration_time / opt.discr_eta)
        print('nb_d_steps = ', opt.nb_d_steps)

    # Discriminator updates
    for discr_step in range(opt.nb_d_steps):
        grad_discr = discr_loss_grad(to_distributed(x_batch_discr, n_gpu),
                                     to_distributed(y_batch, n_gpu),
                                     discr_opt_get_params(discr_opt_state),
                                     discr_apply_fn)
        grad_discr = average_grads_params(grad_discr)
        discr_opt_state = discr_opt_update_fn(step * opt.nb_d_steps + discr_step, grad_discr, discr_opt_state)

    return discr_opt_state, discr_opt_get_params(discr_opt_state)
