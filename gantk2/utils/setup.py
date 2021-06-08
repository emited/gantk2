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
import jax
import random
import torch

import torch.autograd

import numpy as np

from jax.experimental.optimizers import adam, sgd
from torch.utils.data import DataLoader, IterableDataset

import gantk2.data.factory as data
import gantk2.models.factory as models
from gantk2.losses import inf_discr_losses

from gantk2.utils import utils


optimizers = ['sgd', 'adam']


def setup(opt):
    # Device handling (CPU, GPU, multi GPU)
    if opt.device is None:
        os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
        jax.config.update('jax_platform_name', 'cpu')
        print('Learning on CPU')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, opt.device))
        n_gpu = len(opt.device)
        print(f'Learning on {n_gpu} GPU(s) ')
    torch.autograd.set_grad_enabled(False)

    # Seed
    if opt.seed is None:
        opt.seed = random.randint(1, 2**32 - 1)
    else:
        assert isinstance(opt.seed, int) and opt.seed > 0
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    jax_rng_key_discr, jax_rng_key_gen = jax.random.split(jax.random.PRNGKey(opt.seed))
    print(f'Seed: {opt.seed}')

    # Logs Init
    if hasattr(opt, 'save_path') and hasattr(opt, 'save_name'):
        save_dir = os.path.join(opt.save_path, opt.save_name)

        # create dirs for training
        if getattr(opt, 'log_steps', None) is not None:
            img_save_dir, chkpt_save_dir = utils.create_dir(save_dir, opt.erase)
            opt.save_dir = save_dir
            utils.save_config(opt, opt.save_dir)
        else:
            img_save_dir = save_dir
            chkpt_save_dir = None
    else:
        img_save_dir, chkpt_save_dir = None, None

    return jax_rng_key_discr, jax_rng_key_gen, img_save_dir, chkpt_save_dir


def load_dataset(opt):
    print('Loading output data...')
    out_dataset, out_collate_fn = data.dataset_factory(opt, False)
    out_batch_size = opt.out_batch_size if opt.out_batch_size > 0 else len(out_dataset)
    is_out_finite = not isinstance(out_dataset, IterableDataset)
    out_loader = DataLoader(out_dataset, batch_size=out_batch_size, collate_fn=out_collate_fn, shuffle=is_out_finite,
                            drop_last=is_out_finite)
    out_loader = iter(utils.cycle_iterator(out_loader))

    # Source distribution
    print('Loading input data...')
    assert opt.generator or opt.in_nb_samples > 0
    in_dataset, in_collate_fn = data.dataset_factory(opt, True)
    in_batch_size = opt.in_batch_size if opt.in_batch_size > 0 else len(in_dataset)
    is_in_finite = not isinstance(in_dataset, IterableDataset)
    in_loader = DataLoader(in_dataset, batch_size=in_batch_size, collate_fn=in_collate_fn, shuffle=is_in_finite,
                           drop_last=True)
    in_loader = iter(utils.cycle_iterator(in_loader))
    gen_nb_z = opt.gen_nb_z if opt.gen_nb_z > 0 else len(in_dataset)
    z_in_loader = DataLoader(in_dataset, batch_size=gen_nb_z, collate_fn=in_collate_fn, shuffle=is_in_finite,
                             drop_last=True)
    z_in_loader = iter(utils.cycle_iterator(z_in_loader))
    return (in_loader, in_dataset, in_collate_fn), z_in_loader, (out_loader, out_dataset, out_collate_fn)


def load_model(opt):
    print('Building models...')
    discr_model = models.model_factory(opt, False)
    if opt.loss not in inf_discr_losses:
        if opt.discr_optimizer == 'sgd':
            discr_opt = sgd(opt.discr_eta)
        elif opt.discr_optimizer == 'adam':
            discr_opt = adam(opt.discr_eta, opt.b1, opt.b2)
    else:
        discr_opt = None, None, None
    discr_params = None
    if opt.generator:
        gen_model = models.model_factory(opt, True)
        if opt.gen_optimizer == 'sgd':
            gen_opt = sgd(opt.gen_eta)
        elif opt.gen_optimizer == 'adam':
            gen_opt = adam(opt.gen_eta, opt.b1, opt.b2)
    else:
        gen_model = None, None, None
        gen_opt = None, None, None
    gen_params = None
    return (discr_model, discr_params, discr_opt), (gen_model, gen_params, gen_opt)
