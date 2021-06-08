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

import neural_tangents.stax as stax

from gantk2.data.transforms import transform_factory
from gantk2.models.antisymmetric import antisymmetric
from gantk2.models.conv import dcgan
from gantk2.models.mlp import mlp, resnet
from gantk2.models.utils import activation_factory, kernel_nngp, kernel_ntk


models = ['mlp', 'dcgan', 'resnet']


parameterizations = ['ntk', 'standard']


def kernel_transform(transform, kernel_fn):
    transform = transform_factory(transform)
    return lambda x, y: kernel_fn(transform.apply(x), transform.apply(y))


def model_preprocessing(model, transform, nngp):
    init_fn, apply_fn, kernel_fn = model
    if nngp:
        kernel_fn = functools.partial(kernel_nngp, kernel_fn)
    else:
        kernel_fn = functools.partial(kernel_ntk, kernel_fn)
    kernel_fn = kernel_transform(transform, kernel_fn)
    return init_fn, apply_fn, kernel_fn


def model_factory(config, generator):
    prefix = 'gen_' if generator else 'discr_'
    def config_attr(attr): return getattr(config, prefix + attr)

    model_type = config_attr('model')
    assert model_type in models
    model = None
    out_size = config.data_dim[-1] if generator else 1
    if model_type == 'mlp':
        assert not generator or len(config.data_dim) == 1
        model = mlp(generator, out_size, config_attr('hidden'), config_attr('layers'), config_attr('activations'),
                    config, config_attr('layer_norm'), config_attr('W_std'), config_attr('b_std'),
                    config_attr('parameterization'))
    elif model_type == 'dcgan':
        model = dcgan(generator, out_size, config_attr('hidden'), config_attr('layers'), config_attr('activations'),
                      config, config_attr('layer_norm'), config_attr('W_std'), config_attr('b_std'),
                      config_attr('parameterization'))
    elif model_type == 'resnet':
        model = resnet(generator, out_size, config_attr('hidden'), config_attr('layers'), config_attr('activations'),
                       config, config_attr('layer_norm'), config_attr('W_std'), config_attr('b_std'),
                       config_attr('parameterization'), config_attr('block_depth'))
    else:
        raise ValueError(f'No model named {model}')

    if generator and config.gen_out_act is not None:
        model = stax.serial(model, activation_factory(config.gen_out_act, config))
    if not generator and config.discr_antisymmetric:
        model = antisymmetric(model)

    return model_preprocessing(model, config.kernel_transform, config.nngp)
