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


from neural_tangents import stax

from gantk2.models.utils import activation_factory


def dcgan_gen_block(last_block, out_size, activations, act_params, layer_norm, W_std, b_std, parameterization):
    block = [stax.LayerNorm()] if layer_norm else []
    block += [
        stax.FanOut(len(activations)),
        stax.parallel(
            *[stax.serial(
                activation_factory(activation, act_params),
                stax.ConvTranspose(out_size, (4, 4), strides=(2, 2), padding='SAME', W_std=W_std,
                                   b_std=b_std * (not last_block), parameterization=parameterization)
            ) for activation in activations]
        ),
        stax.FanInSum()
    ]
    return stax.serial(*block)


def dcgan_generator(out_size, hidden_size, activations, act_params, layer_norm, W_std, b_std, parameterization):
    out_dims = [hidden_size // 2, hidden_size // 4, hidden_size // 8, out_size]
    layers = [stax.ConvTranspose(hidden_size, (2, 2), padding='VALID', W_std=W_std, b_std=b_std,
                                 parameterization=parameterization)]
    for i, out_dim in enumerate(out_dims):
        layers.append(dcgan_gen_block(i < len(out_dims) - 1, out_dim, activations, act_params, layer_norm, W_std,
                                      b_std, parameterization))
    return stax.serial(*layers)


def dcgan_discr_block(first_block, last_block, out_size, activations, act_params, layer_norm, W_std, b_std,
                      parameterization):
    if last_block:
        kernel_size = 4
        stride = 1
        padding = 'VALID'
    else:
        kernel_size = 4
        stride = 2
        padding = 'SAME'
    block = [stax.LayerNorm()] if layer_norm and not first_block else []
    block += [
        stax.FanOut(len(activations)),
        stax.parallel(
            *[stax.serial(
                activation_factory(activation, act_params),
                stax.Conv(out_size, (kernel_size, kernel_size), strides=(stride, stride), padding=padding, W_std=W_std,
                          b_std=b_std, parameterization=parameterization)
            ) for activation in activations]
        ),
        stax.FanInSum()
    ]
    return stax.serial(*block)


def dcgan_discriminator(hidden_size, activations, act_params, layer_norm, W_std, b_std, parameterization):
    in_dim = hidden_size // 8
    out_dims = [hidden_size // 4, hidden_size // 2, hidden_size, 1]
    layers = [stax.Conv(in_dim, (3, 3), padding='SAME', W_std=W_std, b_std=0, parameterization=parameterization)]
    for i, out_dim in enumerate(out_dims):
        layers.append(dcgan_discr_block(i == 0, i < len(out_dims) - 1, out_dim, activations, act_params, layer_norm,
                                        W_std, b_std * (i < len(out_dims) - 1), parameterization))
    layers.append(stax.Flatten())
    return stax.serial(*layers)


def dcgan(generator, out_size, hidden_size, n_hidden_layers, activations, act_params, layer_norm, W_std, b_std,
          parameterization):
    assert hidden_size // 8 > 0
    if generator:
        return dcgan_generator(out_size, hidden_size, activations, act_params, layer_norm, W_std, b_std,
                               parameterization)
    else:
        assert out_size == 1
        return dcgan_discriminator(hidden_size, activations, act_params, layer_norm, W_std, b_std, parameterization)
