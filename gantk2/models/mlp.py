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


def mlp_block(out_size, activations, act_params, layer_norm, W_std, b_std, parameterization):
    block = [stax.LayerNorm()] if layer_norm else []
    block += [
        stax.FanOut(len(activations)),
        stax.parallel(
            *[stax.serial(
                activation_factory(activation, act_params),
                stax.Dense(out_size, W_std=W_std, b_std=b_std, parameterization=parameterization)
            ) for activation in activations]
        ),
        stax.FanInSum()
    ]
    return stax.serial(*block)


def mlp(generator, out_size, hidden_size, n_hidden_layers, activations, act_params, layer_norm, W_std, b_std,
        parameterization):
    layers = [stax.Flatten(), stax.Dense(hidden_size, W_std=W_std, b_std=b_std, parameterization=parameterization)]
    for i in range(n_hidden_layers):
        # no bias on last layer
        if i == n_hidden_layers - 1:
            b_std = 0
        block_layer_norm = layer_norm and ((generator and i < n_hidden_layers - 1) or (not generator and i > 0))
        layers.append(mlp_block(hidden_size if i < n_hidden_layers - 1 else out_size, activations, act_params,
                                block_layer_norm, W_std, b_std, parameterization))

    return stax.serial(*layers)


def resnet_block(out_size, activations, act_params, layer_norm, W_std, b_std, parameterization, depth):
    assert not layer_norm
    block = []
    for _ in range(depth):
        # if layer_norm and not is_first_layer:
        #     block += [stax.LayerNorm()]
        block += [
            stax.FanOut(len(activations)),
            stax.parallel(
                *[stax.serial(
                    activation_factory(activation, act_params),
                    stax.Dense(out_size, W_std=W_std, b_std=b_std, parameterization=parameterization)
                ) for activation in activations]
            ),
            stax.FanInSum()
        ]
    return stax.serial(*block)


def resnet(generator, out_size, hidden_size, n_hidden_layers, activations, act_params, layer_norm, W_std, b_std,
           parameterization, block_depth):
    assert not generator and not layer_norm
    blocks = [stax.Flatten(), stax.Dense(hidden_size, W_std=W_std, b_std=b_std, parameterization=parameterization)]
    for i in range(n_hidden_layers):
        blocks += [
            stax.FanOut(2),
            stax.parallel(
                stax.Dense(hidden_size, W_std=W_std, b_std=b_std, parameterization=parameterization),
                resnet_block(hidden_size, activations, act_params, layer_norm, W_std, b_std, parameterization, block_depth)
            ),
            stax.FanInSum()
        ]
    blocks += [resnet_block(out_size, activations, act_params, layer_norm, W_std, 0, parameterization, 1)]
    return stax.serial(*blocks)
