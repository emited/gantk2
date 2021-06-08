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


import math

from neural_tangents import stax


activations = ['relu', 'gelu', 'erf', 'rbf', 'arbf', 'sigmoid_like', 'sin', 'leaky_relu', 'abs', 'identity']


def kernel_ntk(kernel, x, y):
    return kernel(x, y, 'ntk')


def kernel_nngp(kernel, x, y):
    return kernel(x, y, 'nngp')


def activation_factory(activation, params):
    assert activation in activations
    if activation == 'gelu':
        return stax.Gelu()
    elif activation == 'erf':
        return stax.Erf(a=params.act_a, b=params.act_b, c=params.act_c)
    elif activation == 'rbf':
        return stax.Rbf(gamma=params.act_gamma)
    elif activation == 'arbf':
        return stax.Sin(a=math.sqrt(2) * params.act_a, b=math.sqrt(2 * params.act_gamma), c=math.pi/4)
    elif activation == 'sigmoid_like':
        return stax.Sigmoid_like()
    elif activation == 'sin':
        return stax.Sin(a=params.act_a, b=params.act_b, c=params.act_c)
    elif activation == 'relu':
        return stax.Relu(do_stabilize=params.act_do_stabilize)
    elif activation == 'leaky_relu':
        return stax.LeakyRelu(alpha=params.act_alpha, do_stabilize=params.act_do_stabilize)
    elif activation == 'abs':
        return stax.Abs(do_stabilize=params.act_do_stabilize)
    elif activation == 'identity':
        return stax.Identity()
    else:
        raise ValueError(f'No activation named `{activation}`')
