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

from neural_tangents.utils.typing import Layer


def antisymmetric(model: Layer) -> Layer:
    base_init_fn, base_apply_fn, kernel_fn = model

    def init_fn(rng, input_shape):
        init = type(input_shape)(base_init_fn(rng, input_shape))
        return init[0], (init[1], init[1])

    def apply_fn(params, inputs, **kwargs):
        return (base_apply_fn(params[0], inputs, **kwargs) - base_apply_fn(params[1], inputs, **kwargs)) / math.sqrt(2)

    return init_fn, apply_fn, kernel_fn
