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


from jax import numpy as jnp


transforms = ['identity', 'tanh', 'atanh']


class AbstractTransform(object):
    def __init__(self):
        super(AbstractTransform).__init__()

    def apply(self, x):
        pass


class Identity(AbstractTransform):
    def __init__(self):
        super(AbstractTransform).__init__()

    def apply(self, x):
        return x


class Tanh(AbstractTransform):
    def __init__(self):
        super(AbstractTransform).__init__()

    def apply(self, x):
        return jnp.tanh(x)


class Atanh(AbstractTransform):
    def __init__(self):
        super(AbstractTransform).__init__()

    def apply(self, x):
        return jnp.arctanh(x)


def transform_factory(transform):
    assert transform in transforms
    if transform == 'identity':
        return Identity()
    elif transform == 'tanh':
        return Tanh()
    elif transform == 'atanh':
        return Atanh()
    else:
        raise ValueError(f'No transform named `{transform}`')
