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


import torch

import jax.lax as lax
import jax.nn as nn
import jax.numpy as jnp
import numpy as np

from geomloss import SamplesLoss
from jax import custom_jvp


# Structure inspired by https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/math/special.py
# but with a new implementation.


def lambertw_first_approx_exp(z):
    return nn.softplus(z / jnp.log(1 + 0.5 * nn.softplus(z)))


def lambertw_iteration_exp(args):
    """See https://link.springer.com/article/10.1007/s10444-017-9530-3"""
    unused_should_stop, w, z, tol, iteration_count = args
    w_next = w / (1 + w) * (1 + z - jnp.log(w))
    delta = w_next - w
    converged = jnp.abs(delta) <= tol * jnp.abs(w_next)

    should_stop_next = jnp.all(converged) | (iteration_count >= 100)
    return should_stop_next, w_next, z, tol, iteration_count + 1


@custom_jvp
def lambertw_principal_branch_exp(z):
    """
    Computes W(exp(z))
    """
    np_finfo = jnp.finfo(z.dtype)
    tolerance = (2. * np_finfo.resolution).astype(z.dtype)
    z0 = lambertw_first_approx_exp(z)
    z0 = lax.while_loop(cond_fun=lambda args: ~args[0],
                        body_fun=lambertw_iteration_exp,
                        init_val=(False, z0, z, tolerance, 0))[1]
    return z0.astype(z.dtype)


@lambertw_principal_branch_exp.defjvp
def lambertw_principal_branch_exp_jvp(primals, tangents):
    z, = primals
    z_dot, = tangents
    ans = lambertw_principal_branch_exp(z)
    ans_dot = ans / (1 + ans) * z_dot
    return ans, ans_dot


def univariate_solution_vanilla_gan(z, t):
    return -lambertw_principal_branch_exp(t * z + 1) + t * z + 1


def to_distributed(x, n_gpu):
    return jnp.stack(jnp.split(x, n_gpu))


def from_distributed(x):
    return jnp.concatenate(x, axis=0)


def samples_loss(x, y, loss, *args, **kwargs):
    sinkhorn_fn = SamplesLoss(loss=loss, *args, **kwargs)
    sinkhorn = sinkhorn_fn(torch.tensor(np.copy(x)), torch.tensor(np.copy(y))).item()
    return sinkhorn
