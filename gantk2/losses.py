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


import jax

import jax.nn as nn
import jax.numpy as jnp

from functools import partial
from neural_tangents import predict

from gantk2.utils.math import univariate_solution_vanilla_gan


inf_discr_losses = ['inf_ipm', 'inf_ipm_external', 'inf_lsgan', 'inf_vanilla', 'inf_vanilla_ode']
losses = inf_discr_losses + ['vanilla', 'hinge', 'lsgan', 'ipm']


def inf_ipm_external_particle_loss(t, fake_particles, fake, real, discr_kernel_fn):
    sol = - t * discr_kernel_fn(fake_particles, real).mean(1)
    return sol.mean(), sol


def inf_ipm_particle_loss(t, fake_particles, fake, real, discr_kernel_fn):
    score = discr_kernel_fn(fake_particles, fake).mean(1) - discr_kernel_fn(fake_particles, real).mean(1)
    score = t * score
    return score.mean(), score


def inf_ipm_gen_loss(t, z, fake, real, discr_kernel_fn, gen_params, gen_apply_fn):
    return inf_ipm_particle_loss(t, gen_apply_fn(gen_params, z), fake, real, discr_kernel_fn)


def inf_lsgan_particle_loss(t, fake_particles, fake, real, discr_kernel_fn):
    dtype = real.dtype
    train = jnp.concatenate([fake, real])
    k_train_train = discr_kernel_fn(train, None)
    y_train = jnp.concatenate([-jnp.ones(fake.shape[0], dtype=dtype), jnp.ones(real.shape[0], dtype=dtype)])
    predict_fn = predict.gradient_descent_mse(k_train_train, y_train, trace_axes=())
    k_test_train = discr_kernel_fn(fake_particles, train)
    sol = (predict_fn(4 * t, jnp.zeros(y_train.shape[0]), jnp.zeros(fake_particles.shape[0]),
                      k_test_train)[1] ) ** 2.
    return sol.mean(), sol


def inf_lsgan_gen_loss(t, z, fake, real, discr_kernel_fn, gen_params, gen_apply_fn):
    return inf_lsgan_particle_loss(t, gen_apply_fn(gen_params, z), fake, real, discr_kernel_fn)


def vanilla_gan_inf_particle_loss(t, fake_particles, fake, real, discr_kernel_fn):
    train = jnp.concatenate([fake, real])
    y_train = jnp.concatenate([-jnp.ones(fake.shape[0]), jnp.ones(real.shape[0])])
    k_train_train = discr_kernel_fn(train, None) / len(train)
    k_test_train = discr_kernel_fn(fake_particles, train)
    solution_inv, = predict._get_fns_in_eigenbasis(k_train_train, 0, False,
                                                   [lambda z, t: univariate_solution_vanilla_gan(z, t) / z])
    solution = jnp.matmul(k_test_train, solution_inv(y_train, t).squeeze(0)) / len(train)
    log_sol = -nn.log_sigmoid(solution)
    return log_sol.mean(), log_sol


def vanilla_gan_inf_gen_loss(t, z, fake, real, discr_kernel_fn, gen_params, gen_apply_fn):
    return vanilla_gan_inf_particle_loss(t, gen_apply_fn(gen_params, z), fake, real, discr_kernel_fn)


def vanilla_gan_inf_ode_particle_loss(t, momentum, fake_particles, fake, real, discr_kernel_fn):
    train = jnp.concatenate([fake, real])
    k_train_train = discr_kernel_fn(train, None)
    y_train = jnp.concatenate([-jnp.ones(fake.shape[0]), jnp.ones(real.shape[0])])
    predict_fn = predict.gradient_descent(lambda f, y: -(nn.log_sigmoid(y * f)).mean(), k_train_train, y_train,
                                          momentum=momentum, trace_axes=())
    k_test_train = discr_kernel_fn(fake_particles, train)
    sol = predict_fn(t, jnp.zeros(y_train.shape[0]), jnp.zeros(fake_particles.shape[0]), k_test_train)[1]
    nlog_sol = -nn.log_sigmoid(sol)
    return nlog_sol.mean(), nlog_sol


def vanilla_gan_inf_ode_gen_loss(t, momentum, z, fake, real, discr_kernel_fn, gen_params, gen_apply_fn):
    return vanilla_gan_inf_ode_particle_loss(t, momentum, gen_apply_fn(gen_params, z), fake, real, discr_kernel_fn)


def vanilla_gan_discr_loss(fake, real, discr_params, discr_apply_fn):
    fake_logits = discr_apply_fn(discr_params, fake)
    real_logits = discr_apply_fn(discr_params, real)
    return -nn.log_sigmoid(real_logits).mean() - nn.log_sigmoid(-fake_logits).mean()


def vanilla_gan_particle_loss(fake, discr_params, discr_apply_fn):
    fake_logits = discr_apply_fn(discr_params, fake)
    nlogsol = -nn.log_sigmoid(fake_logits)
    return nlogsol.mean(), nlogsol.squeeze(1)


def vanilla_gan_gen_loss(z, discr_params, discr_apply_fn, gen_params, gen_apply_fn):
    return vanilla_gan_particle_loss(gen_apply_fn(gen_params, z), discr_params, discr_apply_fn)


def ipm_discr_loss(fake, real, discr_params, discr_apply_fn):
    fake_score = discr_apply_fn(discr_params, fake)
    real_score = discr_apply_fn(discr_params, real)
    return fake_score.mean() - real_score.mean()


def ipm_particle_loss(fake, discr_params, discr_apply_fn):
    fake_score = discr_apply_fn(discr_params, fake).mean(1)
    return -fake_score.mean(), -fake_score


def ipm_gen_loss(z, discr_params, discr_apply_fn, gen_params, gen_apply_fn):
    return ipm_particle_loss(gen_apply_fn(gen_params, z), discr_params, discr_apply_fn)


def hinge_gan_discr_loss(fake, real, discr_params, discr_apply_fn):
    fake_logits = discr_apply_fn(discr_params, fake)
    real_logits = discr_apply_fn(discr_params, real)
    return -jnp.minimum(0., real_logits - 1.).mean() - jnp.minimum(0., -fake_logits - 1.).mean()


def hinge_gan_particle_loss(fake, discr_params, discr_apply_fn):
    fake_logits = discr_apply_fn(discr_params, fake)
    return -fake_logits.mean(), fake_logits.squeeze(1)


def hinge_gan_gen_loss(z, discr_params, discr_apply_fn, gen_params, gen_apply_fn):
    return hinge_gan_particle_loss(gen_apply_fn(gen_params, z), discr_params, discr_apply_fn)


def lsgan_discr_loss(fake, real, discr_params, discr_apply_fn):
    fake_logits = discr_apply_fn(discr_params, fake)
    real_logits = discr_apply_fn(discr_params, real)
    return ((fake_logits + 1.) ** 2.).mean() + ((real_logits - 1.) ** 2.).mean()


def lsgan_particle_loss(fake, discr_params, discr_apply_fn):
    fake_logits_squared = discr_apply_fn(discr_params, fake) ** 2.
    return (fake_logits_squared).mean(), fake_logits_squared.squeeze(1)


def lsgan_gen_loss(z, discr_params, discr_apply_fn, gen_params, gen_apply_fn):
    return lsgan_particle_loss(gen_apply_fn(gen_params, z), discr_params, discr_apply_fn)


def grad_discr_loss_factory(loss):
    if loss in inf_discr_losses:
        return None
    else:
        if loss == 'vanilla':
            gan_discr_loss_grad = jax.grad(vanilla_gan_discr_loss, argnums=2)
        elif loss == 'hinge':
            gan_discr_loss_grad = jax.grad(hinge_gan_discr_loss, argnums=2)
        elif loss == 'lsgan':
            gan_discr_loss_grad = jax.grad(lsgan_discr_loss, argnums=2)
        elif loss == 'ipm':
            gan_discr_loss_grad = jax.grad(ipm_discr_loss, argnums=2)
        else:
            raise ValueError(f'No discriminator loss named `{loss}`')
        return jax.pmap(gan_discr_loss_grad, in_axes=(0, 0, None, None), static_broadcasted_argnums=3)


def grad_particle_loss_factory(loss, integration_time, momentum):
    if loss == 'inf_ipm':
        gan_particle_loss_grad = partial(inf_ipm_particle_loss, integration_time)
    elif loss == 'inf_ipm_external':
        gan_particle_loss_grad = partial(inf_ipm_external_particle_loss, integration_time)
    elif loss == 'inf_lsgan':
        gan_particle_loss_grad = partial(inf_lsgan_particle_loss, integration_time)
    elif loss == 'inf_vanilla':
        gan_particle_loss_grad = partial(vanilla_gan_inf_particle_loss, integration_time)
    elif loss == 'inf_vanilla_ode':
        gan_particle_loss_grad = partial(vanilla_gan_inf_ode_particle_loss, integration_time, momentum)
    elif loss == 'vanilla':
        gan_particle_loss_grad = vanilla_gan_particle_loss
    elif loss == 'ipm':
        gan_particle_loss_grad = ipm_particle_loss
    elif loss == 'hinge':
        gan_particle_loss_grad = hinge_gan_particle_loss
    elif loss == 'lsgan':
        gan_particle_loss_grad = lsgan_particle_loss
    else:
        raise ValueError(f'No generator loss named `{loss}`')
    gan_particle_loss_grad = jax.value_and_grad(gan_particle_loss_grad, argnums=0, has_aux=True)
    if loss in inf_discr_losses:
        return jax.pmap(gan_particle_loss_grad, in_axes=(0, None, None, None), static_broadcasted_argnums=3)
    else:
        return jax.pmap(gan_particle_loss_grad, in_axes=(0, None, None), static_broadcasted_argnums=2)


def grad_gen_loss_factory(loss, integration_time, momentum):
    if loss in inf_discr_losses:
        if loss == 'inf_ipm':
            gan_gen_loss_grad = partial(inf_ipm_gen_loss, integration_time)
        elif loss == 'inf_lsgan':
            gan_gen_loss_grad = partial(inf_lsgan_gen_loss, integration_time)
        elif loss == 'inf_vanilla':
            gan_gen_loss_grad = partial(vanilla_gan_inf_gen_loss, integration_time)
        elif loss == 'inf_vanilla_ode':
            gan_gen_loss_grad = partial(vanilla_gan_inf_ode_gen_loss, integration_time, momentum)
        else:
            raise ValueError(f'No generator loss named `{loss}`')
        gan_gen_loss_grad = jax.value_and_grad(gan_gen_loss_grad, argnums=4, has_aux=True)
        return jax.pmap(gan_gen_loss_grad, in_axes=(0, None, None, None, None, None),
                        static_broadcasted_argnums=(3, 5))
    else:
        if loss == 'vanilla':
            gan_gen_loss_grad = vanilla_gan_gen_loss
        elif loss == 'hinge':
            gan_gen_loss_grad = hinge_gan_gen_loss
        elif loss == 'lsgan':
            gan_gen_loss_grad = lsgan_gen_loss
        elif loss == 'ipm':
            gan_gen_loss_grad = ipm_gen_loss
        else:
            raise ValueError(f'No generator loss named `{loss}`')
        gan_gen_loss_grad = jax.value_and_grad(gan_gen_loss_grad, argnums=3, has_aux=True)
        return jax.pmap(gan_gen_loss_grad, in_axes=(0, None, None, None, None), static_broadcasted_argnums=(2, 4))
