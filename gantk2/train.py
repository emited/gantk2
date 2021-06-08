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

import jax.numpy as jnp
import neural_tangents as nt

from jax import random
from tqdm import trange

import gantk2.args.args as args
from gantk2.losses import inf_discr_losses, grad_discr_loss_factory, grad_gen_loss_factory, grad_particle_loss_factory
from gantk2.utils import utils
from gantk2.utils.eval import evaluate
from gantk2.utils.setup import setup, load_dataset, load_model



def run_descent(opt, in_dataset, in_loader, z_in_loader, in_collate_fn, out_dataset, out_loader, out_collate_fn,
                discr_model, discr_opt, discr_params, gen_model, gen_opt, gen_params, jax_rng_key_discr,
                jax_rng_key_gen, chkpt_save_dir, img_save_dir):
    assert opt.nb_iter % opt.log_steps == 0

    discr_init_fn, discr_apply_fn, discr_kernel_fn = discr_model
    batched_discr_kernel_fn = nt.batch(discr_kernel_fn)
    gen_init_fn, gen_apply_fn, _ = gen_model

    discr_opt_init_fn, discr_opt_update_fn, discr_opt_get_params = discr_opt
    gen_opt_init_fn, gen_opt_update_fn, gen_opt_get_params = gen_opt
    discr_opt_state = None
    gen_opt_state = None

    if not opt.generator:
        # Particles
        x, _ = utils.dataset_to_array(in_dataset, in_collate_fn)
    else:
        x = None
    if opt.save_target:
        y = utils.dataset_to_array(out_dataset, out_collate_fn)
        utils.save_array(y, chkpt_save_dir, None, 'y')

    discr_loss_grad = grad_discr_loss_factory(opt.loss)
    particles_grad = grad_particle_loss_factory(opt.loss, opt.discr_integration_time, opt.discr_momentum)
    if opt.generator:
        gen_particles_grad = grad_gen_loss_factory(opt.loss, opt.discr_integration_time, opt.discr_momentum)
    else:
        gen_particles_grad = particles_grad

    n_gpu = jax.local_device_count()

    # Training loop
    training_range = trange(opt.nb_iter, dynamic_ncols=True, disable=None)
    keep_state = True
    try:
        loss_gen = -1
        for step in training_range:

            # Batch and initialization
            if not opt.generator:
                _, x_indices = next(in_loader)
                x_batch_discr = x[x_indices]
                z_batch_gen = None
            else:
                z_batch_discr = next(in_loader)
                z_batch_gen = next(z_in_loader)
                if gen_opt_state is None:
                    if gen_params is None:
                        _, gen_params = gen_init_fn(jax_rng_key_gen, z_batch_discr.shape)
                    gen_opt_state = gen_opt_init_fn(gen_params)
                else:
                    gen_params = gen_opt_get_params(gen_opt_state)
                x_batch_discr = gen_apply_fn(gen_params, z_batch_discr)

            if opt.loss not in inf_discr_losses and discr_opt_state is None:
                if discr_params is None:
                    _, jax_rng_key_discr = random.split(jax_rng_key_discr)
                    _, discr_params = discr_init_fn(jax_rng_key_discr, x_batch_discr.shape)
                    keep_state = True
                discr_opt_state = discr_opt_init_fn(discr_params)

            y_batch = next(out_loader)

            if step % opt.log_steps == 0:

                metrics = evaluate(opt, particles_grad,
                                   x=x_batch_discr, y=y_batch,
                                   step=step, loss_gen=loss_gen,
                                   discr_params=None if opt.loss in inf_discr_losses
                                                else discr_opt_get_params(discr_opt_state),
                                   discr_apply_fn=discr_apply_fn,
                                   discr_kernel_fn=batched_discr_kernel_fn,
                                   img_save_dir=img_save_dir,
                                   chkpt_save_dir=chkpt_save_dir,
                )
                training_range.set_postfix({k: v for k, v in metrics.items() if k != 'step'})

            if step % opt.save_steps == 0:
                if not opt.generator:
                    utils.save_array(x, chkpt_save_dir, step, 'x')

            # Resetting Discriminator
            if opt.reset_discr and opt.loss not in inf_discr_losses:
                if not keep_state:
                    _, jax_rng_key_discr = random.split(jax_rng_key_discr)
                    _, discr_params = discr_init_fn(jax_rng_key_discr, x_batch_discr.shape)
                    discr_opt_state = discr_opt_init_fn(discr_params)
                else:
                    keep_state = False

            # Discriminator updates
            for discr_step in range((opt.loss not in inf_discr_losses) * opt.nb_d_steps):
                x_batch_discr_distributed = jnp.stack(jnp.split(x_batch_discr, n_gpu))
                y_batch_distributed = jnp.stack(jnp.split(y_batch, n_gpu))
                grad_discr = discr_loss_grad(x_batch_discr_distributed, y_batch_distributed,
                                             discr_opt_get_params(discr_opt_state), discr_apply_fn)
                grad_discr = utils.average_grads_params(grad_discr)
                discr_opt_state = discr_opt_update_fn(step * opt.nb_d_steps + discr_step, grad_discr, discr_opt_state)

            # Generator / particle update
            if opt.generator:
                z_batch_gen_distributed = jnp.stack(jnp.split(z_batch_gen, n_gpu))
                if opt.loss in inf_discr_losses:
                    loss_gen, grad_gen = gen_particles_grad(z_batch_gen_distributed, x_batch_discr, y_batch,
                                                            discr_kernel_fn,
                                                            gen_opt_get_params(gen_opt_state), gen_apply_fn)
                else:
                    loss_gen, grad_gen = gen_particles_grad(z_batch_gen_distributed,
                                                            discr_opt_get_params(discr_opt_state),
                                                            discr_apply_fn, gen_opt_get_params(gen_opt_state),
                                                            gen_apply_fn)
                grad_gen = utils.average_grads_params(grad_gen)
                gen_opt_state = gen_opt_update_fn(step, grad_gen, gen_opt_state)
            else:
                x_distributed = jnp.stack(jnp.split(x, n_gpu))
                if opt.loss in inf_discr_losses:
                    loss_gen, grad_particles = gen_particles_grad(x_distributed, x_batch_discr, y_batch,
                                                                  discr_kernel_fn)
                else:
                    loss_gen, grad_particles = gen_particles_grad(x_distributed, discr_opt_get_params(discr_opt_state),
                                                                  discr_apply_fn)
                x = x - opt.eta * jnp.concatenate(grad_particles)

            if isinstance(loss_gen, tuple):
                # evacuate solution and keep loss
                loss_gen, _ = loss_gen
                loss_gen = loss_gen.mean().item()
                metrics['lg'] = loss_gen
                training_range.set_postfix({k: v for k, v in metrics.items() if k != 'step'})

        status_code = 0

    except KeyboardInterrupt:
        status_code = 130

    evaluate(opt, particles_grad,
             x=x_batch_discr, y=y_batch,
             step=step, loss_gen=loss_gen,
             discr_params=None if opt.loss in inf_discr_losses else discr_opt_get_params(discr_opt_state),
             discr_apply_fn=discr_apply_fn,
             discr_kernel_fn=batched_discr_kernel_fn,
             img_save_dir=img_save_dir,
             chkpt_save_dir=chkpt_save_dir,
    )
    if not opt.generator:
        utils.save_array(x, chkpt_save_dir, step, 'x')

    return status_code


if __name__ == '__main__':
    # Parse arguments
    p = args.create_setup_args()
    p = args.create_log_args(p)
    p = args.create_model_args(p)
    p = args.create_dataset_args(p)
    p = args.create_plot_args(p)
    p = args.create_training_args(p)
    p = args.create_config_args(p)
    opt = args.parse_args(p)

    # Main
    jax_rng_key_discr, jax_rng_key_gen, img_save_dir, chkpt_save_dir = setup(opt)

    print(f'Save dir: {opt.save_dir}')

    (in_loader, in_dataset, in_collate_fn), z_in_loader, (out_loader, out_dataset, out_collate_fn) = load_dataset(opt)

    (discr_model, discr_params, discr_opt), (gen_model, gen_params, gen_opt) = load_model(opt)

    status_code = run_descent(opt, in_dataset, in_loader, z_in_loader, in_collate_fn, out_dataset, out_loader,
                              out_collate_fn, discr_model, discr_opt, discr_params, gen_model, gen_opt, gen_params,
                              jax_rng_key_discr, jax_rng_key_gen, chkpt_save_dir, img_save_dir)

    print('Done')
    print(status_code)
