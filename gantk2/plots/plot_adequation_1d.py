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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import neural_tangents as nt

import gantk2.args.args as args
from gantk2.utils.utils import dataset_to_array, discr_train_step
from gantk2.losses import grad_particle_loss_factory, inf_discr_losses, \
    grad_discr_loss_factory
from gantk2.utils.math import to_distributed, from_distributed
from gantk2.utils.setup import load_model, setup, load_dataset
from gantk2.plots.plot_adequation_2d import finalize_plot


if __name__ == '__main__':
    rc('font', **{'family': 'serif', 'serif': 'Computer Modern'})
    rc('text', usetex=True)

    # Parse arguments
    p = args.create_setup_args()
    p = args.create_model_args(p)
    p = args.create_plot_args(p)
    p = args.create_dataset_args(p)
    p = args.create_config_args(p)
    opt = args.parse_args(p)

    assert opt.loss not in inf_discr_losses, 'You need to select finite loss'

    # Setup
    jax_rng_key_discr, _, _, _ = setup(opt)
    n_gpu = jax.local_device_count()

    # Data
    (in_loader, in_dataset, in_collate_fn), z_in_loader, (out_loader, out_dataset, out_collate_fn) = load_dataset(opt)
    x, _ = dataset_to_array(in_dataset, in_collate_fn)
    y = next(out_loader)
    _, x_indices = next(in_loader)
    x_batch_discr = x[x_indices]
    y_batch = next(out_loader)
    x_grid = jnp.linspace(-3, 3, opt.nx).reshape(-1, 1)

    # Plot Setup
    fig = plt.figure(figsize=(5, 2.5))
    ax = plt.subplot(111)
    ax.tick_params(colors='black')
    ax.grid(which='major', linestyle='--')
    plt.tick_params(axis="x", direction="in")
    plt.tick_params(axis="y", direction="in")
    variable, variable_name = 'discr_hidden', 'Width'
    values = np.array([64, 128, 256, 512])
    markers = iter(['>', 'P', 's', 'D', '$\infty$'])
    colors = iter(['darkviolet', 'green', 'chocolate', 'gold', 'black'])

    for value in values:
        setattr(opt, variable, value)

        # Initializing Finite Discriminator
        (discr_model, discr_params, discr_opt), \
        (gen_model, gen_params, gen_opt) = load_model(opt)
        discr_loss_grad = grad_discr_loss_factory(opt.loss)
        discr_init_fn, discr_apply_fn, discr_kernel_fn = discr_model

        # Training Finite Discriminator
        new_discr_opt_state, new_discr_params = discr_train_step(opt, discr_init_fn, discr_apply_fn, discr_opt,
                                                                 discr_loss_grad, jax_rng_key_discr,
                                                                 x, x_batch_discr, y_batch, n_gpu,
                                                                 discr_opt_state=None, step=0,
                                                                 discr_params=None, )

        gen_particles_grad = grad_particle_loss_factory(opt.loss, opt.discr_integration_time, opt.discr_momentum)

        # Plotting Finite
        (_, sol), _ = gen_particles_grad(to_distributed(x_grid, n_gpu), new_discr_params, discr_apply_fn)
        color = next(colors)
        ax.plot(x_grid.squeeze(1), from_distributed(sol),
                label=f'{value}', mec=color,
                c=color, marker=next(markers), markevery=0.1)

    # Plotting Infinite
    inf_gen_particles_grad = grad_particle_loss_factory('inf_' + opt.loss, opt.discr_integration_time,
                                                        opt.discr_momentum)
    (_, inf_sol), _ = inf_gen_particles_grad(to_distributed(x_grid, n_gpu), x_batch_discr, y_batch,
                                             nt.batch(discr_kernel_fn))
    ax.plot(x_grid.squeeze(1), from_distributed(inf_sol),
            label='$\infty$', c=next(colors), marker=next(markers),
            markevery=0.1, markersize=10)

    # More Plot Configs
    plt.title(f'{opt.loss.upper()}', fontsize=12)
    ax.set_xlabel(r'$x$', fontsize=12)
    ax.xaxis.set_label_coords(1.015, 0.01)
    ax.set_ylabel(r'$c_{f^\star}$', fontsize=12, rotation=0)
    ax.yaxis.set_label_coords(-0.01, 1.01)
    legend = plt.legend(loc="upper right", title=variable_name)
    legend.get_frame().set_linewidth(1.0)
    ax.scatter(x_batch_discr.squeeze(1), jnp.zeros((x_batch_discr.shape[0],)), 75, label='input', c='b', marker=r'^')
    ax.scatter(y_batch.squeeze(1), jnp.zeros((y_batch.shape[0],)), 75, label='target', c='r', marker=r'v')
    finalize_plot(opt.plot_output_file)
