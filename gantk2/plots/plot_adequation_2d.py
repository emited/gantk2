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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc
from mpl_toolkits.axes_grid1 import ImageGrid
import neural_tangents as nt

import gantk2.args.args as args
from gantk2.utils.utils import dataset_to_array, discr_train_step
from gantk2.utils.eval import scatter_ax, make_2d_grid
from gantk2.losses import grad_particle_loss_factory, inf_discr_losses, \
    grad_discr_loss_factory
from gantk2.utils.math import to_distributed, from_distributed
from gantk2.utils.setup import load_model, setup, load_dataset


def finalize_plot(plot_output_file=None):
    plt.tight_layout()
    # SAVING PLOT
    if plot_output_file is None:
        plt.show()
    else:
        print(f'Saving plot to {plot_output_file}')
        plt.savefig(plot_output_file)
    plt.close()


def plot_adequation_2d(opt, x, y, inf_grad_fn, grad_fn):
    rc('font', **{'family': 'serif', 'serif': 'Computer Modern'})
    rc('text', usetex=True)

    fig = plt.figure(figsize=(5, 5))
    axes = ImageGrid(fig, 111,  # as in plt.subplot(111)
                     nrows_ncols=(2, 1),
                     axes_pad=0.3,
                     share_all=True,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="2%",
                     cbar_pad=0.1,
                     aspect=False
                     )

    quiver_params = dict(
        angles='xy',
        scale_units='xy',
        linestyle='solid',
        headwidth=3.5, width=0.006,
        color='white'
    )

    imshow_params = dict(
        origin='lower',
        aspect='auto',
        cmap=cm.viridis,
    )

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    loss2title = {'lsgan': 'LSGAN', 'ipm': 'IPM'}
    axes[0].set_title(f'{loss2title[opt.loss]}, width $= {opt.discr_hidden}$', fontsize=12)
    scatter_ax(np.flip(x, axis=1), 75, ax=axes[0], label='input', c='b', marker=r'^', alpha=0., edgecolor='white',
               linewidths=1.)
    scatter_ax(np.flip(y, axis=1), 75, ax=axes[0], label='target', c='r', marker=r'v', alpha=0.0, edgecolor='white',
               linewidths=1.)

    axes[0].tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off

    scatter_extent = axes[0].axis()
    grid = make_2d_grid(*scatter_extent, nx=opt.nx, ny=opt.ny)
    grid = grid.reshape(opt.nx * opt.ny, -1)

    grad_fn, partial_args = grad_fn
    (_, sol), _ = grad_fn(to_distributed(grid, n_gpu), *partial_args)
    grad_x = from_distributed(grad_fn(to_distributed(x, n_gpu), *partial_args)[1])

    inf_grad_fn, partial_args = inf_grad_fn
    (_, inf_sol), inf_grad = inf_grad_fn(to_distributed(grid, n_gpu), *partial_args)
    inf_grad_x = from_distributed(inf_grad_fn(to_distributed(x, n_gpu), *partial_args)[1])

    # FINITE PLOTTING
    vmin = min(inf_sol.min().item(), sol.min().item())
    vmax = max(inf_sol.max().item(), sol.max().item())
    im0 = axes[0].imshow(from_distributed(sol).reshape(opt.ny, opt.nx).transpose(), extent=scatter_extent, vmin=vmin,
                   vmax=vmax, **imshow_params)
    axes[0].quiver(x[:, 1], x[:, 0], -grad_x[:, 1], -grad_x[:, 0], **quiver_params)

    scatter_ax(np.flip(x, axis=1), 75, ax=axes[0], label='input', c='b', marker=r'^', edgecolor='white',
               linewidths=1.)  # , alpha=0.7)
    scatter_ax(np.flip(y, axis=1), 75, ax=axes[0], label='target', c='r', marker=r'v', edgecolor='white',
               linewidths=1.)  # , alpha=0.7)

    # INFINITE PLOTTING
    axes[1].set_title(f'{loss2title[opt.loss]}, width $= \infty$', fontsize=12)
    axes[1].quiver(x[:, 1], x[:, 0], -inf_grad_x[:, 1], -inf_grad_x[:, 0],
                   **quiver_params)
    im1 = axes[1].imshow(from_distributed(inf_sol).reshape(opt.ny, opt.nx).transpose(),
                   extent=scatter_extent, vmin=vmin, vmax=vmax, **imshow_params)
    scatter_ax(np.flip(x, axis=1), 75, ax=axes[1], label='input', c='b', marker=r'^', edgecolor='white',
               linewidths=1.)  # , alpha=0.7)
    scatter_ax(np.flip(y, axis=1), 75, ax=axes[1], label='target', c='r', marker=r'v', edgecolor='white',
               linewidths=1.)  # , alpha=0.7)

    # COLORBAR
    clb = axes[1].cax.colorbar(im1)
    axes[1].cax.toggle_label(True)
    clb.ax.set_title(r'$c_{f^\star}$')

    finalize_plot(opt.plot_output_file)


if __name__ == '__main__':
    # Parse arguments
    p = args.create_setup_args()
    p = args.create_model_args(p)
    p = args.create_plot_args(p)
    p = args.create_dataset_args(p)
    p = args.create_config_args(p)
    opt = args.parse_args(p)
    assert opt.loss not in inf_discr_losses, 'You need to select finite loss'

    # Setup
    jax_rng_key_discr, jax_rng_key_gen, \
    img_save_dir, chkpt_save_dir = setup(opt)

    # Data
    (in_loader, in_dataset, in_collate_fn), z_in_loader, (out_loader, out_dataset, out_collate_fn) = load_dataset(opt)
    x, _ = dataset_to_array(in_dataset, in_collate_fn)
    y = next(out_loader)
    _, x_indices = next(in_loader)
    x_batch_discr = x[x_indices]
    y_batch = next(out_loader)

    # Model
    (discr_model, discr_params, discr_opt), \
    (gen_model, gen_params, gen_opt) = load_model(opt)
    discr_init_fn, discr_apply_fn, discr_kernel_fn = discr_model
    batched_discr_kernel_fn = nt.batch(discr_kernel_fn)
    discr_opt_state = None
    discr_loss_grad = grad_discr_loss_factory(opt.loss)
    n_gpu = jax.local_device_count()

    # Train finite discriminator and get gradient func
    new_discr_opt_state, new_discr_params = discr_train_step(opt, discr_init_fn, discr_apply_fn, discr_opt,
                                                             discr_loss_grad, jax_rng_key_discr,
                                                             x, x_batch_discr, y_batch, n_gpu,
                                                             discr_opt_state=None, step=0,
                                                             discr_params=None, )
    gen_particles_grad = grad_particle_loss_factory(opt.loss, opt.discr_integration_time, opt.discr_momentum)

    # Directly get infite gradient func
    inf_gen_particles_grad = grad_particle_loss_factory('inf_' + opt.loss, opt.discr_integration_time,
                                                        opt.discr_momentum)

    # plotting finite & infinite
    plot_adequation_2d(opt, x, y, (inf_gen_particles_grad, (x, y, discr_kernel_fn)),
                       grad_fn=(gen_particles_grad, (new_discr_params, discr_apply_fn)))
