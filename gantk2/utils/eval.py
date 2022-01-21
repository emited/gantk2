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


import os
import matplotlib.pyplot as plt
import numpy as np


from gantk2.data.factory import datasets
from gantk2.losses import inf_discr_losses
from gantk2.utils.math import to_distributed, from_distributed, samples_loss
from gantk2.utils.utils import append_dict_to_csv


def scatter_ax(x, *args, ax=None, **kwargs):
    if ax is not None:
        plt = ax
    if len(x.shape) < 3 and x.shape[1] == 1:
        x_reshaped = x.reshape(x.size)
        plt.scatter(x_reshaped, np.zeros_like(x_reshaped), *args, **kwargs)
    elif len(x.shape) < 3 and x.shape[1] == 2:
        plt.scatter(x[:, 0], x[:, 1], *args, **kwargs)
    else:
        raise NotImplementedError


def finalize_plot(save_dir, step):
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{step:08d}.png'))
    plt.close()


def make_2d_grid(xmin, xmax, ymin, ymax, nx=20, ny=20):
    x_grid = np.linspace(xmin, xmax, nx)
    y_grid = np.linspace(ymin, ymax, ny)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    grid = np.concatenate([np.expand_dims(X_grid, -1), np.expand_dims(Y_grid, -1)], -1)
    return grid


def plot_2d(config, particle_grad_fn, x=None, y=None,
            discr_params=None, discr_apply_fn=None, discr_kernel_fn=None,
            ax=None, step=None, loss_gen=None):
    n_gpu = 1
    if ax is None:
        w, h = 5, 5
        plt.figure(figsize=(w, h))
        ax = plt.gca()

    if y is not None:
        scatter_ax(y, ax=ax, label='target', alpha=0.5, c='orange')
    if x is not None:
        scatter_ax(x, ax=ax, label='input', alpha=0.5, c='m')
    scatter_extent = ax.axis()

    grid = make_2d_grid(*scatter_extent, nx=config.nx, ny=config.ny)
    grid = grid.reshape(config.nx * config.ny, 2)

    if config.loss in inf_discr_losses:
        assert discr_kernel_fn is not None
        assert x is not None and y is not None
        grad_fn_args = (x, y, discr_kernel_fn,)
    else:
        assert discr_params is not None and discr_apply_fn is not None
        grad_fn_args = (discr_params, discr_apply_fn)

    loss_grid, _ = particle_grad_fn(to_distributed(grid, n_gpu), *grad_fn_args)
    _, grad_x = particle_grad_fn(to_distributed(x, n_gpu), *grad_fn_args)
    grad_x = from_distributed(grad_x)
    if isinstance(loss_grid, tuple):
        # loss is of form (loss_value, loss_solution)
        _, sol_grid = loss_grid
        ax.imshow(from_distributed(sol_grid).reshape(config.ny, config.nx),
                  origin='lower', extent=scatter_extent, aspect='auto')

    quiver_scale = config.quiver_scale * 10. if config.generator else config.quiver_scale / 3.
    ax.quiver(x[:, 0], x[:, 1], -grad_x[:, 0], -grad_x[:, 1],
              angles='xy', scale_units='xy', scale=quiver_scale)
    plt.legend()
    return plt.gcf(), ax


def evaluate_mnist(x, data_transform):
    _, ax = plt.subplots(1, 1, figsize=(5, 5))
    data = np.concatenate([x[i] for i in range(min(len(x), 5))])
    if data_transform == 'atanh':
        data = np.tanh(data)
    data = data / 1.999 + 0.5
    ax.imshow(data, cmap='gray', vmin=0., vmax=1.)


def evaluate_celeba(x, data_transform):
    _, ax = plt.subplots(1, 1, figsize=(5, 5))
    data = np.concatenate([x[i] for i in range(min(len(x), 5))])
    if data_transform == 'atanh':
        data = np.tanh(data)
    data = data / 1.999 + 0.5
    ax.imshow(data, vmin=0., vmax=1.)


def evaluate(config, particle_grad_fn,
             x=None, y=None,
             discr_params=None, discr_apply_fn=None,
             discr_kernel_fn=None,
             step=None, loss_gen=None,
             img_save_dir=None, chkpt_save_dir=None):
    # Make Plots
    dataset = config.out_dataset
    assert dataset in datasets
    if dataset == 'simple':
        if len(config.data_dim) == 1 and config.data_dim[0] < 3:
            if config.data_dim[0] == 1:
                raise NotImplementedError('1d eval not done')
            else:
                plot_2d(config, particle_grad_fn,
                        x=x, y=y,
                        step=step,
                        discr_params=discr_params,
                        discr_apply_fn=discr_apply_fn,
                        discr_kernel_fn=discr_kernel_fn,
                        )
    elif dataset == 'mnist':
        evaluate_mnist(x, config.out_transform)
    elif dataset == 'celeba':
        evaluate_celeba(x, config.out_transform)
    else:
        raise ValueError(f'Evaluate not defined for dataset `{dataset}`')
    finalize_plot(img_save_dir, step)

    # Compute and Save metrics
    metrics = {'step': step}

    # Sinkhorn
    if dataset not in ['mnist', 'celeba']:
        metrics['s'] = samples_loss(x, y, 'sinkhorn', blur=config.sinkhorn_blur, scaling=.95)
        metrics['e'] = samples_loss(x, y, 'energy')

    if loss_gen is not None:
        metrics['lg'] = loss_gen

    # add row to CSV file
    if chkpt_save_dir is None:
        print(', '.join([f'{k}: {float(v):.4}' for k, v in metrics.items()]))
    else:
        metrics_filepath = os.path.join(chkpt_save_dir, 'metrics.csv')
        append_dict_to_csv(metrics_filepath, metrics)
    return metrics
