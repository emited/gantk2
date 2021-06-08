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


import jax.numpy as np
import matplotlib.pyplot as plt

import gantk2.args.args as args
from gantk2.losses import grad_particle_loss_factory
from gantk2.plots.plot_adequation_2d import finalize_plot
from gantk2.utils.setup import load_model, setup


def grad_field_fn(particle_grad_fn, x, y, discr_kernel_fn):
    grads = []
    for i in range(len(x)):
        x_distributed_i = x[None, np.array([i])]
        x_batch_discr_i = x[np.array([i])]
        _, grad = particle_grad_fn(x_distributed_i, x_batch_discr_i, y, discr_kernel_fn)
        grads.append(grad)
    return np.concatenate(grads, axis=1).squeeze(0)


def vector_field_viz_2d(nx, ny, dim, loss, discr_kernel_fn,
                        discr_integration_time, discr_momentum,
                        plot_output_file=None):
    x0, x1 = np.linspace(-2, 2, nx), np.linspace(0, 2, ny)
    mx0, mx1 = np.meshgrid(x0, x1)
    x01 = np.column_stack([
        mx0.flatten(),
        mx1.flatten(),
    ])

    x = np.column_stack([x01, np.zeros((nx * ny, dim - 2))])
    # x_prime = np.column_stack([np.array([[20, 20]]),
    #                            np.zeros((1, dim - 2))])
    # x_prime = None
    y = np.column_stack([np.array([[1, 0]]),
                         np.zeros((1, dim - 2))])

    gen_particles_grad = grad_particle_loss_factory(loss, discr_integration_time, discr_momentum)
    g = grad_field_fn(gen_particles_grad, x, y, discr_kernel_fn)
    plt.quiver(x[:, 0], x[:, 1], -g[:, 0], -g[:, 1], units='xy')
    plt.scatter(y[:, 0], y[:, 1], 120, label='$\\hat{\\beta}$', c='r', marker=r'x', alpha=1)
    finalize_plot(plot_output_file)

    return plt.gcf()


if __name__ == '__main__':
    # Parse arguments
    p = args.create_setup_args()
    p = args.create_model_args(p)
    p = args.create_plot_args(p)
    p = args.create_config_args(p)
    opt = args.parse_args(p)

    # Setup
    setup(opt)

    (discr_model, _, _), _ = load_model(opt)
    _, _, discr_kernel_fn = discr_model

    vector_field_viz_2d(opt.nx, opt.ny, opt.dim, opt.loss, discr_kernel_fn,
        opt.discr_integration_time, opt.discr_momentum, opt.plot_output_file)

