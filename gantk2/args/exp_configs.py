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


loss_config = {
    'lsgan': {
        'loss': 'lsgan',
        'eta': 1.,
        'discr_eta': 0.1,
        'discr_integration_time': 1.,
        'discr_optimizer': 'sgd',
    },
    'inf_lsgan': {
        'eta': 1000,
        'loss': 'inf_lsgan',
        'discr_integration_time': 1.,
        'quiver_scale': 0.01,
    },
    'inf_ipm': {
        'loss': 'inf_ipm',
        'eta': 1000,
        'discr_eta': 1,
        'discr_integration_time': 1,
    },
    'ipm_reset': {
        'loss': 'ipm',
        'eta': 1000.,
        'discr_eta': 0.1,
        'discr_integration_time': 1.,
        'reset_discr': True,
        'discr_antisymmetric': True,
    },
    'ipm': {
        'loss': 'ipm',
        'eta': 100.,
        'discr_eta': 0.01,
        'discr_integration_time': 0.1,
        'discr_antisymmetric': True,
    }

}


arch_config = {
    'relu': {
        'discr_layers': 3,
        'discr_activations': ['relu'],
        'discr_hidden': 128,
        'discr_b_std': 1.,
    },
    'relu_nobias': {
        'discr_layers': 3,
        'discr_activations': ['relu'],
        'discr_hidden': 128,
        'discr_b_std': 0.,
    },
    'rbf': {
        'discr_integration_time': 1,
        'discr_layers': 1,
        'discr_activations': ['rbf'],
        'act_gamma': 0.5,
        'nngp': True,
        'discr_model': 'mlp',
        'discr_b_std': 0,
    },
    'sigmoid': {
        'discr_layers': 3,
        'discr_activations': ['sigmoid_like'],
        'discr_hidden': 128,
        'discr_b_std': 0.2,
    },
}


data_config = {
    'density': {
        'data_path': 'gantk2/data/images',
        'in_img_name': 'density_a.png',
        'out_img_name': 'density_b.png',
        'in_distribution': 'image',
        'out_distribution': 'image',
        'out_scale': [1.],
        'in_scale': [1.],
        'data_dim': [2],
        'in_nb_samples': 500,
        'out_nb_samples': 500,
        'in_batch_size': 500,
        'out_batch_size': 500,
        'gen_nb_z': 500,
        'in_dataset': 'simple',
        'out_dataset': 'simple',
    },
    'ab': {
        'data_path': 'gantk2/data/images',
        'in_img_name': 'A.png',
        'out_img_name': 'B.png',
        'in_distribution': 'image',
        'out_distribution': 'image',
        'out_scale': [1.],
        'in_scale': [1.],
        'data_dim': [2],
        'in_nb_samples': 500,
        'out_nb_samples': 500,
        'in_batch_size': 500,
        'out_batch_size': 500,
        'gen_nb_z': 500,
        'in_dataset': 'simple',
        'out_dataset': 'simple',
    },
    'eight_gaussians': {
        'in_dataset': 'simple',
        'out_dataset': 'simple',
        'in_nb_samples': 500,
        'in_distribution': 'gaussian',
        'out_distribution': 'gaussian',
        'data_dim': [2],
        'out_nb_samples': 500,
        'out_mix_dev': 5,
        'out_nb_components': 8,
        'gen_nb_z': 500,
        'in_batch_size': 500,
    },
    'mnist': {
        'data_path': 'data',
        'in_dataset': 'simple',
        'out_dataset': 'mnist',
        'in_nb_samples': 1024,
        'in_distribution': 'gaussian',
        'out_nb_samples': 1024,
        'gen_nb_z': 1024,
        'in_batch_size': 1024,
        'eta': 100.,
        'discr_integration_time': 1000.,
    },
}


ade1d_config = {
    'ipm_relu': {
        'in_dataset': 'simple',
        'in_distribution': 'modes_1d_overlap',
        'out_distribution': 'modes_1d_overlap',
        'out_dataset': 'simple',
        'discr_hidden': 128,
        'discr_layers': 3,
        'discr_activations': ['relu'],
        'discr_optimizer': 'sgd',
        'discr_parameterization': 'ntk',
        'discr_integration_time': 1,
        'discr_eta': 0.1,
        'discr_b_std': 1,
        'loss': 'ipm',
        'discr_antisymmetric': True,
        'nx': 100,
    },
    'lsgan_relu': {
        'in_dataset': 'simple',
        'in_distribution': 'modes_1d',
        'out_distribution': 'modes_1d',
        'out_dataset': 'simple',
        'discr_optimizer': 'sgd',
        'discr_parameterization': 'ntk',
        'discr_activations': ['relu'],
        'discr_integration_time': 1,
        'discr_eta': 0.1,
        'discr_b_std': 1,
        'loss': 'lsgan',
        'discr_antisymmetric': True,
        'nx': 100,
    },
}


ade2d_config = {
    'lsgan_relu': {
        'data_path': 'gantk2/data/images',
        'in_img_name': 'density_a.png',
        'out_img_name': 'density_b.png',
        'in_distribution': 'image',
        'out_distribution': 'image',
        'out_scale': [1.],
        'in_scale': [1.],
        'data_dim': [2],
        'in_nb_samples': 30,
        'out_nb_samples': 30,
        'in_batch_size': 30,
        'out_batch_size': 30,
        'in_dataset': 'simple',
        'out_dataset': 'simple',
        'discr_hidden': 512,
        'discr_layers': 3,
        'discr_activations': ['relu'],
        'discr_optimizer': 'sgd',
        'discr_parameterization': 'ntk',
        'discr_integration_time': 1,
        'discr_eta': 0.1,
        'discr_b_std': 1,
        'loss': 'lsgan',
        'discr_antisymmetric': True,
        'nx': 100,
        'ny': 100,
    },
}
