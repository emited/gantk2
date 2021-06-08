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


from copy import deepcopy

import configargparse

from gantk2.args import exp_configs
from gantk2.args.utils import create_default_parser_wrapper, get_arg_group
from gantk2.data.factory import datasets
from gantk2.data.simple import distributions
from gantk2.data.transforms import transforms
from gantk2.losses import losses
from gantk2.models.factory import models, parameterizations
from gantk2.models.utils import activations
from gantk2.utils.setup import optimizers


@create_default_parser_wrapper
def create_setup_args(p):
    p.add('--seed', type=int, metavar='SEED', default=None,
          help='Manual seed. If None, it is chosen randomly.')
    p.add('--device', type=int, metavar='DEVICE', default=None, nargs='+',
          help='If not None, indicates the list of GPU indices to use with CUDA.')
    return p


@create_default_parser_wrapper
def create_log_args(p):
    logs_p = p.add_argument_group(title='Logs',
                                  description='Logging options.')
    logs_p.add('--save_path', type=str, metavar='PATH', required=True,
               help='Path where the experiment directory should be created.')
    logs_p.add('--save_name', type=str, metavar='NAME', required=True,
               help='Experiment directory name.')
    logs_p.add('--erase', action='store_true',
               help='Whether to automatically erase the previous experiment directory with the same name.')
    return p


@create_default_parser_wrapper
def create_dataset_args(p):
    data_p = p.add_argument_group(title='Dataset',
                                  description='Chosen dataset and parameters.')
    data_p.add('--in_dataset', type=str, metavar='DATASET', default=datasets[0], choices=datasets,
               help='Source dataset name.')
    data_p.add('--in_transform', type=str, metavar='TRANS', default=transforms[0], choices=transforms,
               help='Transformation to apply to elements of the source dataset.')
    data_p.add('--in_batch_size', type=int, metavar='SIZE', default=0,
               help='Batch size for the source distribution. If zero, takes the size of the source dataset.')
    data_p.add('--out_dataset', type=str, metavar='DATASET', default=datasets[0], choices=datasets,
               help='Target dataset name.')
    data_p.add('--out_transform', type=str, metavar='TRANS', default=transforms[0], choices=transforms,
               help='Transformation to apply to elements of the target dataset.')
    data_p.add('--out_batch_size', type=int, metavar='SIZE', default=0,
               help='Batch size for the target distribution. If zero, takes the size of the target dataset.')
    data_p.add('--data_dim', type=int, metavar='NB', default=[2], nargs='+',
               help='Data dimensionality (ignored if one of both datasets is not simple).')
    data_p.add('--data_path', type=str, metavar='DIR', required=False,
               help='Data directory.')

    simple_data_p = data_p.add_argument_group(title='Simple Distribution Parameters')
    simple_data_p.add('--out_distribution', type=str, metavar='DISTR', default=distributions[0], choices=distributions,
                      help='Target distribution type.')
    simple_data_p.add('--out_nb_samples', type=int, metavar='NB', default=20,
                      help='Number of samples for the target distribution to use. If zero, samples indefinitely from '
                           + 'the latter.')
    simple_data_p.add('--out_nb_components', type=int, metavar='NB', default=3,
                      help='Number of components in the target mixture distribution (if not image-based).')
    simple_data_p.add('--out_mix_dev', type=float, metavar='LOCDEV', default=3.,
                      help='Deviation from the center of the target mixture distribution (if not image-based).')
    simple_data_p.add('--out_loc', type=float, metavar='LOC', default=[0.], nargs='+',
                      help='Location of the target mixture distribution.')
    simple_data_p.add('--out_scale', type=float, metavar='SCALE', default=[0.5], nargs='+',
                      help='Scale of components of the target mixture distribution.')
    simple_data_p.add('--out_img_name', type=str, metavar='NAME', default='out_data.png',
                      help='Image representation of the target distribution (if image-based).')
    simple_data_p.add('--in_distribution', type=str, metavar='DISTR', default=distributions[0], choices=distributions,
                      help='Source distribution type.')
    simple_data_p.add('--in_nb_samples', type=int, metavar='NB', default=20,
                      help='Number of samples for the source distribution to use. If zero, samples indefinitely from '
                           + 'the latter.')
    simple_data_p.add('--in_nb_components', type=int, metavar='NB', default=1,
                      help='Number of components in the source mixture distribution (if not image-based).')
    simple_data_p.add('--in_mix_dev', type=float, metavar='LOCDEV', default=0.,
                      help='Deviation from the mean parameter for the source mixture distribution (if not image-based).')
    simple_data_p.add('--in_loc', type=float, metavar='LOC', default=[0.], nargs='+',
                      help='Location of the source mixture distribution.')
    simple_data_p.add('--in_scale', type=float, metavar='SCALE', default=[1.], nargs='+',
                      help='Scale of components of the source mixture distribution.')
    simple_data_p.add('--in_img_name', type=str, metavar='NAME', default='in_data.png',
                      help='Image representation of the source distribution (if image-based).')

    return p


@create_default_parser_wrapper
def create_model_args(p):
    model_p = p.add_argument_group(title='Generative Models',
                                   description='Choice of generative models.')
    model_p.add('--generator', action='store_true',
                help='Whether to use a generator. If True, takes as input the input distribution.')
    model_p.add('--loss', type=str, metavar='LOSS', default=losses[0], choices=losses,
                help='Choice of generator and discriminator loss.')
    model_p.add('--discr_integration_time', type=float, metavar='T', default=100,
                help='Integration time for infinite-width discriminator training.')
    model_p.add('--discr_momentum', type=float, metavar='GAMMA', default=None,
                help='Momentum for the infinite-width discriminator of vanilla GAN.')

    archi_p = p.add_argument_group(title='Architecture parameters',
                                   description='Discriminator and generator architecture parameters.')
    archi_p.add('--gen_model', type=str, metavar='MODEL', default=models[0], choices=models,
                help='Generator network type.')
    archi_p.add('--gen_in_dim', type=int, metavar='NB', default=64,
                help='Number of input dimensions of the generator.')
    archi_p.add('--gen_nb_z', type=int, metavar='NB', default=0,
                help='Number of latents used to compute the gradient estimate. If zero, takes the size of the '
                     + 'source dataset.')
    archi_p.add('--gen_parameterization', type=str, metavar='PARAM', default=parameterizations[0],
                choices=parameterizations,
                help='Type of parameterization for the generator.')
    archi_p.add('--gen_activations', type=str, metavar='ACT', default=[activations[0]], choices=activations, nargs='+',
                help='Activation to use in the generator.')
    archi_p.add('--gen_out_act', type=str, metavar='ACT', default=None, choices=activations,
                help='Output activation of the generator.')
    archi_p.add('--gen_layer_norm', action='store_true',
                help='Uses layer norm before activations in the generator.')
    archi_p.add('--gen_hidden', type=int, metavar='NB', default=512,
                help='Hidden size in generator MLP.')
    archi_p.add('--gen_layers', type=int, metavar='NB', default=3,
                help='Number of hidden layers in generator MLP.')
    archi_p.add('--gen_channels', type=int, metavar='NB', default=32,
                help='Number of hidden channels in generative convolutional networks.')
    archi_p.add('--gen_W_std', type=float, metavar='W_STD', default=1.,
                help='Initialisation parameter for the generator weight matrix.')
    archi_p.add('--gen_b_std', type=float, metavar='B_STD', default=0.,
                help='Initialisation parameter for the generator bias weights.')
    archi_p.add('--gen_eta', type=float, metavar='LR', default=1.,
                help='Generator learning rate.')
    archi_p.add('--discr_model', type=str, metavar='MODEL', default=models[0], choices=models,
                help='Discriminator network type.')
    archi_p.add('--discr_parameterization', type=str, metavar='PARAM', default=parameterizations[0],
                choices=parameterizations,
                help='Type of parameterization for the discriminator.')
    archi_p.add('--discr_activations', type=str, metavar='ACT', default=[activations[0]], choices=activations,
                nargs='+',
                help='Activation to use in the discriminator.')
    archi_p.add('--discr_layer_norm', action='store_true',
                help='Uses layer norm before activations in the discriminator.')
    archi_p.add('--discr_hidden', type=int, metavar='NB', default=512,
                help='Hidden size in discriminator MLP.')
    archi_p.add('--discr_layers', type=int, metavar='NB', default=3,
                help='Number of hidden layers in discriminator MLP.')
    archi_p.add('--discr_block_depth', type=int, metavar='NB', default=2,
                help='Number of hidden layers in discriminator Resnet Block.')
    archi_p.add('--discr_channels', type=int, metavar='NB', default=32,
                help='Number of hidden channels in discriminative convolutional networks.')
    archi_p.add('--discr_W_std', type=float, metavar='W_STD', default=1.,
                help='Initialisation parameter for the discriminator weight matrix.')
    archi_p.add('--discr_b_std', type=float, metavar='B_STD', default=0.,
                help='Initialisation parameter for the discriminator bias weights.')
    archi_p.add('--discr_antisymmetric', action='store_true',
                help='Transforms the discriminator to use an antisymmetric initialization.')
    archi_p.add('--discr_optimizer', type=str, default=optimizers[0], choices=optimizers,
                help='Discriminator optimizer')
    archi_p.add('--kernel_transform', type=str, metavar='TRANS', default=transforms[0], choices=transforms,
                help='Transformation to apply to inputs before feeding them to the discriminator NTK.')
    archi_p.add('--nngp', action='store_true',
                help='Uses the NNGP kernels instead of NTKs.')
    archi_p.add('--discr_eta', type=float, metavar='LR', default=1.,
                help='Discriminator learning rate.')
    archi_p.add('--b1', type=float, metavar='BETA1', default=0.9,
                help='First decay rate for Adam.')
    archi_p.add('--b2', type=float, metavar='BETA1', default=0.999,
                help='Second decay rate for Adam.')
    archi_p.add('--gen_optimizer', type=str, default=optimizers[0], choices=optimizers,
                help='Generator optimizer')

    act_p = p.add_argument_group(title='Activation parameters',
                                 description='Discriminator and generator activations parameters.')
    act_p.add('--act_a', type=float, metavar='A', default=1.,
              help='Parameter `a` of Sin and Erf activations.')
    act_p.add('--act_b', type=float, metavar='B', default=1.,
              help='Parameter `b` of Sin and Erf activations.')
    act_p.add('--act_c', type=float, metavar='C', default=1.,
              help='Parameter `c` of Sin and Erf activations.')
    act_p.add('--act_gamma', type=float, metavar='GAMMA', default=1.,
              help='Parameter `gamma` of the Rbf activation.')
    act_p.add('--act_alpha', type=float, metavar='ALPHA', default=0.2,
              help='Parameter `alpha` of the LeakyRelu activation.')
    act_p.add('--act_do_stabilize', action='store_true',
              help='Parameter `do_stabilize` of the Relu, LeakyRelu and Abs activations.')
    return p


@create_default_parser_wrapper
def create_training_args(p):
    logs_p = get_arg_group(p, 'Logs')
    logs_p.add('--log_steps', type=int, metavar='SIZE', default=100,
               help='Frequency of logging steps.')
    logs_p.add('--save_steps', type=int, metavar='SIZE', default=5000,
               help='Frequency of saving steps.')
    logs_p.add('--save_target', action='store_true',
               help='Saves samples from the target distribution, if possible.')

    desc_p = p.add_argument_group(title='Gradient Descent',
                                  description='Descent parameters.')
    desc_p.add('--nb_iter', type=int, metavar='NB', default=50000,
               help='Number of gradient descent iterations.')
    desc_p.add('--nb_d_steps', type=int, metavar='NB', default=1,
               help='Number of optimization steps of the discriminator per iteration.')
    desc_p.add('--reset_discr', action='store_true',
               help='Resets the discriminator at each generator iteration.')
    desc_p.add('--eta', type=float, metavar='LR', default=1.,
               help='Particle descent learning rate.')
    desc_p.add('--sinkhorn_blur', type=float, metavar='SB', default=0.001,
               help='Particle descent learning rate.')
    return p


@create_default_parser_wrapper
def create_plot_args(p):
    plot_p = p.add_argument_group(title='Visualization',
                                 description='Choice of Visualization.')

    def check_dim(dim):
        dim = int(dim)
        if dim < 2:
            raise configargparse.ArgumentTypeError("dim must be at least 2 but got %s" % dim)
        return dim

    plot_p.add('--dim', type=check_dim, metavar='NB', default=2,
              help='Number of input dimensions of the generator.')
    plot_p.add('--nx', type=int, metavar='NB', default=15,
              help='Resolution of x axis.')
    plot_p.add('--ny', type=int, metavar='NB', default=15,
              help='Resolution of x axis.')
    plot_p.add('--plot_grad', action='store_true',
              help='Whether to grad of discr or discr. For 1d plots only.')
    plot_p.add('--quiver_scale', type=float, metavar='QS', default=1.,
              help='Quiver scale of gradients of 2d plot. Smaller means larger arrows.')
    plot_p.add('--plot_output_file', type=str, metavar='FILEPATH', default=None,
               help='Output file to save the plot.')
    return p



@create_default_parser_wrapper
def create_config_args(p):
    config_p = p.add_argument_group(title='Config Args',
                                 description='Choice of sets of Config Arguments to load and overwrite standard args.')
    config_p.add('--loss_config', type=str, metavar='LOSS', default=None, choices=list(exp_configs.loss_config.keys()),
               help='')
    config_p.add('--data_config', type=str, metavar='DATA', default=None, choices=list(exp_configs.data_config.keys()),
               help='')
    config_p.add('--arch_config', type=str, metavar='ARCH', default=None, choices=list(exp_configs.arch_config.keys()),
               help='')
    config_p.add('--ade1d_config', type=str, metavar='ADE1D', default=None, choices=list(exp_configs.ade1d_config.keys()),
               help='Load config used for the 1d Adequation experiments in the paper.')
    config_p.add('--ade2d_config', type=str, metavar='ADE2D', default=None, choices=list(exp_configs.ade2d_config.keys()),
                 help='Load config used for the 2d Adequation experiments in the paper.')
    return p


def parse_args(p, verbose=True):
    opt = p.parse_args()
    new_opt = deepcopy(opt)

    # load args from args with *_config and overwrite others
    # with these loaded sets of arguments
    for opt_name, opt_value in opt.__dict__.items():
        if opt_name.endswith('_config') and opt_value is not None:
            if verbose:
                print(f'Loading Config {opt_name}: {opt_value}...')
            conf = getattr(exp_configs, opt_name)[opt_value]
            for opt_name_conf in conf:
                if hasattr(opt, opt_name_conf):
                    old_value = getattr(opt, opt_name_conf)
                    new_value = conf[opt_name_conf]
                    if old_value != new_value:
                        setattr(new_opt, opt_name_conf, conf[opt_name_conf])
                        if verbose:
                            print(f'\treplacing {opt_name_conf}'
                                  f' from {old_value}'
                                  f' to {new_value}')

    return new_opt
