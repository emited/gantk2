<!-- Copyright 2021 Ibrahim Ayed, Emmanuel de Bézenac, Mickaël Chen, Jean-Yves Franceschi, Sylvain Lamprier, Patrick Gallinari

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. -->


# GAN(TK)²: Generative Adversarial Neural Tangent Kernel ToolKit

GAN analysis toolkit accompanying the paper *A Neural Tangent Kernel Perspective of GANs* (Jean-Yves Franceschi,* Emmanuel de Bézenac,* Ibrahim Ayed,* Mickaël Chen, Sylvain Lamprier, Patrick Gallinari).


## Requirements

This code was tested with Python 3.8.1 and 3.9.2, and run on GPUs Nvidia Titan RTX (24GB of VRAM) with CUDA 11.2 as well as Nvidia Titan V (12GB) and Nvidia GeForce RTX 2080 Ti (11 GB) with CUDA 10.2.

The code is primarily based on [JAX](https://github.com/google/jax) and [Neural Tangents](https://github.com/google/neural-tangents).
A list of required Python packages is available in the [`requirements.txt`](requirements.txt) file.

We refer to [Jax installation instructions](https://github.com/google/jax/#installation) in order to perform computations on GPU.

:warning: We strongly advise users to use the GPU, as inconsistent behavior has been observed when using the CPU.


## Reproducing Experiments


### Dataset Download

To download the Density and AB datasets, execute the following command, which will save them in `gantk2/data/images`.
```bash
bash gantk2/data/download_images.sh
```


### Launch

We provide the following proxy command in order to reproduce the experiments of the paper.
```bash
python -m gantk2.train --loss_config $LOSS_CONFIG --arch_config $ARCH_CONFIG --data_config $DATA_CONFIG --save_path $SAVE_PATH --save_name $SAVE_NAME --device $DEVICE
```
where `$DEVICE` is the GPU index, `$SAVE_PATH` is the directory where the experiment folder will be created and `$SAVE_NAME` is the name of the experiment folder.

Different options are available for `$LOSS_CONFIG`, `$ARCH_CONFIG` and `$DATA_CONFIG`, corresponding to the sets of hyperparameters used for the experiments of the paper:
 - for `$LOSS_CONFIG`: `inf_ipm` (infinite-width IPM), `ipm` (finite-width IPM), `ipm_reset` (finite-width IPM with reset), `inf_lsgan` (infinite-width LSGAN) or `lsgan` (finite-width LSGAN);
 - for `$ARCH_CONFIG`: `rbf` (RBF kernel, only for infinite-width losses), `relu`, `relu_nobias`;
 - for `$DATA_CONFIG`: `eight_gaussians`, `density`, `ab`, `mnist`.

For example, to reproduce the experiment on the eight Gaussians dataset with a ReLU network in the infinite-width regime and the IPM loss:
```bash
python -m gantk2.train --loss_config inf_ipm --arch_config relu --data_config eight_gaussians --device 0 --save_path saves --save_name test
```

The saved experiment folder contains a configuration file, visualizations in the `img` subfolder and checkpoints and metrics in `chkpt`.
In particular, `chkpt/metrics.csv` contains metrics for all tested timesteps during training (the Sinkhorn divergence corresponding to the `s` column).

We refer to [`gantk2/args/exp_configs.py`](gantk2/args/exp_configs.py) for details about these premade configurations, and to [`gantk2/args/args.py`](gantk2/args/args.py) for the complete set of arguments of the training script, which can also be obtained via:
```
python -m gantk2.train --help
```

## Reproducing Plots

We provide here commands to reproduce the plots shown in the paper.

### 1D Adequation Plots (Figure 1, left)

Execute the following command:
```bash
python -m gantk2.plots.plot_adequation_1d --ade1d_config $ADE1D_CONFIG --device $DEVICE [--plot_output_file $PLOT_OUTPUT_FILE]
```
where `$DEVICE` is the GPU index and `$PLOT_OUTPUT_FILE` is the file name where the plot will be saved.
By default, the plot is shown and not saved.

Two options are available for `$ADE1D_CONFIG`, corresponding to the sets of hyperparameters used for the 1d plots of the paper: `ipm_relu` (IPM with ReLU Discriminator), `lsgan_relu` (LSGAN with ReLU Discriminator).

### 2D Adequation Plots (Figure 1, right)

```bash
python -m gantk2.plots.plot_adequation_2d  --ade2d_config $ADE2D_CONFIG --device $DEVICE [--plot_output_file $PLOT_OUTPUT_FILE]
```

where `$ADE2D_CONFIG` takes only `lsgan_relu` (LSGAN with ReLU Discriminator) as option, corresponding to the sets of hyperparameters used for the 2d plots of the paper.


Note that other arguments may also be tested, such as `--loss_config $LOSS_CONFIG`, or `--arch_config $ARCH_CONFIG`, etc... where
`$LOSS_CONFIG` takes value `ipm` or `lsgan`. For other argument values, refer to the last section and to [*Reproducing Experiments*](#Reproducing-Experiments).

### Vector Field Plots (Figures 6 and 7)
```bash
python -m gantk2.plots.plot_vector_field --loss_config $LOSS_CONFIG --arch_config $ARCH_CONFIG --device $DEVICE [--plot_output_file $PLOT_OUTPUT_FILE]
```

For argument values, refer to the last sections and to [*Reproducing Experiments*](#Reproducing-Experiments).

### Visualization of Distributions (Figure 2)

Corresponding plots can be found in the `img` subfolder of the chosen experiment directory.
