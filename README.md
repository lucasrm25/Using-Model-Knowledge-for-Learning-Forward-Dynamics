# Using Model Knowledge for Learning Forward Dynamics

Code related to the paper "Using Model Knowledge for Learning Forward Dynamics" submitted to the Conference on Robot Learning (CoRL) 2021.

Inspired on the paper "Learning Constrained Dynamics with Gauss’ Principle adhering Gaussian Processes" - A. Rene Geist and Sebastian Trimpe.

Main S-GP implementation can be found here: [sgp/sgp.py](sgp/sgp.py).

### How to execute the experiments

#### Generate Dataset

Before starting the to train the models, it is necessary to generate the dataset, obtained from simulation.
To do that, execute the script [01_simulate_KUKA_robot_arm.py](KUKA-experiment/01_simulate_KUKA_robot_arm.py), which requires, among other libraries, the pyBullet library installed.

This will execute the simulation and save the simulation results to ```KUKA-experiment/results/KUKA-surf-dataset/simdata_raw.dat```. The name and location of the file, as well as many other simulation settings can be changed in the configuration file [config_KUKA.py](KUKA-experiment/results/KUKA-surf-dataset/config_KUKA.py).

Next, execute the script [02_generate_dataset.py](KUKA-experiment/02_generate_dataset.py) that generates a dataset file ```KUKA-experiment/results/KUKA-surf-dataset/simdata.dat```. We are now ready to train the proposed learning methods.

<img src="/doc/images/pyBulletMovie.gif" alt="drawing" width="350"/>

#### Mean Absolute Error vs. Number of training points

In this experiment we compare the performance of a vanilla GP to the proposed Structured-GP w.r.t. to the number of training points used. Optionaly, an analytical model is used as a mean function to the S-GP.
Execute the following main scripts to train the models

- [03_train_S-GP.py](KUKA-experiment/03_train_S-GP.py)
- [03_train_GP.py](KUKA-experiment/03_train_GP.py)

At the top of each script, you find the path to the model configuration file. Please, make sure you find the following path
```python
cfg_model = importlib.import_module('results.KUKA-surf-dataset.exp_MAEvsTrainpoints.config_ML')
```
which points to [exp_MAEvsTrainpoints/config_ML.py](KUKA-experiment/results/KUKA-surf-dataset/exp_MAEvsTrainpoints/config_ML.py).

Inside the config_ML file, there is an option ```s_gp.use_Fa_mean = False```, that lets you change wheter the ```S-GP``` will use the analytical model or the zero mean as a mean function.
In addition, the option ```ds.datasetsize_train = 600``` lets you change the number of traning points to be used by the different ```GP``` approaches during training.


<img src="/doc/images/MAE-vs-nbr_training_samples.png" alt="drawing" width="350"/>
<img src="/doc/images/const_viol-vs-nbr_training_samples.png" alt="drawing" width="350"/>

<br>

#### Learning end-effector mass and CoG alongside unmodeled forces

Here, compare how the S-GP method performs when using a analytical multi-body dynamics model as a mean function, whose end-effector mass and CoG parameters are initially wrongly estimated. The S-GP approach aims at learning unmodeled friction forces alongside the analytical model parameters.

Execute the following main scripts to generate the results

- [03_train_S-GP_n_analytical-model.py](KUKA-experiment/03_train_S-GP_n_analytical-model.py)
- [03_train_analytical-model.py](KUKA-experiment/03_train_analytical-model.py)

and make sure the configuration file [exp_learn_massCoG_alongside/config_ML.py](KUKA-experiment/results/KUKA-surf-dataset/exp_learn_massCoG_alongside/config_ML.py) is set in the first lines of each code:
```python
cfg_model = importlib.import_module('results.KUKA-surf-dataset.exp_learn_massCoG_alongside.config_ML')
```

<img src="/doc/images/kin-param-learning-progress.png" alt="drawing" width="300"/>

<br>

#### Compare GP vs. S-GP vs Analytical Model vs. Neural Network

Execute the following main scripts to generate the results

- [03_train_S-GP.py](KUKA-experiment/03_train_S-GP.py)
- [03_train_GP.py](KUKA-experiment/03_train_GP.py)
- [03_train_NN.py](KUKA-experiment/03_train_NN.py)
- [03_run_analytical-model.py](KUKA-experiment/03_run_analytical-model.py)

and make sure each code points to the right configuration file [exp_comp_gp-sgp-nn-mbd/config_ML.py](KUKA-experiment/results/KUKA-surf-dataset/exp_comp_gp-sgp-nn-mbd/config_ML.py): 
```python
cfg_model = importlib.import_module('results.KUKA-surf-dataset.exp_comp_gp-sgp-nn-mbd.config_ML')
```

<img src="/doc/images/compare-approaches.png" alt="drawing" width="600"/>
<img src="/doc/images/compare-approaches-error.png" alt="drawing" width="600"/>


### Installation

This project is written 100% in python. The user is only required to install all python libraries used in this project.

General libraries

- [addict](https://github.com/mewwts/addict)

PyBullet simulations can only be executed in a pip or virtualenv environment (it does not support conda)

- [pybullet](https://github.com/bulletphysics/bullet3)

For training Gaussian Processes, please use a conda environment with the following packages

- [pytorch](https://pytorch.org/) >= 1.6
- [gpytorch](https://gpytorch.ai/)
- [pytorch_cluster](https://github.com/rusty1s/pytorch_cluster)


### External Softwares

#### urdf_parser_py

http://wiki.ros.org/urdfdom_py
https://github.com/ros/urdf_parser_py

License: BSD

Author(s): Thomas Moulard, David Lu, Kelsey Hawkins, Antonio El Khoury, Eric Cousineau, Ioan Sucan , Jackie Kay


#### KUKA robot arm model

https://github.com/bulletphysics/bullet3/tree/master/data/kuka_iiwa
https://github.com/bulletphysics/bullet3

License: Zlib
