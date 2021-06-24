# Using Model Knowledge for Learning Forward Dynamics

Code related to the paper "Using Model Knowledge for Learning Forward Dynamics" submitted to the Conference on Robot Learning (CoRL) 2021.

Inspired on the paper "Learning Constrained Dynamics with Gauss’ Principle adhering Gaussian Processes" - A. Rene Geist and Sebastian Trimpe


### How to execute the experiments

Main python scripts are provided in the folder ```KUKA-experiment/```.

```KUKA-experiment/results/KUKA-surf-dataset/```

#### Generate Dataset


#### Mean Absolute Error vs. Number of training points

<img src="/doc/images/MAE-vs-nbr_training_samples.png" alt="drawing" width="350"/>
<img src="/doc/images/const_viol-vs-nbr_training_samples.png" alt="drawing" width="350"/>

<br>

#### Learning end-effector mass and CoG alongside unmodeled forces

<img src="/doc/images/kin-param-learning-progress.png" alt="drawing" width="300"/>

<br>

#### Compare GP vs. S-GP vs Analytical Model vs. Neural Network

<img src="/doc/images/compare-approaches.png" alt="drawing" width="600"/>
<img src="/doc/images/compare-approaches-error.png" alt="drawing" width="600"/>


### Installation

This project is written 100% in python. Required is to install all python libraries used in this project.

Basic libraries

- dill
- json
- tqdm
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

