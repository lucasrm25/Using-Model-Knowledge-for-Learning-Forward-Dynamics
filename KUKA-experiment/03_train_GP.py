'''
This script trains a traditional multi-task GP on the generated KUKA-surf dataset.
The GP inputs are the 7 joint positions, velocities and torques, while outputs are the 7 joint accelerations. 
'''

import os, sys, importlib
sys.path.append( os.path.join( os.path.dirname(__file__), os.path.pardir ) )
import gc
import math
import time
from collections import ChainMap, namedtuple
from datetime import date, datetime
from enum import Enum, auto
from typing import List
import dill
import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from tqdm import tqdm
from addict import Dict
import sgp.sgp as sgp
from utils.evalGPplots import *
from utils.Tee import Tee
from scipy.spatial.transform import Rotation as R
from MBD_simulator_torch.classes.RigidBody import *
from MBD_simulator_torch.classes.BodyOnSurfaceBilateralConstraint import *
from MBD_simulator_torch.classes.MultiRigidBody import *
from MBD_simulator_torch.classes.RotationalJoint import *

# setup print options
np.set_printoptions(precision=4,threshold=1000,linewidth=500,suppress=True)
torch.set_printoptions(precision=4,threshold=1000,linewidth=500)

# clean GPU cache
sgp.cleanGPUcache()

# ! VERY IMPORTANT: change torch to double precision
torch.set_default_tensor_type(torch.DoubleTensor)


''' ------------------------------------------------------------------------
Open config file
------------------------------------------------------------------------ '''

# load configuration module either from standard file or from file argument
if len(sys.argv) >=3:
    cfg_dataset = importlib.import_module(sys.argv[1])
    cfg_model   = importlib.import_module(sys.argv[2])
else:
    cfg_dataset = importlib.import_module('results.KUKA-surf-dataset.config_KUKA')
    cfg_model   = importlib.import_module('results.KUKA-surf-dataset.exp_MAEvsTrainpoints.config_ML')
    # cfg_model   = importlib.import_module('results.KUKA-surf-dataset.exp_comp_gp-sgp-nn-mbd.config_ML')



with Tee(cfg_model.gp.addFolderAndPrefix('TrainingResults-log')):

    ''' ------------------------------------------------------------------------
    Load training data
    ------------------------------------------------------------------------ '''

    # load sate logs
    with open(cfg_dataset.log.resultsFileName, 'rb') as f:
        data = dill.load(f)

    # select choosen device if available
    device = torch.device("cuda") if torch.cuda.is_available() and cfg_model.gp.useGPU else torch.device("cpu")
    print(f'\nUsing device: {device}')

    # convert dataset to torch and move to right device
    dataset_train = data.dataset_train.to(device, dtype=torch.DoubleTensor)
    dataset_test  = data.dataset_test_list[0].to(device, dtype=torch.DoubleTensor)

    # reduce dataset size if required (sometimes needed to be able to fit in the memory)
    dataset_train = dataset_train[:cfg_model.ds.datasetsize_train]
    dataset_test  = dataset_test[:cfg_model.ds.datasetsize_test]

    ''' ------------------------------------------------------------------------
    Create new GP model object and load parameters if they exist
    ------------------------------------------------------------------------ '''

    model = sgp.MultitaskGPModel(
        dataset_train = dataset_train,
        nq = data.nq,
        dt = data.dt,
        train_target_quant = sgp.QUANT.ddq,
        standardize = cfg_model.gp.standardize
    ).to(device)

    with torch.no_grad():
        model.likelihood.noise = torch.full_like(model.likelihood.noise, cfg_dataset.ds.ddq.noise.std**2)
    model.likelihood.raw_noise.requires_grad = False

    # load hyperparameters if provided file exists 
    if os.path.isfile(cfg_model.gp.fileName) and not cfg_model.gp.trainFromScratch:
        state_dict = torch.load(cfg_model.gp.fileName)
        model.load_state_dict(state_dict)
        print('\nFound existing trained model! Loading parameters from this model!')
    elif cfg_model.gp.train:
        print('\nTraining from scratch!')
    else:
        raise Exception(f'No trained GP found at {cfg_model.gp.fileName}')


    if cfg_model.gp.train:

        ''' ------------------------------------------------------------------------
        Train
        ------------------------------------------------------------------------ '''

        sgp.printParameterList(model)
        sgp.cleanGPUcache()

        # set GP to training mode (prediction outputs prior)
        model.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=cfg_model.gp.lr)

        for i in range(cfg_model.gp.training_iterations):
            print(f'Iter {i+1:3d}/{cfg_model.gp.training_iterations} {" ":5s}', end='')
            optimizer.zero_grad()
            loss = - model.marginalLogLikelihood(verbose=True)
            loss.backward()
            optimizer.step()
            print(f'-mll: {loss.item():10.2f}')

            # restart optimizer every iterRestartOptimizer iterations 
            if not i % cfg_model.gp.iterRestartOptimizer and i>0:
                optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=cfg_model.gp.lr)

            if not i % cfg_model.gp.display_every_x_iter and i>0:
                model.eval()
                with torch.no_grad():
                    if data.contact:
                        MAE = torch.mean(torch.abs(model.ddq(dataset_test).mean - dataset_test.ddq), dim=0).cpu()
                    else:
                        MAE = torch.mean(torch.abs(model(quant=sgp.QUANT.ddqa, dataset_test=dataset_test).mean - dataset_test.ddqa), dim=0).cpu()
                print(f'\nMAE_ddq_test = {MAE}\n')
                model.train()

        sgp.printParameterList(model)
        sgp.cleanGPUcache()

        ''' ------------------------------------------------------------------------
        Save model
        ------------------------------------------------------------------------ '''

        if cfg_model.gp.saveModel:
            torch.save(model.state_dict(), cfg_model.gp.fileName)
            print(f'\nGP SAVED to {cfg_model.gp.fileName}\n')

        # do a quick performance evaluation on the test dataset
        model.eval()
        perf_string, perf_dict = evalPredictionAccuracy(
            model, 
            dataset_test, 
            (sgp.QUANT.ddq|sgp.QUANT.dqn|sgp.QUANT.qn) if data.contact else (sgp.QUANT.ddqa|sgp.QUANT.dqan|sgp.QUANT.qan)
        )
        with open(cfg_model.gp.addFolderAndPrefix('TrainingResults-text'), 'w') as f:
            f.write(perf_string)
        torch.save(perf_dict, cfg_model.gp.addFolderAndPrefix('TrainingResults-dict'))


''' ------------------------------------------------------------------------
Eval
------------------------------------------------------------------------ '''

if cfg_model.gp.eval:
    print('Evaluating model...')
    model.eval()

    fig1, fig2 = plotPredictions(
        model, dataset_test, data.nq,
        sgp.QUANT.ddq,
        # (sgp.QUANT.ddq|sgp.QUANT.dqn|sgp.QUANT.qn) if data.contact else (sgp.QUANT.ddqa|sgp.QUANT.dqan|sgp.QUANT.qan),
        timeRange=[10,13]
    )
    if cfg_model.log.saveImages: 
        fig1.savefig( cfg_model.gp.addFolderAndPrefix('evalPrediction.pdf'), dpi=cfg_model.log.dpi)
        fig2.savefig( cfg_model.gp.addFolderAndPrefix('evalPredictionError.pdf'), dpi=cfg_model.log.dpi)
    if cfg_model.log.showImages: plt.show()
    
    fig3 = evalConstraintSatisfaction( model, dataset_train, dataset_test )
    if cfg_model.log.saveImages: fig3.savefig( cfg_model.gp.addFolderAndPrefix('ConstraintError.pdf'), dpi=cfg_model.log.dpi)
    if cfg_model.log.showImages: plt.show()
    print('...finished')

''' ------------------------------------------------------------------------
Eval Long-term Prediction
------------------------------------------------------------------------ '''


if cfg_model.gp.eval_LongTerm:
    print('Evaluating model (long-term predictions)...')

    idxDatasetToEval = 0
    dataset_longTerm = data.dataset_longTerm_list[idxDatasetToEval].to(device, dtype=torch.DoubleTensor)

    model.eval()
    mbdKuka = sgp.loadKUKA(
        urdfPath        = cfg_dataset.kuka.urdf_filename, 
        basePos         = cfg_dataset.kuka.basePos, 
        baseOrn         = cfg_dataset.kuka.baseOrn, 
        F_r_FE          = cfg_dataset.ctrl.F_r_FE, 
        gravity         = cfg_dataset.sim.gravity, 
        surf_fun        = cfg_dataset.surftorch.fun, 
        surf_fun_J      = cfg_dataset.surftorch.fun_J, 
        surf_fun_H      = cfg_dataset.surftorch.fun_H,
        endEffectorName = cfg_dataset.kuka.endEffectorName,
        baumgarte_wn    = cfg_dataset.contact_dynamics.baumgarte_wn,
        baumgarte_ksi   = cfg_dataset.contact_dynamics.baumgarte_ksi
    ).to(device)

    model.multiRigidBody = mbdKuka

    fig4, fig5 = evalLongTermPrediction(model, mbdKuka, dataset_longTerm)
    if cfg_model.log.saveImages: fig4.savefig( cfg_model.gp.addFolderAndPrefix('LongTermPred.pdf'), dpi=cfg_model.log.dpi)
    if cfg_model.log.saveImages: fig5.savefig( cfg_model.gp.addFolderAndPrefix('LongTermConstViolation.pdf'), dpi=cfg_model.log.dpi)
    if cfg_model.log.showImages: plt.show()
    print('...finished')

