'''
This script trains a (S-GP + analytical mean) model on the generated KUKA dataset.

At the beginning of the learning, we introduce an error on the end-effector mass and CoG.
We then keep track, to check if the proposed approach is able to learn at the same time
the errors coming from unmodeled (friction) forces and learn the model parameters.
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
import sgp.sgp as sgp
from utils.evalGPplots import *
from utils.Tee import Tee
from scipy.spatial.transform import Rotation as R
from MBD_simulator_torch.classes.RigidBody import *
from MBD_simulator_torch.classes.BodyOnSurfaceBilateralConstraint import *
from MBD_simulator_torch.classes.MultiRigidBody import *
from MBD_simulator_torch.classes.RotationalJoint import *
from addict import Dict

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

# load configuration module (get from arguments if provided)
if len(sys.argv) >=3:
    cfg_dataset = importlib.import_module(sys.argv[1])
    cfg_model   = importlib.import_module(sys.argv[2])
else:
    cfg_dataset = importlib.import_module('results.KUKA-surf-dataset.config_KUKA')
    cfg_model   = importlib.import_module('results.KUKA-surf-dataset.exp_kin.config_ML')


with Tee(cfg_model.s_gp_dyn.addFolderAndPrefix('TrainingResults-log')):

    ''' ------------------------------------------------------------------------
    Load training data
    ------------------------------------------------------------------------ '''

    # load state logs
    with open(cfg_dataset.log.resultsFileName, 'rb') as f:
        data = dill.load(f)

    # select choosen device if available
    device = torch.device("cuda") if torch.cuda.is_available() and cfg_model.s_gp_dyn.useGPU else torch.device("cpu")
    print(f'\nUsing device: {device}')

    # convert dataset to torch and move to right device
    dataset_train = data.dataset_train.to(device, dtype=torch.DoubleTensor)
    dataset_test  = data.dataset_test_list[0].to(device, dtype=torch.DoubleTensor)

    # reduce dataset size if required (sometimes needed to be able to fit in the memory)
    dataset_train = dataset_train[:cfg_model.ds.datasetsize_train]
    dataset_test  = dataset_test[:cfg_model.ds.datasetsize_test]

    ''' ------------------------------------------------------------------------
    Create new GP2 model object and load parameters if they exist
    ------------------------------------------------------------------------ '''
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

    dataset_train = sgp.get_KUKA_SGPMatrices_from_MDB(mbdKuka, dataset_train)
    dataset_test  = sgp.get_KUKA_SGPMatrices_from_MDB(mbdKuka, dataset_test)

    if hasattr(cfg_model.s_gp_dyn, 'lengthscale_constraint'):
        lengthscale_constraint = cfg_model.s_gp_dyn.lengthscale_constraint
    else:
        lengthscale_constraint = None

    model = sgp.SGPModel(
        dataset_train       = dataset_train,
        nq                  = data.nq,
        dt                  = data.dt,
        train_target_quant  = sgp.QUANT.ddq, # sgp.QUANT.Fz,
        standardize         = cfg_model.s_gp_dyn.standardize,
        use_Fa_mean         = cfg_model.s_gp_dyn.use_Fa_mean,
        # lengthscale_constraint = lengthscale_constraint,
        multiRigidBody      = mbdKuka,
        trainMBD            = cfg_model.s_gp_dyn.train_mbd # True
    ).to(device)


    '''------------------------------------------------------------------------
    Create error for some parameters that are meant to be learned alongside the GP 
    ------------------------------------------------------------------------'''
    
    # store initial hyper-parameter configuration
    learningHistory = Dict()
    learningHistory.params.S_r_SDs_6.name    = f'joint[6].S_r_SDs'
    learningHistory.params.S_r_SDs_6.true    = model.multiRigidBody.jointList[6].S_r_SDs().tolist()
    learningHistory.params.m_B_7.name    = f'joint[7].m_B'
    learningHistory.params.m_B_7.true    = model.multiRigidBody.linkList[7].m_B().tolist()

    # introduce initial bias in the mbd parameters
    with torch.no_grad():
        # TRUE jointList[6].S_r_SDs: [ 0.0000,  0.0000, -0.0200]
        model.multiRigidBody.jointList[6].S_r_SDs = ConstrainedParameter(
            torch.tensor([0.05,0.05,-0.05], device=device), 
            Interval( [-0.1,-0.1,-0.1], [0.1,0.1,0.1] ), 
            requires_grad=True
        )
        # TRUE linkList[7].m_B: 2. 
        model.multiRigidBody.linkList[7].m_B = ConstrainedParameter(torch.tensor(2.5, device=device), Interval(1.5, 3.), requires_grad=True)  

    '''------------------------------------------------------------------------
    Set initial GP parameters
    ------------------------------------------------------------------------'''
    def rand(size=(1,),lb=0.,ub=1.):
        res = np.random.rand(*size)*(ub-lb) + lb
        return torch.tensor(res).to(device)

    with torch.no_grad():
        model.likelihood.noise = torch.full_like(model.likelihood.noise, cfg_dataset.ds.ddq.noise.std**2)
        model.likelihood.raw_noise.requires_grad = False

        for j in range(data.nq):
            # model.covar_module.SE_covar_module[j].base_kernel.lengthscale = 2.
            # model.covar_module.SE_covar_module[j].outputscale = [1., 1., 1., 1., 1., 1., 0.001][j]
            model.covar_module.SE_covar_module[j].base_kernel.lengthscale = rand((1,21),lb=0.01,ub=20)
            model.covar_module.SE_covar_module[j].outputscale = rand(
                lb=[2., 2., 2., 2., 2., 2., 0.1][j], 
                ub=[0.01, 0.01, 0.01, 0.01, 0.01, 0.001, 0.00001][j]
            )

    # load hyperparameters if provided file exists
    if os.path.isfile(cfg_model.s_gp_dyn.fileName) and not cfg_model.s_gp_dyn.trainFromScratch:
        state_dict = torch.load(cfg_model.s_gp_dyn.fileName, map_location=device)
        model.load_state_dict(state_dict)
        print('\nFound existing trained model! Loading parameters from this model!')
        # load learning history
        learningHistory = dill.load( open(cfg_model.s_gp_dyn.addFolderAndPrefix('learningHistory'), 'rb')  )
        iter_init = max(learningHistory.nmll.keys()) +1
    elif cfg_model.s_gp_dyn.train:
        print('\nTraining from scratch!')
        iter_init = 0
    else:
        raise Exception(f'No trained GP found at {cfg_model.s_gp_dyn.fileName}')


    '''------------------------------------------------------------------------
    Store pointers for learning history
    ------------------------------------------------------------------------'''

    learningHistory.params.S_r_SDs_6.pointer = model.multiRigidBody.jointList[6].S_r_SDs
    learningHistory.params.m_B_7.pointer = model.multiRigidBody.linkList[7].m_B


    ''' ------------------------------------------------------------------------
    Train
    ------------------------------------------------------------------------ '''

    if cfg_model.s_gp_dyn.train:

        sgp.printParameterList(model)
        sgp.cleanGPUcache()

        # set GP to training mode (prediction outputs prior)
        model.train()

        # cfg_model.s_gp_dyn.lr = 0.2
        checkpoint = model.state_dict()

        # split parameter list in  1) gp2 hyper-parameters and  2) kinematic parameters
        params_mbd = list(model.multiRigidBody.parameters())
        params_gp = list(model.covar_module.parameters()) + list(model.mean_module.parameters()) + list(model.likelihood.parameters())
        
        # Init optimizers
        opt_gp = torch.optim.Adam([{'params': params_gp}], lr=cfg_model.s_gp_dyn.lr)
        opt_mbd = torch.optim.Adam([{'params': params_mbd}], lr=cfg_model.s_gp_dyn.lr)

        targetIter = iter_init + cfg_model.s_gp_dyn.training_iterations
        for i in range(iter_init, targetIter):
            
            print(f'Iter {i+1:3d}/{targetIter} {" ":5s}', end='')
            
            loss = - model.marginalLogLikelihood(verbose=True)

            # store history of model learning parameters
            learningHistory.nmll[i]       = loss.item()
            learningHistory.state_dict[i] = copy.deepcopy(model.state_dict())

            with torch.no_grad():
                print(f'-mll: {np.array(learningHistory.nmll[i]):10.2f}')
                
                for k, v in learningHistory.params.items():
                    v.history[i] = v.pointer().tolist()

                    print( f'\terror {v.name} = {v.true - np.array(v.history[i])}')

            opt_gp.zero_grad()
            opt_mbd.zero_grad()

            loss.backward(retain_graph=True)
            
            if cfg_model.s_gp_dyn.train_gp2: opt_gp.step()
            if cfg_model.s_gp_dyn.train_mbd: opt_mbd.step()            
            
            # eval on test dataset
            if not i % cfg_model.s_gp_dyn.display_every_x_iter and i>0:
                with torch.no_grad():
                    model.eval()
                    if data.contact:
                        MAE = torch.mean(torch.abs(model.ddq(dataset_test).mean - dataset_test.ddq), dim=0).cpu()
                    else:
                        MAE = torch.mean(torch.abs(model(quant=sgp.QUANT.ddqa, dataset_test=dataset_test).mean - dataset_test.ddqa), dim=0).cpu()
                    print(f'\nMAE_ddq_test = {MAE}\n')
                    learningHistory.MAE[i] = MAE.tolist()
                    model.train()

            # restart optimizer every iterRestartOptimizer iterations 
            if not i % cfg_model.s_gp_dyn.iterRestartOptimizer and i>0:
                opt_gp = torch.optim.Adam([{'params': params_gp}], lr=cfg_model.s_gp_dyn.lr)
                opt_mbd = torch.optim.Adam([{'params': params_mbd}], lr=cfg_model.s_gp_dyn.lr)   

        sgp.printParameterList(model)
        sgp.cleanGPUcache()

        ''' ------------------------------------------------------------------------
        Save model
        ------------------------------------------------------------------------ '''

        # save model parameters
        if cfg_model.s_gp_dyn.saveModel:
            torch.save(model.state_dict(), cfg_model.s_gp_dyn.fileName)
            print(f'\nGP SAVED to {cfg_model.s_gp_dyn.fileName}\n')
        
        # save learning history
        learningHistory = Dict(learningHistory)
        with open(cfg_model.s_gp_dyn.addFolderAndPrefix('learningHistory'), 'wb') as f:
            dill.dump(learningHistory, f)

        # do a quick performance evaluation on the test dataset and save results
        model.eval()
        perf_string, perf_dict = evalPredictionAccuracy(
            model, 
            dataset_test, 
            (sgp.QUANT.ddq|sgp.QUANT.dqn|sgp.QUANT.qn) if data.contact else (sgp.QUANT.ddqa|sgp.QUANT.dqan|sgp.QUANT.qan)
        )
        with open(cfg_model.s_gp_dyn.addFolderAndPrefix('TrainingResults-text'), 'w') as f:
            f.write(perf_string)
        torch.save(perf_dict, cfg_model.s_gp_dyn.addFolderAndPrefix('TrainingResults-dict'))



''' ------------------------------------------------------------------------
Eval learning progress
------------------------------------------------------------------------ '''

# update Dict recursively
learningHistory = Dict(learningHistory)

fig, ax = plt.subplots(3,1,figsize=(6,5),sharex=True)
# plot kin. parameter errors in %
ax[0].grid(True)
for k, v in learningHistory.params.items():
    param_error = np.array(list(v.history.values())) - v.true
    ax[0].plot( list(v.history.keys()), param_error, '--', label=k )
ax[0].legend(loc='upper left', bbox_to_anchor=(1.05, 1))
# plot nmll
ax[1].grid(True)
ax[1].plot( list(learningHistory.nmll.keys()), list(learningHistory.nmll.values()), '-', label='nmll' )
ax[1].legend(loc='upper left', bbox_to_anchor=(1.05, 1))
# MAE error range
MAE_v = np.array(list(learningHistory.MAE.values()))
MAE_it = list(learningHistory.MAE.keys())
ax[2].plot(MAE_it, MAE_v)
# ax[2].fill_between( 
#     MAE_it,
#     np.max(MAE_v, axis=1),
#     np.min(MAE_v, axis=1),
#     color='k', alpha=0.3, zorder=1,
#     label='MAE range'
# )
ax[2].legend([f'{i}' for i in range(data.nq)],loc='upper left', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
if cfg_model.log.saveImages: fig.savefig( cfg_model.s_gp_dyn.addFolderAndPrefix('learning-progress.pdf'), dpi=cfg_model.log.dpi)
if cfg_model.log.showImages: plt.show()


''' ------------------------------------------------------------------------
Eval Prediction
------------------------------------------------------------------------ '''

if cfg_model.s_gp_dyn.eval:
    print('Evaluating model...')

    model.eval()

    fig1, fig2 = plotPredictions(
        model, dataset_test, data.nq,
        (sgp.QUANT.ddq|sgp.QUANT.dqn|sgp.QUANT.qn) if data.contact else (sgp.QUANT.ddqa|sgp.QUANT.dqan|sgp.QUANT.qan)
    )
    if cfg_model.log.saveImages: 
        fig1.savefig( cfg_model.s_gp_dyn.addFolderAndPrefix('evalPrediction.pdf'), dpi=cfg_model.log.dpi)
        fig2.savefig( cfg_model.s_gp_dyn.addFolderAndPrefix('evalPredictionError.pdf'), dpi=cfg_model.log.dpi)
    if cfg_model.log.showImages: plt.show()
    
    fig3 = evalConstraintSatisfaction( model, dataset_train, dataset_test )
    if cfg_model.log.saveImages: fig3.savefig( cfg_model.s_gp_dyn.addFolderAndPrefix('ConstraintError.pdf'), dpi=cfg_model.log.dpi)
    if cfg_model.log.showImages: plt.show()
    print('...finished')

