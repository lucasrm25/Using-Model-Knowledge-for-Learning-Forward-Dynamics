''' This script shows the performance of the Multi Rigid Body dynamical model.
Note that we assume that the dynamical model is not perfect, but lacks the
modelling of friction forces.
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
import sgp.sgp as sgp

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
    cfg_model   = importlib.import_module('results.KUKA-surf-dataset.exp_comp_gp-sgp-nn-mbd.config_ML')


with Tee(cfg_model.mbd.addFolderAndPrefix('TrainingResults-log')):

    ''' ------------------------------------------------------------------------
    Load training data
    ------------------------------------------------------------------------ '''

    # load sate logs
    with open(cfg_dataset.log.resultsFileName, 'rb') as f:
        data = dill.load(f)

    # select choosen device if available
    device = torch.device("cuda") if torch.cuda.is_available() and cfg_model.mbd.useGPU else torch.device("cpu")
    print(f'\nUsing device: {device}')

    # convert dataset to torch and move to right device
    dataset_train = data.dataset_train.to(device, dtype=torch.DoubleTensor)
    dataset_test  = data.dataset_test_list[0].to(device, dtype=torch.DoubleTensor)

    ''' ------------------------------------------------------------------------
    Create new multi-body dynamics model object
    ------------------------------------------------------------------------ '''

    from torch import nn

    class MBD(nn.Module):
        def __init__(self):
            super().__init__()
            
            self.mbd = sgp.loadKUKA(
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
        
        def forward(self, dataset:StructTorchArray):
            sgp.get_KUKA_SGPMatrices_from_MDB(self.mbd, dataset)
            F =  dataset.f + dataset.g + dataset.tau
            ddqa = torch.einsum('nab,nb->na', dataset.Minv, F )
            ddq = torch.einsum('nab,nb->na', dataset.L, dataset.b) + torch.einsum('nab,nb->na', dataset.T, ddqa)
            return ddq

    model = MBD().to(device)


''' ------------------------------------------------------------------------
Eval Prediction
------------------------------------------------------------------------ '''

if cfg_model.mbd.eval:
    print('Evaluating model...')


    model.eval()

    with torch.no_grad():

        timeRange=[10,13]

        dataset = dataset_test
        ddq_pred = model(dataset)
        # t = dataset.t.cpu() - dataset.t[0].item()
        t = dataset.t.cpu() - dataset.t[0].item() if timeRange is None else dataset.t.cpu() - timeRange[0]


        MAE = torch.mean(torch.abs(ddq_pred - dataset.ddq), dim=0).cpu()
        print(f'\nMAE_ddq_test = {MAE}\n')

        ''' Plot prediction'''
        fig1, axs = plt.subplots(1, data.nq, figsize=(20,3), sharex=True)
        for j in range(data.nq):  
            axs[j].grid(True)
            axs[j].set_title(f'$\ddot q_{{ {j} }}$')
            axs[j].set_xlabel(f'time [s]')
            axs[j].plot( 
                t, 
                dataset.ddq[:,j].cpu(), 
                marker='', ls='--', lw=1, color='k', mfc=None, zorder=3
            )
            axs[j].plot( 
                t, 
                ddq_pred[:,j].cpu(), 
                marker='', ls='-', lw=1, color='purple', mfc=None, zorder=2
            )
            # axs[j].set_xlim([min(t).item(),max(t).item()])
            axs[j].set_xlim([0,timeRange[1]-timeRange[0]])
        plt.tight_layout()
        # plt.show()

        if cfg_model.log.saveImages: 
            fig1.savefig( cfg_model.mbd.addFolderAndPrefix('evalPrediction.pdf'), dpi=cfg_model.log.dpi)
        if cfg_model.log.showImages: plt.show()


        ''' Plot prediction error'''
        fig1, axs = plt.subplots(1, data.nq, figsize=(20,3), sharex=True)
        for j in range(data.nq):  
            axs[j].grid(True)
            axs[j].set_title(f'$\ddot q_{{ {j},pred }} - \ddot q_{{ {j},true }}$')
            axs[j].set_xlabel(f'time [s]')
            axs[j].plot( 
                t, 
                t * 0, 
                marker='', ls='--', lw=1, color='k', mfc=None, zorder=3
            )
            axs[j].plot( 
                t, 
                ddq_pred[:,j].cpu() - dataset.ddq[:,j].cpu(), 
                marker='', ls='-', lw=1, color='purple', mfc=None, zorder=2
            )
            # axs[j].set_xlim([min(t).item(),max(t).item()])
            axs[j].set_xlim([0,timeRange[1]-timeRange[0]])

            # if j<=3:
            #     axs[j].set_ylim([-1,1])
            # else:
            #     axs[j].set_ylim([-4,4])

        plt.tight_layout()
        # plt.show()

        if cfg_model.log.saveImages: 
            fig1.savefig( cfg_model.mbd.addFolderAndPrefix('evalPredictionError.pdf'), dpi=cfg_model.log.dpi)
        if cfg_model.log.showImages: plt.show()

