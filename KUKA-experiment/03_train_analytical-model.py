''' This script trains the parameters of a differentiable Multi Rigid Body dynamical model 
using RMSE loss on the generated KUKA-surf dataset.

At the beginning of the learning, we introduce an error on the end-effector mass and CoG.
'''

import os, sys, importlib
sys.path.append( os.path.join( os.path.dirname(__file__), os.path.pardir ) )
import dill
import matplotlib.pyplot as plt
import numpy as np
import torch
import sgp.sgp as sgp
from utils.evalGPplots import *
from utils.Tee import Tee
from MBD_simulator_torch.classes.RigidBody import *
from MBD_simulator_torch.classes.BodyOnSurfaceBilateralConstraint import *
from MBD_simulator_torch.classes.MultiRigidBody import *
from MBD_simulator_torch.classes.RotationalJoint import *
import sgp.sgp as sgp
# from addict import Dict

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
    cfg_model   = importlib.import_module('results.KUKA-surf-dataset.exp_learn_massCoG_alongside.config_ML')


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
    Create new MBD model object and load parameters if they exist
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


    '''------------------------------------------------------------------------
    Create error for some parameters that are meant to be learned
    ------------------------------------------------------------------------'''
    # store initial hyper-parameter configuration

    learningHistory = Dict()

    learningHistory.params.S_r_SDs_6.name    = f'joint[6].S_r_SDs'
    learningHistory.params.S_r_SDs_6.true    = model.mbd.jointList[6].S_r_SDs().tolist()
    learningHistory.params.m_B_7.name    = f'joint[7].m_B'
    learningHistory.params.m_B_7.true    = model.mbd.linkList[7].m_B().tolist()

    with torch.no_grad():
        model.mbd.jointList[6].S_r_SDs = ConstrainedParameter(
            torch.tensor([0.05,0.05,-0.05], device=device), # [ 0.0000,  0.0000, -0.0200]
            Interval( [-0.1,-0.1,-0.1], [0.1,0.1,0.1] ), 
            requires_grad=True
        )
        model.mbd.linkList[7].m_B = ConstrainedParameter(torch.tensor(2.5, device=device), Interval(1.5, 3.), requires_grad=True)  


    '''------------------------------------------------------------------------
    Load Model if already exists
    ------------------------------------------------------------------------'''

    # load hyperparameters if provided file exists 
    if os.path.isfile(cfg_model.mbd.fileName) and not cfg_model.mbd.trainFromScratch:
        state_dict = torch.load(cfg_model.mbd.fileName, map_location=device)
        model.load_state_dict(state_dict)
        print('\nFound existing trained model! Loading parameters from this model!')
        # load learning history
        learningHistory = dill.load( open(cfg_model.mbd.addFolderAndPrefix('learningHistory'), 'rb')  )
        iter_init = max(learningHistory.mse.keys()) +1
    elif cfg_model.mbd.train:
        print('\nTraining from scratch!')
        iter_init = 0
    else:
        raise Exception(f'No trained model found at {cfg_model.mbd.fileName}')

    '''------------------------------------------------------------------------
    Store pointers for learning history
    ------------------------------------------------------------------------'''

    learningHistory.params.S_r_SDs_6.pointer = model.mbd.jointList[6].S_r_SDs
    learningHistory.params.m_B_7.pointer = model.mbd.linkList[7].m_B


    ''' ------------------------------------------------------------------------
    Train
    ------------------------------------------------------------------------ '''

    if cfg_model.mbd.train:

        sgp.cleanGPUcache()

        # switch to training mode (prediction outputs prior)
        model.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=cfg_model.mbd.lr)

        targetIter = iter_init + cfg_model.mbd.training_iterations
        for i in range(iter_init, targetIter):

            print(f'Iter {i+1:3d}/{targetIter} {" ":5s}', end='')


            # select batch
            idx_batch = np.random.choice(len(dataset_train), cfg_model.mbd.batchsize)
            dataset_train_batch = dataset_train[idx_batch]

            ddq_pred = model(dataset_train_batch)

            loss = nn.functional.mse_loss( input=dataset_train_batch.ddq, target=ddq_pred, reduction='mean' )

            # store history of model learning parameters
            learningHistory.mse[i] = loss.item()
            learningHistory.state_dict[i] = copy.deepcopy(model.state_dict())

            with torch.no_grad():
                print(f'mse: {np.array(learningHistory.mse[i]):10.2f}')
                
                for k, v in learningHistory.params.items():
                    v.history[i] = v.pointer().tolist()
                    print( f'\terror {v.name} = {v.true - np.array(v.history[i])}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # restart optimizer every iterRestartOptimizer iterations 
            if not i % cfg_model.mbd.iterRestartOptimizer and i>0:
                optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=cfg_model.mbd.lr)
            
            ''' ------------------------------------------------------------------------
            Test model
            ------------------------------------------------------------------------ '''
            if not i % cfg_model.mbd.display_every_x_iter and i>0:

               with torch.no_grad():
                    model.eval()
                    ddq_pred = model(dataset_test)
                    MAE = torch.mean(torch.abs(ddq_pred - dataset_test.ddq), dim=0).cpu()
                    print(f'\nMAE_ddq_test = {MAE}\n')
                    learningHistory.MAE[i] = MAE.tolist()
                    model.train()                

        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f'{name}, {param.data}\n')

        # sgp.printParameterList(model)
        sgp.cleanGPUcache()

        ''' ------------------------------------------------------------------------
        Save model
        ------------------------------------------------------------------------ '''

        if cfg_model.mbd.saveModel:
            torch.save(model.state_dict(), cfg_model.mbd.fileName)
            print(f'\nNN SAVED to {cfg_model.mbd.fileName}\n')

        # save learning history
        learningHistory = Dict(learningHistory)
        with open(cfg_model.mbd.addFolderAndPrefix('learningHistory'), 'wb') as f:
            dill.dump(learningHistory, f)


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
# plot mse
ax[1].grid(True)
ax[1].plot( list(learningHistory.mse.keys()), list(learningHistory.mse.values()), '-', label='mse' )
ax[1].legend(loc='upper left', bbox_to_anchor=(1.05, 1))
# MAE error range
MAE_v = np.array(list(learningHistory.MAE.values()))
MAE_it = list(learningHistory.MAE.keys())
ax[2].plot(MAE_it, MAE_v)
ax[2].legend([f'{i}' for i in range(data.nq)],loc='upper left', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
if cfg_model.log.saveImages: fig.savefig( cfg_model.mbd.addFolderAndPrefix('learning-progress.pdf'), dpi=cfg_model.log.dpi)
if cfg_model.log.showImages: plt.show()


''' ------------------------------------------------------------------------
Eval Prediction
------------------------------------------------------------------------ '''

if cfg_model.mbd.eval:
    print('Evaluating model...')

    model.eval()

    with torch.no_grad():

        dataset = dataset_test
        t = dataset.t.cpu() - dataset.t[0].item()
        ddq_pred = model(dataset)

        ''' Plot comparison test dataset and prediction'''
        fig1, axs = plt.subplots(1, data.nq, figsize=(20,3), sharex=True)
        for j in range(data.nq):  
            axs[j].grid(True)
            axs[j].set_title(f'$ddq_{{ {j} }}$')
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
            axs[j].set_xlim([min(t).item(),max(t).item()])
        plt.tight_layout()
        plt.show()

    if cfg_model.log.saveImages: 
        fig1.savefig( cfg_model.mbd.addFolderAndPrefix('evalPrediction.pdf'), dpi=cfg_model.log.dpi)
    if cfg_model.log.showImages: plt.show()