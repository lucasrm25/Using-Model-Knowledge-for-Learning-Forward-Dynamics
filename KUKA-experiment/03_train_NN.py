'''
This script trains a vanila feedforward neural network on the generated KUKA-surf dataset.
The NN inputs are the 7 joint positions, velocities and torques, while outputs are the 7 joint accelerations. 
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
    cfg_model   = importlib.import_module('results.KUKA-surf-dataset.exp_comp_gp-sgp-nn-mbd.config_ML')


with Tee(cfg_model.nn.addFolderAndPrefix('TrainingResults-log')):

    ''' ------------------------------------------------------------------------
    Load training data
    ------------------------------------------------------------------------ '''

    # load sate logs
    with open(cfg_dataset.log.resultsFileName, 'rb') as f:
        data = dill.load(f)

    # select choosen device if available
    device = torch.device("cuda") if torch.cuda.is_available() and cfg_model.nn.useGPU else torch.device("cpu")
    print(f'\nUsing device: {device}')

    # convert dataset to torch and move to right device
    dataset_train = data.dataset_train.to(device, dtype=torch.DoubleTensor)
    dataset_test  = data.dataset_test_list[0].to(device, dtype=torch.DoubleTensor)

    ''' ------------------------------------------------------------------------
    Create new NN model object and load parameters if they exist
    ------------------------------------------------------------------------ '''

    from torch import nn

    class NN(nn.Module):
        def __init__(self, nbrInputs:int=21, nbrHidenNeurons=[30,20], nbrOutputs:int=7, dataset:StructTorchArray=None):
            super().__init__()
            
            self.layers = nn.ModuleList()
            self.layers.append( nn.Linear(in_features=nbrInputs,  out_features=nbrHidenNeurons[0]) )
            for i in range(len(nbrHidenNeurons)-1):
                self.layers.append( nn.Sigmoid() )
                self.layers.append(  nn.Linear(in_features=nbrHidenNeurons[i],  out_features=nbrHidenNeurons[i+1]) )
            self.layers.append( nn.Sigmoid() )
            self.layers.append( nn.Linear(in_features=nbrHidenNeurons[-1], out_features=nbrOutputs) )

            if dataset is not None:
                feat = torch.cat([dataset.q,dataset.dq,dataset.tau], axis=-1)
                out  = dataset.ddq
                self.in_mean  = feat.mean(axis=0, keepdim=True)
                self.in_std   = feat.std(axis=0, keepdim=True)
                self.out_mean = out.mean(axis=0, keepdim=True)
                self.out_std  = out.std(axis=0, keepdim=True)

        
        def forward(self, dataset:StructTorchArray):
            feat = torch.cat([dataset.q,dataset.dq,dataset.tau], axis=-1)
            x = (feat - self.in_mean) / self.in_std
            for l in self.layers:
                x = l(x)
            ddq = (x * self.out_std) + self.out_mean
            return ddq

    model = NN(nbrInputs=21, nbrHidenNeurons=[80], nbrOutputs=7, dataset=dataset_train).to(device)

    # load hyperparameters if provided file exists 
    if os.path.isfile(cfg_model.nn.fileName) and not cfg_model.nn.trainFromScratch:
        state_dict = torch.load(cfg_model.nn.fileName, map_location=device)
        model.load_state_dict(state_dict)
        print('\nFound existing trained model! Loading parameters from this model!')
    elif cfg_model.nn.train:
        print('\nTraining from scratch!')
    else:
        raise Exception(f'No trained NN found at {cfg_model.nn.fileName}')


    if cfg_model.nn.train:

        ''' ------------------------------------------------------------------------
        Train
        ------------------------------------------------------------------------ '''

        sgp.cleanGPUcache()

        # set GP to training mode (prediction outputs prior)
        model.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=cfg_model.nn.lr)

        for i in range(cfg_model.nn.training_iterations):

            print(f'Iter {i+1:3d}/{cfg_model.nn.training_iterations} {" ":5s}', end='')


            # select batch
            idx_batch = np.random.choice(len(dataset_train), cfg_model.nn.batchsize)
            dataset_train_batch = dataset_train[idx_batch]

            ddq_pred = model(dataset_train_batch)

            loss = nn.functional.mse_loss( input=dataset_train_batch.ddq, target=ddq_pred, reduction='mean' )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'-mse: {loss.item():10.2f}')

            # restart optimizer every iterRestartOptimizer iterations 
            if not i % cfg_model.nn.iterRestartOptimizer and i>0:
                optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=cfg_model.nn.lr)
            
            ''' ------------------------------------------------------------------------
            Test model
            ------------------------------------------------------------------------ '''
            if not i % cfg_model.nn.display_every_x_iter and i>0:

                with torch.no_grad():
                    model.eval()
                    if data.contact:
                        ddq_pred = model(dataset_test)
                        MAE = torch.mean(torch.abs(ddq_pred - dataset_test.ddq), dim=0).cpu()
                        # MAE = nn.functional.l1_loss( input=dataset_test.ddq, target=ddq_pred, reduction='mean' ).cpu()
                        print(f'\nMAE_ddq_test = {MAE}\n')

                    model.train()

        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f'{name}, {param.data}\n')

        # sgp.printParameterList(model)
        sgp.cleanGPUcache()

        ''' ------------------------------------------------------------------------
        Save model
        ------------------------------------------------------------------------ '''

        if cfg_model.nn.saveModel:
            torch.save(model.state_dict(), cfg_model.nn.fileName)
            print(f'\nNN SAVED to {cfg_model.nn.fileName}\n')

''' ------------------------------------------------------------------------
Eval Prediction
------------------------------------------------------------------------ '''

if cfg_model.nn.eval:
    print('Evaluating model...')

    model.eval()

    with torch.no_grad():

        timeRange=[10,13]

        dataset = dataset_test
        ddq_pred = model(dataset)
        # t = dataset.t.cpu() - dataset.t[0].item()
        t = dataset.t.cpu() - dataset.t[0].item() if timeRange is None else dataset.t.cpu() - timeRange[0]

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
            fig1.savefig( cfg_model.nn.addFolderAndPrefix('evalPrediction.pdf'), dpi=cfg_model.log.dpi)
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
            # axs[j].set_ylim([-1,1])
        plt.tight_layout()
        # plt.show()

        if cfg_model.log.saveImages: 
            fig1.savefig( cfg_model.nn.addFolderAndPrefix('evalPredictionError.pdf'), dpi=cfg_model.log.dpi)
        if cfg_model.log.showImages: plt.show()

    
        # fig3 = evalConstraintSatisfaction( model, dataset_train, dataset_test )
        # if cfg_model.log.saveImages: fig3.savefig( cfg_model.nn.addFolderAndPrefix('ConstraintError.pdf'), dpi=cfg_model.log.dpi)
        # if cfg_model.log.showImages: plt.show()
        # print('...finished')

