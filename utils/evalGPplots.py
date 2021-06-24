import os, sys
sys.path.append( os.path.join( os.path.dirname(__file__), os.path.pardir ) )



import matplotlib.pyplot as plt
import numpy as np
import torch
from addict import Dict
from tqdm import tqdm
from utils.StructArray import StructNumpyArray, StructTorchArray
from sgp.sgp import QUANT, MultitaskMultivariateNormal, get_KUKA_SGPMatrices_from_MDB

torch.set_printoptions(precision=4,threshold=1000,linewidth=500)


def evalPredictionAccuracy(model, dataset, quant:QUANT):
    KPI = Dict()
    s = ''
    quantVec = [d for d in QUANT if bool(d & quant)]

    s += f'Performance evaluated on {len(dataset)} test points: \n\n'
    with torch.no_grad():
        
        # evaluate on training data
        preds = { q.name: model(q, dataset) for q in quantVec }

        KPI.RMSE = {
            q.name: torch.sqrt(torch.mean((preds[q.name].mean - dataset[q.name])**2,   dim=0)).cpu()
            for q in quantVec
        }
        s += f'Prediction performance: (RMSE)\n'
        s += ''.join([f'   {q.name:8s}: {KPI.RMSE[q.name]}\n' for q in quantVec])

        KPI.MAE = {
            q.name: torch.mean(torch.abs(preds[q.name].mean - dataset[q.name]),   dim=0).cpu()
            for q in quantVec
        }
        s += f'Prediction performance: (MAE)\n'
        s += ''.join([f'   {q.name:8s}: {KPI.MAE[q.name]}\n' for q in quantVec])

        KPI.ConfReg = {
            q.name: 
            torch.mean(
                torch.logical_and(
                    dataset[q.name] >= preds[q.name].confidence_region()[0],
                    dataset[q.name] <= preds[q.name].confidence_region()[1]
                ).double(), dim=0
            ).cpu()
            for q in quantVec
        }
        s += f'Uncertainty prediction performance: (% or predictions inside confidence region)\n'
        s += ''.join([f'   {q.name:8s}: {KPI.ConfReg[q.name]}\n' for q in quantVec])

        if bool(QUANT.ddq & quant):
            pred_ConstraintError = (
                torch.einsum('ncj,nj->nc',dataset.A, preds['ddq'].mean) - dataset.b
            ).cpu().numpy()
            KPI.ConstError = {
                'Mean': np.mean(np.abs(pred_ConstraintError)),
                'Min':  np.min(np.abs(pred_ConstraintError)),
                'Max':  np.max(np.abs(pred_ConstraintError)),
            }
            s += f'Constraint satisfaction performance: abs(A @ ddq - b)\n'
            s += ''.join([f'   {k:8s}: {v:.3e}\n' for k,v in KPI.ConstError.items()])
    print(s)
    return s, KPI

def plotPredictions(model, dataset, nq:int, quant:QUANT, timeRange=None):

    with torch.no_grad():
        
        # evaluate on training data
        quantVec = [d for d in QUANT if bool(d & quant)]
        preds = [ model(q, dataset) for q in quantVec ]

        # remove time offset to make plots nicer
        t = dataset.t.cpu() - dataset.t[0].item() if timeRange is None else dataset.t.cpu() - timeRange[0]

        ''' Plot comparison test dataset and prediction'''
        fig1, axs = plt.subplots(len(preds), nq, figsize=(20,9/3*len(preds)), sharex=False)
        axs = axs.reshape([len(preds),-1])
        for q in range(len(preds)):
            for j in range(nq):  
                axs[q,j].grid(True)
                if quantVec[q].name == 'ddq':
                    axs[q,j].set_title(f'$\ddot q_{{ {j} }}$')
                else:
                    axs[q,j].set_title(f'${quantVec[q].name}_{{ {j} }}$')
                axs[q,j].set_xlabel(f'time [s]')
                axs[q,j].plot( 
                    t, 
                    dataset[quantVec[q].name][:,j].cpu(), 
                    marker='', ls='--', lw=1, color='k', mfc=None, zorder=3
                )
                axs[q,j].plot( 
                    t, 
                    preds[q].mean[:,j].cpu(), 
                    marker='', ls='-', lw=1, color='purple', mfc=None, zorder=2
                )
                axs[q,j].fill_between( 
                    t.flatten(),
                    preds[q].confidence_region()[0][:,j].cpu(),
                    preds[q].confidence_region()[1][:,j].cpu(),
                    color='purple', alpha=0.3, zorder=1
                )
                if timeRange is None:
                    axs[q,j].set_xlim([min(t).item(),max(t).item()])
                else:
                    axs[q,j].set_xlim([0,timeRange[1]-timeRange[0]])
        plt.tight_layout()
        # plt.show()

        ''' Plot error between test dataset and prediction'''
        fig2, axs = plt.subplots(len(preds), nq, figsize=(20,9/3*len(preds)), sharex=False)
        axs = axs.reshape([len(preds),-1])
        for q in range(len(preds)):
            for j in range(nq):
                
                axs[q,j].grid(True)
                if quantVec[q].name == 'ddq':
                    axs[q,j].set_title(f'$\ddot q_{{ {j},pred }} - \ddot q_{{ {j},true }}$')
                else:
                    axs[q,j].set_title(f'${quantVec[q].name}_{{{j},pred}} - {quantVec[q].name}_{{{j},true}}$')
                axs[q,j].set_xlabel(f'time [s]')
                axs[q,j].plot( 
                    t, 
                    t * 0, 
                    marker='', ls='--', lw=1, color='k', mfc=None, zorder=3
                )
                axs[q,j].plot( 
                    t, 
                    preds[q].mean[:,j].cpu() - dataset[quantVec[q].name][:,j].cpu(), 
                    marker='', ls='-', lw=1, color='purple', mfc=None, zorder=2
                )
                axs[q,j].fill_between( 
                    t.flatten(),
                    preds[q].confidence_region()[0][:,j].cpu() - dataset[quantVec[q].name][:,j].cpu(),
                    preds[q].confidence_region()[1][:,j].cpu() - dataset[quantVec[q].name][:,j].cpu(),
                    color='purple', alpha=0.3, zorder=1
                )     
                if timeRange is None:
                    axs[q,j].set_xlim([min(t).item(),max(t).item()])
                else:
                    axs[q,j].set_xlim([0,timeRange[1]-timeRange[0]]) 
                # axs[q,j].set_ylim([-1,1])
        plt.tight_layout()
        # plt.show()
        return fig1, fig2

def evalConstraintSatisfaction(model, dataset_train, dataset_test):

    # predictions are ddq
    with torch.no_grad():
        # evaluate on training data
        pred_ddq_train = model.ddq(dataset_train)
        # evaluate on test data
        pred_ddq_test = model.ddq(dataset_test)

        train_ConstraintError = (
            torch.einsum('ncj,nj->nc',dataset_train.A, pred_ddq_train.mean) - dataset_train.b
        ).cpu().numpy()

        test_ConstraintError = (
            torch.einsum('ncj,nj->nc',dataset_test.A, pred_ddq_test.mean) - dataset_test.b
        ).cpu().numpy()

        # prediction variance
        train_constraint_var = (
            torch.einsum('naj,nji,nbi->nab',dataset_train.A, torch.diag_embed(pred_ddq_train.variance), dataset_train.A)
        )
        train_constraint_var = torch.diagonal(train_constraint_var, dim1=1, dim2=2).cpu().numpy()

        test_constraint_var = (
            torch.einsum('naj,nji,nbi->nab',dataset_test.A, torch.diag_embed(pred_ddq_test.variance), dataset_test.A)
        )
        test_constraint_var = torch.diagonal(test_constraint_var, dim1=1, dim2=2).cpu().numpy()


    fig, axs = plt.subplots(2, 1, figsize=(8,5))

    axs[0].grid(True)
    axs[0].fill_between(
        range(len(train_ConstraintError)),
        (train_ConstraintError + 2*np.sqrt(train_constraint_var)).flatten(),
        (train_ConstraintError - 2*np.sqrt(train_constraint_var)).flatten(),
        color='purple', alpha=0.3
    )
    axs[0].plot(range(len(train_ConstraintError)), train_ConstraintError, '-', color='purple', lw=1)
    axs[1].plot(range(len(train_ConstraintError)), 0*train_ConstraintError, '-', color='black', lw=0.5)
    axs[0].set_title('$\mu_e \pm 2 \sigma_e$')
    axs[0].set_xlabel('training dataset')
    axs[0].set_xlim([0, len(train_ConstraintError)-1])
    # 
    axs[1].grid(True)
    axs[1].fill_between(
        range(len(test_ConstraintError)),
        (test_ConstraintError + 2*np.sqrt(test_constraint_var)).flatten(),
        (test_ConstraintError - 2*np.sqrt(test_constraint_var)).flatten(),
        color='purple', alpha=0.3
    )
    axs[1].plot(range(len(test_ConstraintError)), test_ConstraintError, '-', color='purple', lw=1)
    axs[1].plot(range(len(test_ConstraintError)), 0*test_ConstraintError, '-', color='black', lw=0.5)
    axs[1].set_title('$\mu_e \pm 2 \sigma_e$')
    axs[1].set_xlabel('test dataset')
    axs[1].set_xlim([0, len(test_ConstraintError)-1])

    plt.suptitle('Constaint Satisfaction Analysis of the GP Predictions: $e = A \; \ddot q - b$')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()
    return fig

def evalLongTermPrediction(model, mbd, dataset_longTerm, ti=0., tf_pred=1., tf_true=2.):

    print('Calculating long-term predictions...')
    with torch.no_grad():

        idxPred = range(0, int(tf_pred/model.dt))
        idxTrue = range(0, int(tf_true/model.dt))

        device = dataset_longTerm.q.device
        log = StructTorchArray()

        q, dq = dataset_longTerm[idxPred[0]].q, dataset_longTerm[idxPred[0]].dq
        for i in tqdm(idxPred):

            tau = dataset_longTerm[i].tau
            curr_state = StructTorchArray( q=q, dq=dq, tau=tau ).to(device)

            # generate dynamical matrices 
            curr_state = get_KUKA_SGPMatrices_from_MDB( mbd, curr_state)

            ddq_dist  = model.ddq(curr_state)
            dq_dist   = model.dqn(curr_state)
            q_dist    = model.qn(curr_state)
            # lamb_dist = model(QUANT.lamb, curr_state)

            # store data in log
            log.cat(
                t = dataset_longTerm[i].t, 
                q_mu    = q_dist.mean,
                q_var   = q_dist.covariance_matrix.unsqueeze(0) + (log.q_var[-1] if len(log) else 0.),
                dq_mu   = dq_dist.mean,
                dq_var  = dq_dist.covariance_matrix.unsqueeze(0) + (log.dq_var[-1] if len(log) else 0.),
                ddq_mu  = ddq_dist.mean,
                ddq_var = ddq_dist.covariance_matrix.unsqueeze(0),
                # lamb_mu  = lamb_dist.mean,
                # lamb_var = lamb_dist.covariance_matrix.unsqueeze(0),
                tau     = tau,
                A       = curr_state.A,
                b       = curr_state.b,
                I_c     = curr_state.I_c
            )

            q, dq = q_dist.mean, dq_dist.mean
    print('...finished')

    # move everything to cpu
    dataset_longTerm = dataset_longTerm.to("cpu")
    log = log.to("cpu")
    # remove time offset to make plots look nicer
    t_true = (dataset_longTerm[idxTrue].t.numpy() - dataset_longTerm[idxTrue].t.numpy()[0]).squeeze()
    t_pred = (log.t.numpy() - log.t.numpy()[0]).squeeze()

    # PLOT LONG-TERM PREDICTION and COMPARE TO MEASURED TRAJECTORY

    nq = log.q_mu.shape[1]
    fig1, axs = plt.subplots(4,nq,sharex=True, figsize=(20,7))
    for i in range(nq):
        axs[0,i].grid(True)
        axs[0,i].set_title(f'$ddq_{i}$')
        # axs[0,i].set_xlabel(f'time [s]')
        axs[0,i].plot(
            t_true, dataset_longTerm[idxTrue].ddq[:,i].numpy(), 
            marker='', ls='--', lw=1, color='k', mfc=None, zorder=3
        )
        axs[0,i].plot(
            t_pred, log.ddq_mu[:,i], 
            marker='', ls='-', lw=1, color='purple', mfc=None, zorder=2 
        )
        axs[0,i].fill_between( 
            t_pred, 
            (log.ddq_mu[:,i] + 2*np.sqrt(log.ddq_var.diagonal(dim1=-1,dim2=-2)[:,i])).numpy(), 
            (log.ddq_mu[:,i] - 2*np.sqrt(log.ddq_var.diagonal(dim1=-1,dim2=-2)[:,i])).numpy(),
            color='purple', alpha=0.3, zorder=1
        )

        axs[1,i].grid(True)
        axs[1,i].set_title(f'$dq_{i}$')
        # axs[1,i].set_xlabel(f'time [s]')
        axs[1,i].plot(
            t_true, dataset_longTerm[idxTrue].dq[:,i],  
            marker='', ls='--', lw=1, color='k', mfc=None, zorder=3
        )
        axs[1,i].plot(
            t_pred, log.dq_mu[:,i], 
            marker='', ls='-', lw=1, color='purple', mfc=None, zorder=2 
        )
        axs[1,i].fill_between( 
            t_pred, 
            log.dq_mu[:,i] + 2*np.sqrt( log.dq_var.diagonal(dim1=-1,dim2=-2)[:,i] ), 
            log.dq_mu[:,i] - 2*np.sqrt( log.dq_var.diagonal(dim1=-1,dim2=-2)[:,i] ),
            color='purple', alpha=0.3, zorder=1
        )
        if np.ptp(axs[1,i].get_ylim()) < 0.03:
            axs[1,i].set_ylim( np.mean(axs[1,i].get_ylim()) + [-0.015,0.015] )

        axs[2,i].grid(True)
        axs[2,i].set_title(f'$q_{i}$')
        # axs[2,i].set_xlabel(f'time [s]')
        axs[2,i].plot(
            t_true, dataset_longTerm[idxTrue].q[:,i],  
            marker='', ls='--', lw=1, color='k', mfc=None, zorder=3
        )
        axs[2,i].plot(
            t_pred, log.q_mu[:,i], 
            marker='', ls='-', lw=1, color='purple', mfc=None, zorder=2 
        )
        axs[2,i].fill_between( 
            t_pred, 
            log.q_mu[:,i] + 2*np.sqrt( log.q_var.diagonal(dim1=-1,dim2=-2)[:,i] ), 
            log.q_mu[:,i] - 2*np.sqrt( log.q_var.diagonal(dim1=-1,dim2=-2)[:,i] ),
            color='purple', alpha=0.3, zorder=1
        )
        if np.ptp(axs[2,i].get_ylim()) < 0.05:
            axs[2,i].set_ylim( np.mean(axs[2,i].get_ylim()) + [-0.025,0.025] )

        axs[3,i].grid(True)
        axs[3,i].set_title(f'$tau_{i}$')
        axs[3,i].set_xlabel(f'time [s]')
        axs[3,i].plot(
            t_true, dataset_longTerm[idxTrue].tau.numpy()[:,i], 
            marker='', ls='--', lw=1, color='k', mfc=None, zorder=3
        )

    axs[0,0].set_xlim([ti,tf_true])
    plt.tight_layout()  
    # plt.show()


    fig2, axs = plt.subplots(3,1,figsize=(8,5),sharex=True)

    # CONSTRAINT VIOLATION IN THE ACCELERATION LEVEL (A*ddq-b=0)
    constViolation_mu = torch.einsum('bij,bj->bi', log.A, log.ddq_mu ) - log.b
    constViolation_var = torch.einsum('bij,bjk,blk->bil', log.A, log.ddq_var, log.A ).diagonal(dim1=-1,dim2=-2)

    axs[0].grid(True)
    axs[0].set_title(f'Prediction Constraint Violation - Acceleration level')
    axs[0].set_xlabel(f'time [s]')
    axs[0].set_ylabel(f'$A \ddot q - b$')
    axs[0].plot(
        t_pred, constViolation_mu.cpu(), 
        marker='', ls='-', lw=1, color='purple', mfc=None, zorder=2 
    )
    axs[0].fill_between( 
        t_pred, 
        (constViolation_mu.cpu() + 2*np.sqrt( 1e-8 + constViolation_var.cpu() )).squeeze(), 
        (constViolation_mu.cpu() - 2*np.sqrt( 1e-8 + constViolation_var.cpu() )).squeeze(),
        color='purple', alpha=0.3, zorder=1
    )

    # CONSTRAINT VIOLATION IN THE VELOCITY LEVEL (A*dq=0)
    constViolation_mu = torch.einsum('bij,bj->bi', log.A, log.dq_mu )
    constViolation_var = torch.einsum('bij,bjk,blk->bil', log.A, log.dq_var, log.A ).diagonal(dim1=-1,dim2=-2)

    axs[1].grid(True)
    axs[1].set_title(f'Prediction Constraint Violation - Velocity level')
    axs[1].set_xlabel(f'time [s]')
    axs[1].set_ylabel(f'$A \dot q$')
    axs[1].plot(
        t_pred, constViolation_mu.cpu(), 
        marker='', ls='-', lw=1, color='purple', mfc=None, zorder=2 
    )
    axs[1].fill_between( 
        t_pred, 
        (constViolation_mu.cpu() + 2*np.sqrt( 1e-8 + constViolation_var.cpu() )).squeeze(), 
        (constViolation_mu.cpu() - 2*np.sqrt( 1e-8 + constViolation_var.cpu() )).squeeze(),
        color='purple', alpha=0.3, zorder=1
    )
    
    axs[2].grid(True)
    axs[2].set_title(f'Prediction Constraint Violation - Position level')
    axs[2].set_xlabel(f'time [s]')
    axs[2].set_ylabel(f'$c(q)$')
    axs[2].plot(
        t_pred, log.I_c.cpu(), 
        marker='', ls='-', lw=1, color='purple', mfc=None, zorder=2 
    )

    # axs[3].grid(True)
    # axs[3].set_title(f'Lagrange multiplier')
    # axs[3].set_xlabel(f'time [s]')
    # axs[3].set_ylabel(f'$\lambda$')
    # axs[3].plot(
    #     t_pred, log.lamb_mu.cpu(), 
    #     marker='', ls='-', lw=1, color='purple', mfc=None, zorder=2 
    # )
    # axs[3].fill_between( 
    #     t_pred, 
    #     log.lamb_mu.cpu().squeeze() + 2*np.sqrt( 1e-8 + log.lamb_var.cpu().squeeze() ), 
    #     log.lamb_mu.cpu().squeeze() - 2*np.sqrt( 1e-8 + log.lamb_var.cpu().squeeze() ),
    #     color='purple', alpha=0.3, zorder=1
    # )

    axs[0].set_xlim([ti,tf_pred]) 
    plt.tight_layout()
    # plt.show()

    return fig1, fig2
