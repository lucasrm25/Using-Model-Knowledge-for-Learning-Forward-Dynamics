
from matplotlib.markers import MarkerStyle
import numpy as np
import PIL
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn
import torch


seaborn.set_style("darkgrid")

# change pyplot format
params = {
    'legend.fontsize': 'large',
    #   'figure.figsize': (20,8),
    'axes.labelsize': 10.0,
    'axes.titlesize': 10.0,
    'figure.titlesize': 12.0,
    'xtick.labelsize': 10.0,
    'ytick.labelsize': 10.0,
    'axes.titlepad': 0,
    'legend.fontsize': 8.0,
    'legend.handlelength': 2
}
plt.rcParams.update(params)



save_figs = True
# folder_to_save = 'Apollo/simRes/KUKA-stribeck_friction'
folder_to_save = 'Apollo/simRes/KUKA-surf-dataset'



''' ------------------------------------------------------------------------------------------------- '''
''' --------------------------------- PLOT PREDICTION RESULTS --------------------------------------- '''
''' ------------------------------------------------------------------------------------------------- '''

# cfg_gp  = importlib.import_module('results.KUKA-surf-dataset.exp_kin_0.config_ML')
# cfg_nn  = importlib.import_module('results.KUKA-surf-dataset.exp_mbd.config_ML')
# cfg_mbd = importlib.import_module('results.KUKA-surf-dataset.exp_mbd.config_ML')





''' ------------------------------------------------------------------------------------------------- '''
''' ------------------- EVALUATE RELATION WITH NUMBER OF TRAINING SAMPLES --------------------------- '''
''' ------------------------------------------------------------------------------------------------- '''

# expMainPath = '/home/lucas/Desktop/11_Github/Masterthesis/project/Apollo/simRes/KUKA-stribeck_friction'
expMainPath = '/home/lucas/Desktop/11_Github/Masterthesis/project/Apollo/simRes/KUKA-surf-dataset'

samples_expPath_dict = {
    # 50:   os.path.join(expMainPath, 'exp_iter_0'),
    100:  os.path.join(expMainPath, 'exp_iter_1'),
    200:  os.path.join(expMainPath, 'exp_iter_2'),
    400:  os.path.join(expMainPath, 'exp_iter_3'),
    600:  os.path.join(expMainPath, 'exp_iter_4'),
    800:  os.path.join(expMainPath, 'exp_iter_5'),
    1000: os.path.join(expMainPath, 'exp_iter_6'),
}

models2analise = [
    # label,    model name
    # ['SSGP',    'SSGP-st1-muFa0-TrainingResults-dict'],
    ['GP',      'GP-st1-muFa0-TrainingResults-dict'],
    ['GP2',     'SGP-st1-muFa0-SpK0-TrainingResults-dict'],
    ['GP2-muF', 'SGP-st1-muFa1-SpK0-TrainingResults-dict'],
]

MAE_ddq = []
ConstrViol = []
for modelLabel, modelName in models2analise:
    MAE_ddq.append( { k:0. for k in samples_expPath_dict.keys() } )
    ConstrViol.append( { k:0. for k in samples_expPath_dict.keys() } )
    for nbrSamples, modelpath in samples_expPath_dict.items():
        res = torch.load( os.path.join( modelpath, modelName ) )
        MAE_ddq[-1][nbrSamples] = res.MAE['ddq'].cpu().numpy() 
        ConstrViol[-1][nbrSamples] = [res.ConstError['Max'], res.ConstError['Mean'] , res.ConstError['Min'] ]



''' PLOT MAE vs NUMBER OF TRAINING SAMPLES  '''

colors = ['tab:blue', 'tab:orange', 'tab:green']
markers = ['s', 'o', 'd']
labels = ['GP', 'S-GP', 'S-GP + analytical mean']
fig = plt.figure(figsize=(4,2.5))
# plt.grid(True)
for modelIdx, perf in enumerate(MAE_ddq):
    MAE = np.stack( list(perf.values()) ).T
    nbrsamples = np.tile( list(perf.keys()), [7,1] )
    for i in range(len(MAE)):
        plt.plot( nbrsamples[i], MAE[i], color=colors[modelIdx], lw=0.5)
    plt.scatter( nbrsamples.flatten(), MAE.flatten(), color=colors[modelIdx], marker=markers[modelIdx], edgecolors='k', label=labels[modelIdx], alpha=1. )
    plt.fill_between( 
        nbrsamples[0],
        np.max(MAE, axis=0),
        np.min(MAE, axis=0),
        color=colors[modelIdx], alpha=0.2, zorder=1
    )
plt.legend()
plt.ylabel('Mean absolute error')
plt.xlabel('number of training points')
plt.ylim([ 0.07, 1. ])
plt.yscale('log')
plt.tight_layout()
if save_figs: fig.savefig(os.path.join(folder_to_save,'MAE-vs-nbr_training_samples.pdf'))
plt.show()


''' PLOT CONSTRAINT ERROR vs NUMBER OF TRAINING SAMPLES  '''

colors = ['tab:blue', 'tab:orange', 'tab:green']
markers = ['s', 'o', 'd']
labels = ['GP', 'GP2', 'GP2+mean']
fig = plt.figure(figsize=(4,2.5))
# plt.grid(True)
for modelIdx, perf in enumerate(ConstrViol):
    constr_viol = np.stack( list(perf.values()) ).T
    nbrsamples = np.array(list(perf.keys())) #np.tile( list(perf.keys()), [3,1] )

    plt.plot( nbrsamples, constr_viol[1,:], color=colors[modelIdx], lw=0.5)
    plt.scatter( nbrsamples, constr_viol[1,:], color=colors[modelIdx], marker=markers[modelIdx], edgecolors='k', label=labels[modelIdx] )
    # plt.errorbar( nbrsamples, constr_viol[1,:], yerr=constr_viol[[0,2],:], fmt=markers[modelIdx], color=colors[modelIdx], ecolor='lightgray', elinewidth=3, capsize=0)

# plt.legend()
plt.ylabel('Abs. constraint violation mean') # \n E[$|A\;\hat{\ddot q} - b|$]
plt.xlabel('number of training points')
plt.yscale('log')
plt.tight_layout()
# if save_figs: fig.savefig(os.path.join(folder_to_save,'const_viol-vs-nbr_training_samples.pdf'))
plt.show()



''' ------------------------------------------------------------------------------------------------- '''
''' ---------------------------- EVALUATE LEARNING KIN PARAMETERS WITH GP --------------------------- '''
''' ------------------------------------------------------------------------------------------------- '''

import dill
import os, sys, importlib
sys.path.append( os.path.join( os.path.dirname(__file__), os.path.pardir ) )

cfg_kin       = importlib.import_module('results.KUKA-surf-dataset.exp_mbd.config_ML')
cfg_kin_n_gp2 = importlib.import_module('results.KUKA-surf-dataset.exp_kin_0.config_ML')

lh_kin = dill.load( open(cfg_kin.mbd.addFolderAndPrefix('learningHistory'), 'rb')  )
lh_kin_n_gp2 = dill.load( open(cfg_kin_n_sgp.s_gp_dyn.addFolderAndPrefix('learningHistory'), 'rb')  )

def getItemsFromLogHistory(dict_iter_array):
    return np.array(list(dict_iter_array.keys())), np.array(list(dict_iter_array.values()))


c1 = 'k'
# c2 = 'tab:red'
c2 = 'xkcd:sky blue'


fig, axs = plt.subplots(3,1,figsize=(4,5), sharex=True)
'''subplot1'''
it, m_B = getItemsFromLogHistory(lh_kin.params.m_B_7.history)
axs[0].plot( it, m_B - lh_kin.params.m_B_7.true, color=c1 )
it, m_B = getItemsFromLogHistory(lh_kin_n_sgp.params.m_B_7.history)
axs[0].plot( it, m_B - lh_kin_n_sgp.params.m_B_7.true, '-', color=c2 )
axs[0].set_ylabel('mass error')
axs[0].grid(True)
'''subplot2'''
it, S_r_SDs = getItemsFromLogHistory(lh_kin.params.S_r_SDs_6.history)
axs[1].plot( it, S_r_SDs - lh_kin.params.S_r_SDs_6.true, color=c1 )
it, S_r_SDs = getItemsFromLogHistory(lh_kin_n_sgp.params.S_r_SDs_6.history)
axs[1].plot( it, S_r_SDs - lh_kin_n_sgp.params.S_r_SDs_6.true, '-', color=c2)
axs[1].set_ylabel('CoG error')
axs[1].grid(True)
'''subplot3'''
it, MAE = getItemsFromLogHistory(lh_kin.MAE)
it, MAE = it[::3], MAE[::3,:]
it, MAE = np.tile(it,(7,1)), MAE.T
for i in range(len(MAE)):
    plt.plot( it[i], MAE[i], color=c1, lw=0.5)
plt.scatter( it.flatten(), MAE.flatten(), color=c1, marker='s', edgecolors='k' )
plt.fill_between( 
    it[0],
    np.max(MAE, axis=0),
    np.min(MAE, axis=0),
    color=c1, alpha=0.2, zorder=1
)
#
it, MAE = getItemsFromLogHistory(lh_kin_n_sgp.MAE)
it, MAE = it[::3], MAE[::3,:]
it, MAE = np.tile(it,(7,1)), MAE.T
for i in range(len(MAE)):
    plt.plot( it[i], MAE[i], color=c2, lw=0.5)
plt.scatter( it.flatten(), MAE.flatten(), color=c2, marker='o', edgecolors='k' )
plt.fill_between( 
    it[0],
    np.max(MAE, axis=0),
    np.min(MAE, axis=0),
    color=c2, alpha=0.4, zorder=1
)
axs[2].set_yscale('log')
axs[2].set_ylim([0.003,200])
axs[2].set_ylabel('Mean absolute error')
axs[2].grid(True)
axs[2].set_xlabel('iteration')
#
axs[0].set_xlim([0,400])
axs[0].legend(['analytical model','S-GP + analytical mean'], loc='upper right')
plt.tight_layout()
if save_figs: fig.savefig(os.path.join(folder_to_save,'kin-param-learning-progress.pdf'))
plt.show()

