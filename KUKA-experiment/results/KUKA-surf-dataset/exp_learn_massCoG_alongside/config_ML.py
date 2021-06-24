''' Configuration file for different learning approaches
'''

import os
import numpy as np
from addict import Dict

np.set_printoptions(precision=4,threshold=1000,linewidth=500,suppress=True)

pwd = os.path.dirname(__file__)

''' Logging config '''
log								= Dict()
log.saveImages 		    		= True
log.showImages		    		= False
log.saveVideo		    		= False
log.dpi				    		= 1000
log.resultsFolder       		= pwd
log.addFolder  					= lambda filename: os.path.join(log.resultsFolder, filename)

if not os.path.exists(log.resultsFolder):
    print(f'\nCreating new results folder {log.resultsFolder}\n')
    os.makedirs(log.resultsFolder, exist_ok=True)

''' Dataset config '''
ds 								= Dict()
ds.datasetsize_train 			= 600
ds.datasetsize_test             = 1000

''' Structured-GP config (mbd parameters are learned alongside)'''
s_gp_dyn						= Dict()
s_gp_dyn.train_mbd              = True
s_gp_dyn.train_sgp              = True
s_gp_dyn.train                  = True
s_gp_dyn.eval                   = True
s_gp_dyn.eval_LongTerm          = False
s_gp_dyn.eval_extraQuant		= False
s_gp_dyn.useGPU 				= True
s_gp_dyn.saveModel 			    = True
s_gp_dyn.standardize            = True
s_gp_dyn.use_Fa_mean            = False
s_gp_dyn.trainFromScratch		= False
s_gp_dyn.training_iterations 	= 500
s_gp_dyn.iterRestartOptimizer   = 100
s_gp_dyn.display_every_x_iter   = 20
s_gp_dyn.lr 					= 0.2
s_gp_dyn.addFolderAndPrefix 	= lambda filename: log.addFolder(f'SGP-DYN-st{int(s_gp_dyn.standardize)}-muFa{int(s_gp_dyn.use_Fa_mean)}-' + filename)
s_gp_dyn.fileName               = s_gp_dyn.addFolderAndPrefix('hyperparams.sgp_dyn')
s_gp_dyn.MBDlearning            = True


''' Analytical model (Multi Body Dynamics / mbd) config'''
mbd						        = Dict()
mbd.train                       = True
mbd.eval                        = True
mbd.useGPU 				        = True
mbd.saveModel 			        = True
mbd.standardize                 = True
mbd.trainFromScratch		    = False
mbd.training_iterations 	    = 500
mbd.batchsize                   = 200
mbd.iterRestartOptimizer        = 200
mbd.display_every_x_iter        = 20
mbd.lr 					        = 0.2
mbd.addFolderAndPrefix 	        = lambda filename: log.addFolder(f'MBD-' + filename)
mbd.fileName 			        = mbd.addFolderAndPrefix('hyperparams.mbd')
