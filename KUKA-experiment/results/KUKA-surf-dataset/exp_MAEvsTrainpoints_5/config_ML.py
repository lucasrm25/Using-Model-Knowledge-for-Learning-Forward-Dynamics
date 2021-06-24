'''

'''

import sys, os
import numpy as np
from enum import Enum, auto
from addict import Dict
import math
from scipy.spatial.transform import Rotation as R

np.set_printoptions(precision=4,threshold=1000,linewidth=500,suppress=True)

pwd = os.path.dirname(__file__)


log								= Dict()
log.saveImages 		    		= True
log.showImages		    		= False
log.saveVideo		    		= False
log.dpi				    		= 600
log.resultsFolder       		= pwd
log.addFolder  					= lambda filename: os.path.join(log.resultsFolder, filename)

if not os.path.exists(log.resultsFolder):
    print(f'\nCreating new results folder {log.resultsFolder}\n')
    os.makedirs(log.resultsFolder, exist_ok=True)


ds 								= Dict()
ds.datasetsize_train 			= 800
ds.datasetsize_test             = 1000

gp							= Dict()
gp.train                      = True
gp.eval                       = True
gp.eval_LongTerm              = False
gp.useGPU 					= True
gp.saveModel 					= True
gp.standardize            	= True
gp.trainFromScratch			= False
gp.training_iterations 		= 500
gp.iterRestartOptimizer       = 100
gp.display_every_x_iter 		= 20
gp.lr 						= 0.2
gp.addFolderAndPrefix 		= lambda filename: log.addFolder(f'GP-st{int(gp.standardize)}-muFa0-' + filename)
gp.fileName 					= gp.addFolderAndPrefix('hyperparams.sgp')

s_gp						    = Dict()
s_gp.train                      = True
s_gp.eval                       = True
s_gp.eval_LongTerm              = False
s_gp.eval_extraQuant            = False
s_gp.useGPU 					= True
s_gp.saveModel 				    = True
s_gp.standardize            	= True
s_gp.use_Fa_mean            	= False

s_gp.trainFromScratch			= False
s_gp.training_iterations 		= 500
s_gp.iterRestartOptimizer       = 100
s_gp.display_every_x_iter 	    = 20
s_gp.lr 						= 0.2
s_gp.addFolderAndPrefix 		= lambda filename: log.addFolder(f'SGP-st{int(s_gp.standardize)}-muFa{int(s_gp.use_Fa_mean)}-' + filename)
s_gp.fileName 				    = s_gp.addFolderAndPrefix('hyperparams.sgp')


s_gp_dyn						= Dict()
s_gp_dyn.train_mbd            = True
s_gp_dyn.train_gp2            = True
s_gp_dyn.train                = True
s_gp_dyn.eval                 = True
s_gp_dyn.eval_LongTerm        = False
s_gp_dyn.eval_extraQuant		= False
s_gp_dyn.useGPU 				= True
s_gp_dyn.saveModel 			= True
s_gp_dyn.standardize          = True
s_gp_dyn.use_Fa_mean          = False
s_gp_dyn.trainFromScratch		= False
s_gp_dyn.training_iterations 	= 500
s_gp_dyn.iterRestartOptimizer = 100
s_gp_dyn.display_every_x_iter = 20
s_gp_dyn.lr 					= 0.2
s_gp_dyn.addFolderAndPrefix 	= lambda filename: log.addFolder(f'SGP-DYN-st{int(s_gp_dyn.standardize)}-muFa{int(s_gp_dyn.use_Fa_mean)}-' + filename)
s_gp_dyn.fileName 			= s_gp_dyn.addFolderAndPrefix('hyperparams.sgp')
s_gp_dyn.MBDlearning 			= True

nn							    = Dict()
nn.train                        = True
nn.eval                         = True
nn.eval_LongTerm                = False
nn.eval_extraQuant              = True
nn.useGPU 					    = True
nn.saveModel 				    = True
nn.standardize          	    = True
nn.use_Fa_mean          	    = False
nn.trainFromScratch			    = False
nn.training_iterations 		    = 5000
nn.batchsize                    = 1000
nn.iterRestartOptimizer         = 100
nn.display_every_x_iter 	    = 10
nn.lr 						    = 0.01
nn.addFolderAndPrefix 		    = lambda filename: log.addFolder(f'NN-st{int(nn.standardize)}-muFa{int(nn.use_Fa_mean)}-' + filename)
nn.fileName 				    = nn.addFolderAndPrefix('hyperparams.sgp')
nn.MBDlearning 				    = True
