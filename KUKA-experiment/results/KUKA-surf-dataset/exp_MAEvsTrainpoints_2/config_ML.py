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
ds.datasetsize_train 			= 200
ds.datasetsize_test             = 1000

gp							    = Dict()
gp.train                        = False
gp.eval                         = True
gp.eval_LongTerm                = False
gp.useGPU 					    = True
gp.saveModel 					= True
gp.standardize            	    = True
gp.trainFromScratch			    = False
gp.training_iterations 		    = 500
gp.iterRestartOptimizer         = 100
gp.display_every_x_iter 		= 20
gp.lr 						    = 0.2
gp.addFolderAndPrefix 		    = lambda filename: log.addFolder(f'GP-st{int(gp.standardize)}-muFa0-' + filename)
gp.fileName 					= gp.addFolderAndPrefix('hyperparams.sgp')

s_gp							= Dict()
s_gp.train                      = True
s_gp.eval                       = True
s_gp.eval_LongTerm              = False
s_gp.eval_extraQuant            = False
s_gp.useGPU 					= True
s_gp.saveModel 				    = True
s_gp.standardize          	    = True
s_gp.use_Fa_mean          	    = False
s_gp.trainFromScratch			= False
s_gp.training_iterations 		= 500
s_gp.iterRestartOptimizer       = 100
s_gp.display_every_x_iter 	    = 20
s_gp.lr 						= 0.2
s_gp.addFolderAndPrefix 		= lambda filename: log.addFolder(f'SGP-st{int(s_gp.standardize)}-muFa{int(s_gp.use_Fa_mean)}-' + filename)
s_gp.fileName 				    = s_gp.addFolderAndPrefix('hyperparams.sgp')



