'''
This file contains the user configurations for the KUKA experiment.
'''


import sys, os
import torch
import numpy as np
from enum import Enum, auto
import math
from scipy.spatial.transform import Rotation as R
from MBD_simulator_torch.classes.torch_utils import *
from addict import Dict

# import pybullet_data
# import pybullet as p
# p.setAdditionalSearchPath(pybullet_data.getDataPath())

pwd = os.path.dirname(__file__)
np.set_printoptions(precision=4,threshold=1000,linewidth=500,suppress=True)

class ControlMode(Enum):
	BULLET = auto()		# standard pyBullet robot control (not recommended) 
	JSIDC = auto()		# Joint-space inverse dynamics control
	TSIDC = auto()		# task-space inverse dynamics control


''' ---- Log config ----'''
log								= Dict()
log.saveSimResults 	    		= True
log.saveImages 		    		= False
log.showImages		    		= True
log.saveVideo		    		= False	
log.dpi				    		= 300 		# image resolution
log.resultsFolder       		= pwd		# main folder where results will be saved
log.addFolder  					= lambda filename: os.path.join(log.resultsFolder, filename)
log.resultsFileName_raw     	= log.addFolder('simdata_raw.dat')		# name of the raw simulation data
log.resultsFileName     		= log.addFolder('simdata.dat') 			# name of the dataset name (after processing the raw simulation data)


if not os.path.exists(log.resultsFolder):
    print(f'\nCreating new results folder {log.resultsFolder}\n')
    os.makedirs(log.resultsFolder, exist_ok=True)


''' ---- General kuka arm config ----'''
kuka 							= Dict()
kuka.urdf_filename              = os.path.join(pwd,'../../models/model-ext.urdf')
kuka.endEffectorName			= 'lbr_iiwa_link_7'
kuka.basePos            		= [0,0,0.05] # [0,0,1.2] # 
kuka.baseOrn            		= [0,0,0,1]
kuka.restPoses          		= np.array([ 85,  50, -10 , -100, -170, -25, -100]) * np.pi/180
kuka.lowerLimits				= [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05] 	# for null space IK
kuka.upperLimits				= [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]			# for null space IK
kuka.jointRanges				= [5.8, 4, 5.8, 4, 5.8, 4, 6]						# for null space IK


''' ------ Surface configuration (the one the Kuka arm is sliding its end-effector):
	constraint surface with equation:  fun = As @ x - b = 0
	fun_J is the Jacobian dfun/dx and fun_H is the Hessian matrix 
'''
surf							= Dict()
surf.path               		= os.path.join(pwd,'../../models/flatSurfaceLarge.urdf')
surf.pos                		= tuple([0,0.65,0.05])
surf.orn                		= tuple(R.from_euler('xyz',[0,0,math.pi/2]).as_quat()) 
surf.As							= np.array([[0,0,1]])
surf.bs							= np.array([surf.pos[2] + 5e-3])
surf.fun   						= lambda r: np.array(surf.As @ r - surf.bs) 		# constraint equation c(x)>=0
surf.fun_J 						= lambda r: np.array(surf.As)						# constraint Jacobian Jc(x)>=0
surf.fun_H 						= lambda r: np.zeros((3,3))							# constraint Hessian  Hc(x)>=0

# same as surf.* but implemented in pyTorch (TODO: merge both numpy and pytorch surface configurations)
surftorch						= Dict()
surftorch.pos                	= torch.tensor([0,0.65,0.05])
surftorch.orn                	= torch.from_numpy(R.from_euler('xyz',[0,0,math.pi/2]).as_quat()) 
surftorch.As					= torch.tensor([0,0,1.])
surftorch.bs					= torch.tensor(surftorch.pos[2] + 5e-3)
surftorch.fun   				= lambda r: binner(surftorch.As.to(r.device), r) - surftorch.bs.to(r.device) 		# constraint equation c(x)>=0
surftorch.fun_J 				= lambda r: surftorch.As.to(r.device).repeat(r.shape[0],1)							# constraint Jacobian Jc(x)>=0
surftorch.fun_H 				= lambda r: torch.zeros((1,3,3), device=r.device).repeat(r.shape[0],1,1)			# constraint Hessian  Hc(x)>=0


''' ---- config for the contact dynamics ----'''
contact_dynamics                   		= Dict()
contact_dynamics.baumgarte_wn      		= 10.
contact_dynamics.baumgarte_ksi     		= 1.
contact_dynamics.friction.viscous.mu_C  = 1.	# viscous friction coefficient


''' ---- Ground plane config (for visualization) ----'''
ground							= Dict()
ground.pos         				= [0,0,0]
ground.orn         				= [0, 0, 0, 1]
ground.path             		= os.path.join(pwd,'../../models/plane.urdf')

''' ---- Controller config ----'''
ctrl							= Dict()
ctrl.mode	            		= [ControlMode.JSIDC,ControlMode.TSIDC][1]
ctrl.useOrientation     		= True
ctrl.useNullSpaceIK     		= False
ctrl.wn_q 		 				= 100.0		# joint space controller parameters
ctrl.xi_q 		 				= 1.0
ctrl.wn_s 		 				= 100.0		# task space translational controller parameters
ctrl.xi_s 		 				= 1.0
ctrl.wn_r 		 				= 100.0		# task space rotational controller parameters
ctrl.xi_r 		 				= 1.0
ctrl.K_reg						= np.diag([0,10000,10000,0,10000,0,0])	# mull space controller gain
ctrl.F_r_FE             		= [0,0,0.1]
ctrl.Q_FE               		= [0,0,0,1.0]
ctrl.maxTorque 					= [320,320,176,176,110,40,40]

''' ---- camera config (for visualization) ----'''
camera 							= Dict()
camera.distance         		= 1.8 						
camera.yaw              		= 180
camera.pitch            		= -40.0
camera.targetPosition   		= [-0.012065, -0.004299, -0.3839992]

''' ---- trajectory planning config ----'''
trajplan 						= Dict()
trajplan.ddx_max        		= 0.5
trajplan.numTrajPoints  		= 20 # 20, 300             # number of trajectory points to sample
trajplan.dtRange        		= np.array([0.5,1.0])       # range for the traveling time (point-to-point)
trajplan.xy_Box       			= np.array([				# x-y area where the end-effector will 
									[-0.5, 0.4], 			# lower-left corner
									[ 0.5, 0.6]				# top-right corner
								]) 

''' ---- general simulation configs ----'''
sim								= Dict()
sim.mode                		= [1,2][0]   	# 1=GUI, 2=without GUI
sim.freq                		= 240			# simulation time step = 1/sim_freq  # DO NOT change this value -> you might need to retune other simulation parameters
sim.gravity             		= [0,0,-9.8]
sim.realTime            		= False
sim.repeatSimulation 			= False

''' ---- debug configuration (checks if pyBullet simulation is performing as expected) ----'''
debug 							= Dict()
debug.DEBUG_MODE 				= False
debug.showMenu          		= True
debug.showTrail	        		= True
debug.trailDuration     		= 5    # [seconds]         # NOTE: simulation slows down if traiPeriod is small

''' ---- Dataset configuration ----'''
ds 								= Dict()
ds.datasetsize_train 			= 10000
ds.threshold_const_train   		= 0.05
ds.threshold_const_eval    		= 0.05
ds.ddq.noise.std 				= 0.1
