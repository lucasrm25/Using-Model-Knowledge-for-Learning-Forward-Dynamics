'''
This script simulates and controls a KUKA iiwa robot arm with the PyBullet library.

In this experiment, the robot's end effector slides on a flat surface, subject to friction forces.
The controller used is a task-space inverse dynamic controller. 
The trajectory is given by task space trapezoidal profile. Reachable task-space points are sampled on a flat surface
and the robot end-effector tries to connect these points with a straight line.

In order to have a better control of the contact dynamics, the pyBullet collison/contact dynamics are deactivated.
Instead, we run a self-developed multi-rigid body dynamics library in parallel to pyBullet that calculates the 
contact (normal and friction) forces acting on the robot. 

Simulation data (joint positions, velocities, accelerations and torques) is collected and saved.

All configurations can be changed in a separate configuration file (see code below).
NOTE: set cfg.sim.mode=2 in the configuration file to hide the visualization and speed up the simulation

Notation:
	Coodinate frames: 
		- R = Robot base frame
		- B = link CoM frame
		- F = link frame
		- I = inertial frame
		- P = frame of interest on end-effector to be controlled
	Transformations:
		- A  = transformation matrix, Q=quaternion
		- w  = angular velocity
		- dw = angular acceleration
		- r  = linear displacement
		- v  = linear velocity
		- a  = linear acceleration
		- Js = translational jacobian
		- Jr = rotational jacobian
	Quantities:
		- q, dq, ddq = generalized position, velocities and accelerations
		- g, f, tau, C = generalized gravitational, bias, control and joint friction forces
		- M = generalized mass matrix
	
	A_IB   = The rotational orientation of the body CoM B with respect to the inertial frame I
	R_r_IB = displacement from I to B in R coordinates
	I_Js_B = translational jacobian of frame B described in inertial frame I, such that:  I_v_B = I_Js_B @ dq
	I_Jr_B = rotational    jacobian of frame B described in inertial frame I, such that:  I_w_B = I_Jr_B @ dq
'''

import os, sys, importlib
sys.path.append( os.path.join( os.path.dirname(__file__), os.path.pardir ) )

import copy
import math
import dill
import time
from datetime import datetime
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p

from utils.bullet_utils import *
from utils.TrajectoryPlanning import TrajectoryPlanning
from utils.StructArray import StructTorchArray, StructNumpyArray
from MBD_simulator.classes.RigidBody import *
from MBD_simulator.classes.BodyOnSurfaceBilateralConstraint import *
from MBD_simulator.classes.MultiRigidBody import *
from MBD_simulator.classes.RotationalJoint import *
from addict import Dict

np.set_printoptions(precision=4,threshold=1000,linewidth=500,suppress=True)


''' ------------------------ LOAD CONFIGURATION FILE ------------------------------------- '''


# import configurations
cfg = importlib.import_module('results.KUKA-surf-dataset.config_KUKA')
# cfg = importlib.import_module('results.KUKA-stribeck_friction.config_KUKA')

# connect to pyBullet
clid = p.connect(cfg.sim.mode)


''' ------------------------ LOAD BODIES IN PYBULLET ------------------------------------- '''

# load plane
groundId = p.loadURDF(
	cfg.ground.path, 
	basePosition 	= cfg.ground.pos,
	baseOrientation = cfg.ground.orn,
	useFixedBase=True
)
# load constraining surface
surfId = p.loadURDF(
	cfg.surf.path, 
	basePosition 	= cfg.surf.pos,
	baseOrientation = cfg.surf.orn,
	useFixedBase=True
)
# load Kuka IIWA
kukaId = p.loadURDF(
	cfg.kuka.urdf_filename, 
	basePosition 	= cfg.kuka.basePos,
	baseOrientation = cfg.kuka.baseOrn,
	useFixedBase 	= True,
	flags		 	= p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
)
nq = p.getNumJoints(kukaId)
kukaEndEffectorIndex = nq-1

for link_idx in range(nq+1):
	# disable all nonlinear effects acting on the apollo joints
	p.changeDynamics(kukaId, link_idx, linearDamping=0.0, angularDamping=0.0,jointDamping=0.0, maxJointVelocity=1000)
	# disable collisions
	p.setCollisionFilterGroupMask(bodyUniqueId=kukaId, linkIndexA=link_idx, collisionFilterGroup=0, collisionFilterMask=0)


''' ------------------------ LOAD BODIES IN MULTI RIGID BODY DYNAMICS LIBRARY --------------------- '''

mbdKuka = MultiRigidBody.fromURDF(
	robotFileName   = cfg.kuka.urdf_filename, 
	basePosition    = cfg.kuka.basePos,
	baseOrientation = R.from_quat(cfg.kuka.baseOrn).as_matrix(),
	I_grav			= np.array(cfg.sim.gravity) 
)
mbdEE = mbdKuka.linkMap['lbr_iiwa_link_7']
surfaceConstraint = BodyOnSurfaceBilateralConstraint(
	predBody	  = mbdEE,
	P_r_PDp 	  = cfg.ctrl.F_r_FE - mbdEE.parentJoint.A_SDs.T @ -mbdEE.parentJoint.S_r_SDs,
	surface_fun   = cfg.surf.fun,
	surface_fun_J = cfg.surf.fun_J,
	surface_fun_H = cfg.surf.fun_H,
	wn=cfg.contact_dynamics.baumgarte_wn, ksi=cfg.contact_dynamics.baumgarte_ksi 	# Baumgarte stabilization
)
mbdKuka.bilateralConstraints = [surfaceConstraint]
mbdKuka.forwardKinematics()
mbdKuka.getJointStates()
mbdKuka.printKinTree()
# mbdKuka.initGraphics(width=1600, height=1200, range=1.5, title='MBD Kuka', updaterate=60)
# mbdKuka.updateGraphics()



''' ------------------------ PREPARE SIMULATION -------------------------------- '''

# init controller
jointSpaceController = InverseDynControl(
	kukaId,
    endEffectorLinkIndex= kukaEndEffectorIndex, 
	F_r_FE = cfg.ctrl.F_r_FE,  # this is the farthest point of collision of the end-effector (obtained from CAD)
	Q_FE   = cfg.ctrl.Q_FE,
	restPoses	= cfg.kuka.restPoses,
	lowerLimits	= cfg.kuka.lowerLimits,
	upperLimits	= cfg.kuka.upperLimits,
	jointRanges	= cfg.kuka.jointRanges,
	wn_q = cfg.ctrl.wn_q, xi_q = cfg.ctrl.xi_q,
	wn_s = cfg.ctrl.wn_s, xi_s = cfg.ctrl.xi_s,
	wn_r = cfg.ctrl.wn_r, xi_r = cfg.ctrl.xi_r,
	K_reg = cfg.ctrl.K_reg,
	multiRigidBody = mbdKuka
)

# disable Bullet position and velocity default controls
if cfg.ctrl.mode in [cfg.ControlMode.JSIDC, cfg.ControlMode.TSIDC]:
	p.setJointMotorControlArray(kukaId, range(nq), controlMode=p.VELOCITY_CONTROL, forces=[0]*nq)

# read Joint information from URDF file
jointsInfo = getJointsInfo(kukaId)

# joint torque limits are the minimum between CTRL config and URDF specsheet limits
jointsInfo.jointMaxForce[:] = np.expand_dims([np.min(i) for i in zip(jointsInfo.jointMaxForce.flatten(), cfg.ctrl.maxTorque)], axis=-1)

# set joint angles to restposes
for i in range(nq):
	p.resetJointState(kukaId, i, cfg.kuka.restPoses[i])

# set simulation parameters
p.setGravity(*cfg.sim.gravity)
p.setRealTimeSimulation(cfg.sim.realTime)
p.setTimeStep(1./cfg.sim.freq)

# enable joint sensors
for j in range(nq):
	p.enableJointForceTorqueSensor(bodyUniqueId=kukaId, jointIndex=j, enableSensor=True)

# set camera position
p.resetDebugVisualizerCamera(	# yaw,pitch,dist,target = p.getDebugVisualizerCamera()[-4:]
	cameraDistance 		 = cfg.camera.distance,
	cameraYaw	   		 = cfg.camera.yaw,
	cameraPitch	   		 = cfg.camera.pitch,
	cameraTargetPosition = cfg.camera.targetPosition
)

# hide debugger menu
if not cfg.debug.showMenu or cfg.log.saveVideo:
	p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)	

# start recording video
if cfg.log.saveVideo:  
	logMP4 = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, cfg.log.addFolder("ApolloMovie.mp4"))

# Add debug parameters
debugParamsId = {
	'wn_q':  	p.addUserDebugParameter('Controller [JSIDC] - wn',   0, 500, jointSpaceController.wn_q ),
	'xi_q': 	p.addUserDebugParameter('Controller [JSIDC] - xi',   0,   2, jointSpaceController.xi_q ),
	'wn_s':  	p.addUserDebugParameter('Controller [TSIDC] - wn_s', 0, 500, jointSpaceController.wn_s ),
	'xi_s': 	p.addUserDebugParameter('Controller [TSIDC] - xi_s', 0,   2, jointSpaceController.xi_s ),
	'wn_r':  	p.addUserDebugParameter('Controller [TSIDC] - wn_r', 0, 500, jointSpaceController.wn_r ),
	'xi_r': 	p.addUserDebugParameter('Controller [TSIDC] - xi_r', 0,   2, jointSpaceController.xi_r ),
	'taugain': 	p.addUserDebugParameter('Overall controller gain',   0,   1, 1 ),
	'tcx':  	p.addUserDebugParameter('Trajectory center X ',     -1,   1, 0 ),
	'tcy':  	p.addUserDebugParameter('Trajectory center Y ',      0,   1, 0.4 ),
	'tcz':  	p.addUserDebugParameter('Trajectory center Z ',      0,   2, cfg.surf.pos[2]+0.01 ),
	'tf':		p.addUserDebugParameter('Trajectory freq [Hz]',      0,  10, 1.0 )	
}

# prepare data logger
statelogs = StructTorchArray()


''' ------------------------ PREPARE TRAJECTORY -------------------------------- '''

trajplan = TrajectoryPlanning(ddx_max=cfg.trajplan.ddx_max)

ti = 0.0
xi = np.array(jointSpaceController.getEndEffectorState()[0])
for i in range(cfg.trajplan.numTrajPoints):
	# sample the time that will take from going from xi to xf
	dt = (np.random.rand(1)*(cfg.trajplan.dtRange[1]-cfg.trajplan.dtRange[0]) + cfg.trajplan.dtRange[0]).item()
	# sample x an y coordinates of trajectory point randomly inside a 'xy_Box'
	xf_x, xf_y = np.random.rand(2)*(cfg.trajplan.xy_Box[1,:]-cfg.trajplan.xy_Box[0,:]) + cfg.trajplan.xy_Box[0,:]
	# the z coordinate is obtained from the intersection of the surface plane with the sampled (x,y) position
	xf_z = (cfg.surf.bs[0] - cfg.surf.As[0,:2] @ [xf_x,xf_y] ) / cfg.surf.As[0,-1] - 1e-4
	xf = np.array([xf_x, xf_y, xf_z])
	# append new target point
	trajplan.add(ti,ti+dt,xi,xf)
	ti=ti+dt
	xi=xf

# plot generated trajectory
fig = trajplan.plot_trajectory()
if cfg.log.saveImages: fig.savefig( cfg.log.addFolder('planned-OpSp-trajectory.pdf'), dpi=cfg.log.dpi)
if cfg.log.showImages: plt.show()


# plot trajectory projected on the surface
fig = plt.figure(figsize=(5,4))
plt.plot(trajplan.xi[:,0],trajplan.xi[:,1], color='grey', ls='-', lw=2, mfc='b', mec='b', marker='o')
plt.plot( 
	[cfg.trajplan.xy_Box[0,0],cfg.trajplan.xy_Box[0,0],cfg.trajplan.xy_Box[1,0],cfg.trajplan.xy_Box[1,0],cfg.trajplan.xy_Box[0,0]],
	[cfg.trajplan.xy_Box[0,1],cfg.trajplan.xy_Box[1,1],cfg.trajplan.xy_Box[1,1],cfg.trajplan.xy_Box[0,1],cfg.trajplan.xy_Box[0,1]],
	'--k',lw=2
)
plt.xlabel('x-position')
plt.ylabel('y-position')
plt.title('Projected trajectory on the surface')
plt.axis('equal')
plt.grid(True)
plt.tight_layout()
if cfg.log.saveImages: fig.savefig( cfg.log.addFolder('planned-OpSp-trajectory-projected-surf.pdf'), dpi=cfg.log.dpi)
if cfg.log.showImages: plt.show()



''' ------------------------ INIT SIMULATION -------------------------------- '''

# auxiliary variables
targetLine_old = jointSpaceController.getEndEffectorState()[0]
currentPos_old = targetLine_old
I_v_E_old      = np.zeros(3)
dq_old 	       = np.zeros(nq)
tau    	       = np.zeros(nq)
sim_t 		   = 0 if not cfg.sim.realTime else datetime.now().second / 60.	# simulation time
sim_k 		   = 0	# simulation time step
contacts	   = 0

while sim_t < trajplan.tf[-1] or cfg.sim.repeatSimulation:

	''' ------- Measure joint positions, velocities, and accelerations -------------'''

	# read measurements (accelerations are no available in pyBullet)
	jointStates = p.getJointStates(kukaId, range(nq))
	q, dq, reacTorque, appliedTorque = np.array(list(zip(*jointStates)))
	# convert type object to type float 
	q, dq = q.astype(np.double), dq.astype(np.double)
	# estimate joint accelerations
	ddq = (dq - dq_old)*cfg.sim.freq
	dq_old = copy.deepcopy(dq)


	# query contacts
	I_r_IE, I_v_E, Q_IE, I_w_E = jointSpaceController.getEndEffectorState()
	contacts = contacts or cfg.surf.fun(I_r_IE)[0] <= 0
	# contacts = len(p.getContactPoints(bodyA=kukaId,bodyB=surfId,linkIndexA=kukaEndEffectorIndex))

	''' ----- 
	If in debug mode, check if all the pyBullet results are matching with the MBD library that is running in parallel
		ddq 				   = acceleration read from pyBullet by doing a time step
		ddq_bullet (for debug) = acceleration calculated with inverse dynamics and mass 
		ddq_mbd (for debug)    = acceleration calculated by multi-rigid body dynamics (MBD) library
	----'''
	if sim_k > 1 and cfg.debug.DEBUG_MODE:
		ddq_f 	  	 = [f'{i:8.2f}' for i in ddq]
		ddq_bullet_f = [f'{i:8.2f}' for i in ddq_bullet]
		ddq_mbd_f 	 = [f'{i:8.2f}' for i in ddq_mbd]
		print(f' {int(contacts)}  ddq_UKE = {ddq_bullet_f}\n    ddq_sim = {ddq_f}\n    ddq_mbd = {ddq_mbd_f}')
		print(f'  M error: {np.sum(M2-M)}')
		print(f'  eta error: {np.sum(eta2-eta)}')
		print(f'  Constraint error: {(A @ ddq - b).item():.4f}')
		print(f'  lambda: {lambda_mbd}')
		print()
		# if not contacts:
		threshold = 0.001
		# assert np.sum(np.abs(M2-M)) < threshold and np.sum(np.abs(eta2-eta)) < threshold and np.sum(np.abs(ddq_UKE-ddq)) < threshold and np.sum(np.abs(ddq_mbd-ddq)) < threshold


	''' ------------------------- Query trajectory ------------------------------'''

	# query desired end-effector liner position, velocities and accelerations	
	I_r_IEd, I_v_Ed, I_a_Ed = trajplan(sim_t)

	# desired end-effector orientation in quaternions, angular velocity and angular acceleration
	Q_IEd   = np.array(p.getQuaternionFromEuler([0, -math.pi, 0]))
	I_w_Ed  = np.array([0,0,0])
	I_dw_Ed = np.array([0,0,0])


	''' --------------------- Calculate control input -------------------------'''

	# if GUI is available, use parameters provided by the debug parameter list
	torqueGain = 1
	if p.getConnectionInfo()['connectionMethod'] == p.GUI:
		jointSpaceController.wn_q  = p.readUserDebugParameter(debugParamsId['wn_q'])
		jointSpaceController.xi_q  = p.readUserDebugParameter(debugParamsId['xi_q'])
		jointSpaceController.wn_s  = p.readUserDebugParameter(debugParamsId['wn_s'])
		jointSpaceController.xi_s  = p.readUserDebugParameter(debugParamsId['xi_s'])
		jointSpaceController.wn_r  = p.readUserDebugParameter(debugParamsId['wn_r'])
		jointSpaceController.xi_r  = p.readUserDebugParameter(debugParamsId['xi_r'])
		torqueGain = p.readUserDebugParameter(debugParamsId['taugain'])


	q_des, dq_des, ddq_des = jointSpaceController.inverseDiffKinematics(
		I_r_IEd, I_v_Ed, I_a_Ed, 
		Q_IEd, I_w_Ed, I_dw_Ed, 
		useOrientation=cfg.ctrl.useOrientation,
		useNullSpaceIK=cfg.ctrl.useNullSpaceIK
	)

	if cfg.ctrl.mode == cfg.ControlMode.JSIDC:		
		tau = jointSpaceController.jointSpaceControl(q_des, dq_des, ddq_des)

	elif cfg.ctrl.mode == cfg.ControlMode.TSIDC:
		tau = jointSpaceController.taskSpaceControl(
			I_r_IEd, I_v_Ed, I_a_Ed, 
			Q_IEd, I_w_Ed, I_dw_Ed, 
			useOrientation=cfg.ctrl.useOrientation
		)
	else: raise NotImplementedError('Control mode specified has not been implemented yet')

	# if desired, apply an additional torque gain, given by user
	tau *= torqueGain

	# clip torques to actuation limits
	tau = np.clip(tau, -jointsInfo.jointMaxForce.flatten(), jointsInfo.jointMaxForce.flatten())



	''' -------------------------------- Debug -----------------------------------'''
	# add pybullet "debug line" that allows visualization of the end-effector trajetory

	# update debug line for the end-effector
	if cfg.sim.mode == p.GUI and sim_k > 0 and not sim_k % cfg.debug.trailDuration and cfg.debug.showTrail:
		targetPos  = I_r_IEd
		currentPos = I_r_IE
		# debug line for the desired end-effector position
		p.addUserDebugLine(
			targetLine_old, targetPos, lineWidth=3, lifeTime=cfg.debug.trailDuration, lineColorRGB=[0,0,0.3]
		)
		# debug line for the current end-effector position
		p.addUserDebugLine(
			currentPos_old, currentPos, lineWidth=3, lifeTime=cfg.debug.trailDuration, 
			lineColorRGB = [0,1,0] if contacts>0 else [1, 0, 0]
		)
		targetLine_old = copy.deepcopy(targetPos)
		currentPos_old = copy.deepcopy(currentPos)


	''' ---------------------- Handle constraints --------------------------------'''

	# get unconstrained acceleration and constraint matrices from MBD model
	mbdKuka.setJointStates(q, dq, q*0)
	mbdKuka.forwardKinematics()
	M, f, g = mbdKuka.computationOfMfg()
	C = mbdKuka.getJointFriction(dq)
	eta = - f - g
	J_lambda, sigma_lambda = surfaceConstraint.getConstraintTerms()
	A, b = J_lambda, -sigma_lambda

	# unconstrained forces Fa (generalized gravitational g, bias f, control inputs tau and joint friction forces C)
	Fa = g + f + tau + C

	''' ----- Calculate friction forces ------ '''
	if contacts:
		# calculate end-effector Jacobian
		I_J_FE,_ = I_J_P(kukaId, F_r_FP=np.array(cfg.ctrl.F_r_FE), linkIndex=kukaEndEffectorIndex)

		# OPTION A) viscous friction
		mu_C = cfg.contact_dynamics.friction.viscous.mu_C
		Ff = - mu_C * I_v_E 

		# OPTION B) stribeck friction		
		# N = lambda_mbd * (lambda_mbd > 0) 	# get normal force calculated from last step
		# mu_s = 0.2 	# static friction coeff.
		# mu_d = 0.1  # dynamic friction coeff.
		# v1 = 0.05
		# v2 = 0.0001
		# v = I_v_E
		# v_norm = np.linalg.norm(I_v_E)
		# Ff = - ( mu_d + (mu_s - mu_d) * np.exp(-v_norm/v1) ) * np.tanh(v_norm/v2) * v/v_norm * N

		# project friction force into generalized coordinates 
		Fz = I_J_FE.T @ Ff
	else: 
		Fz = 0*q
	
	''' ----- Calculate contact normal forces (lagrangian multiplier = lambda_mbd) and constrained generalized acceleration ddq_mbd (only for debug)  ----- '''
	if not contacts:
		ddq_mbd = np.linalg.solve(M, Fa + Fz)  # ddq_mbd, lambda_mbd = mbdKuka.forwardDynamics(q,dq,tau)
		lambda_mbd = np.array([0])
	else:
		As = block([
			[M,        -J_lambda.T], 
			[J_lambda,  0] 
		])
		bs = concatenate(( 
			Fa + Fz,
			-sigma_lambda
		))
		x = solve(As,bs)
		ddq_mbd    = x[0:nq]
		lambda_mbd = x[nq:]


	''' ----- if running in debug mode, check if the pyBullet generalized mass and inverse dynamics generate
	reasonable accelerations  ----- '''
	if cfg.debug.DEBUG_MODE:
		M2 = np.array(p.calculateMassMatrix(kukaId,q.tolist()))
		eta2 = np.array(p.calculateInverseDynamics(
			kukaId,
			objPositions     = q.tolist(),
			objVelocities    = dq.tolist(),
			objAccelerations = [0.0]*nq
		))
		Fa2 = - eta2 + tau + C
		ddq_bullet = np.linalg.solve(M2, Fa2 + Fz)

		if contacts:
			Ml = np.linalg.solve( A @ np.linalg.solve(M,A.T) , np.eye(len(A)) )
			L = np.linalg.solve(M,A.T) @ Ml
			T = eye(nq) - L @ A
			ddq_bullet = L @ b + T @ ddq_bullet


	''' -------------------------------- Log -------------------------------------'''

	# if cfg.debug.DEBUG_MODE:
	# 	plotMBD(mbdKuka)

	if sim_k > 0:
		# end-effector position of frame (P) in inertial (I) coordinates
		I_r_IE, I_v_E, _, _ = jointSpaceController.getEndEffectorState()
		I_a_E  = (I_v_E - I_v_E_old)*cfg.sim.freq	# finite diff. approximations
		I_v_E_old = copy.deepcopy(np.array(I_v_E))

		if not sim_k % (50 if p.getConnectionInfo()['connectionMethod'] == p.GUI else cfg.sim.freq):
			# print control info
			print(
				f'simulation time: {sim_t:.2f} [s] \n'
				f'  tau [Nm]: {(tau).astype(np.float)} \n'
				f'  error_q [deg]: {((q_des-q)*180/np.pi).astype(np.float)} \n'
				f'  error_dq [deg/s]: {((dq_des-dq)*180/np.pi).astype(np.float)} \n\n'
			)

		# Store current data into the log dictionary
		statelogs.append(
			t=sim_t, k=sim_k,
			q = q, 
			dq = dq,
			ddq = ddq,
			tau = tau + C,  				# discounting joint friction to applied torques
			contacts = contacts,
			M=M, f=f, g=g, A=A, b=b,
			q_des=q_des, dq_des=dq_des, ddq_des=ddq_des, 
			I_r_IEd=I_r_IEd, I_v_Ed=I_v_Ed, I_a_Ed=I_a_Ed,		# desired end-effector position
			I_r_IE=I_r_IE, I_v_E=I_v_E, I_a_E=I_a_E				# current end-effector position
		)


	''' ---------------- Apply control input and add Constraint Force ---------------------- '''

	# send torques to pyBullet/Apollo
	p.setJointMotorControlArray(
		bodyIndex=kukaId, 
		jointIndices=range(nq), 
		controlMode=p.TORQUE_CONTROL, 
		forces = tau + A.T @ lambda_mbd + Fz			# control and constraint forces
	)

	''' ---------------------- Simulation Step --------------------------------'''

	if cfg.sim.realTime:
		sim_t = datetime.now().second / 60.
	else:
		p.stepSimulation()
		sim_t += 1/cfg.sim.freq
	sim_k += 1

	if cfg.debug.DEBUG_MODE:
		time.sleep(1e-1)

print('\nSimulation finished succesfully\n')


''' ------------------ Save simulation results ------------------------- '''

if cfg.log.saveSimResults:
	with open( cfg.log.resultsFileName_raw, 'wb') as f:
		dill.dump(
			Dict({
				'dt': 1./cfg.sim.freq,
				'statelogs': statelogs,
				'trajplan': trajplan,
				'nq': nq
			}), 
			f
		)
	print(f'\nSAVED: {len(statelogs)} data points saved to {cfg.log.resultsFileName_raw}\n')

