'''
Help functions for simulating and controlling robots with pyBullet.

NOTE 
	Coodinate frames: 
		- R = Robot base frame
		- B = link CoM frame
		- F = link frame
		- I = inertial frame
		- E = end effector frame
		- Ed = desired end effector frame configuration
	Transformations:
		- A  = transformation matrix, Q=quaternion
		- w  = angular velocity
		- dw = angular acceleration
		- r  = linear displacement
		- v  = linear velocity
		- a  = linear acceleration
		- Js = translational jacobian
		- Jr = rotational jacobian
	
	A_IB   = The rotational orientation of the body CoM B with respect to the inertial frame I
	R_r_IB = displacement from I to B in R coordinates
	I_Js_B = translational jacobian of frame B described in inertial frame I, such that:  I_v_B = I_Js_B @ dq
	I_Jr_B = rotational    jacobian of frame B described in inertial frame I, such that:  I_w_B = I_Jr_B @ dq

'''

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import copy
import math
import os, sys
import time
from collections import ChainMap, namedtuple
from datetime import datetime
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data
import scipy
from scipy.spatial.transform import Rotation as R
from scipy.sparse.linalg import lsqr
from numpy.linalg import lstsq

# load multi-body dynamics engine (self-made)
from MBD_simulator.classes.RotationalJoint import RotationalJoint
from MBD_simulator.classes.MultiRigidBody import MultiRigidBody
from MBD_simulator.classes.BodyOnSurfaceBilateralConstraint import BodyOnSurfaceBilateralConstraint
from MBD_simulator.classes.RigidBody import RigidBody, Ground
from utils.StructArray import StructTorchArray, StructNumpyArray


''' 
	----------------------------- AUXILIARY FUNCTIONS ---------------------------------------- 
	------------------------------------------------------------------------------------------
'''

def getJointsInfo(robotId: int) -> StructNumpyArray:
	fieldNames = (
		'jointIndex', 'jointName', 'jointType', 'qIndex', 'uIndex',
		'flags', 'jointDamping', 'jointFriction', 'jointLowerLimit',
		'jointUpperLimit', 'jointMaxForce', 'jointMaxVelocity', 'linkName',
		'jointAxis', 'parentFramePos', 'parentFrameOrn', 'parentIndex'
	)
	jointsInfo = StructNumpyArray()
	numJoints = p.getNumJoints(robotId)
	for i in range(numJoints):
		jointsInfo.append( **dict(zip(fieldNames, p.getJointInfo(robotId, jointIndex=i))) )
	return jointsInfo

def getLinkStates(robotId: int) -> StructNumpyArray:
	fieldNames = (
		'I_r_IB', 'Q_IB', 'F_r_FB', 'Q_FB', 'I_r_IF', 'Q_IF', 'I_v_B', 'I_w_B'
	)
	linkStates = StructNumpyArray()
	numJoints = p.getNumJoints(robotId)
	for i in range(numJoints):
		linkStates.append( **dict(zip(fieldNames, p.getLinkState(robotId, linkIndex=i, computeLinkVelocity=1, computeForwardKinematics=0))) )
	return linkStates

def getLinksInfo(robotId: int) -> StructNumpyArray:

	fieldNames = (
		'm', 'lat_friction', 'B_I_B', 'F_r_FB', 'Q_FB', 'rest_coeff',    # R or Q is wrong
		'roll_friction', 'spinn_friction', 'cont_damping',
		'cont_stiffness', 'body_type', 'coll_margin'
	)
	linksInfo = StructNumpyArray()
	numJoints = p.getNumJoints(robotId)
	for i in range(numJoints):
		linksInfo.append( **dict(zip(fieldNames, p.getDynamicsInfo(robotId, linkIndex=i))) )
	return linksInfo

def plotMBD(mbd:MultiRigidBody):
	for a in mbd.jointList:
		p.addUserDebugLine(
			a.predBody.A_IB @ ( a.predBody.B_r_IB),
			a.predBody.I_r_IQ( a.P_r_PDp ),
			lineWidth=3, lifeTime=0.1, lineColorRGB=[0,0,1.0]
		)
		p.addUserDebugLine(
			a.predBody.I_r_IQ( a.P_r_PDp ),
			a.sucBody.A_IB @ ( a.sucBody.B_r_IB),
			lineWidth=3, lifeTime=0.1, lineColorRGB=[0,0,1.0]
		)

def skew(vec):
	''' 
	Generates a skew-symmetric matrix given a vector w
	'''
	S = np.zeros([3, 3])
	S[0, 1] = -vec[2]
	S[0, 2] = vec[1]
	S[1, 2] = -vec[0]
	S[1, 0] = vec[2]
	S[2, 0] = -vec[1]
	S[2, 1] = vec[0]
	return S

def forwardsDynamics(robotId: int, q, dq, tau, jointDampCoeff=0):
	''' Compute unconstrained joint accelerations, given joint positions, 
	velocities and torque applied.

	tau = ID(ddq,dq,q) = M*ddq + eta    -->   tau = ID(0,dq,q) = eta
	--> ddq = M^-1 * (tau - eta)

	Returns:
		- ddq: joint accelerations
	'''
	numJoints = p.getNumJoints(robotId)
	eta = np.array(p.calculateInverseDynamics(
		robotId,
		objPositions=q.tolist(),
		objVelocities=dq.tolist(),
		objAccelerations=[0.0]*numJoints
	))
	M = np.vstack(p.calculateMassMatrix(robotId, objPositions=q.tolist()))
	ddq = scipy.linalg.solve(M, tau - eta - jointDampCoeff * dq, assume_a='sym')
	return ddq

def matrixFromQuat(quat):
	''' Get transformation matrix from quaternions
	'''
	return np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)

def I_J_B(robotId: int, linkIndex: int, debug=False):
	''' Calculate the translational (Js) and rotational (Jr) jacobian for each link CoM (B),
	described in inertial coordinates (I)
	Args:
		- linkIndex: link (B), whose Jacobian will be calculated 
	Returns:
		- I_Js_B: translational jacobian such that  I_v_B = I_Js_B @ dq 
		- I_Jr_B: translational jacobian such that  I_w_B = I_Jr_B @ dq 
	'''
	numJoints = p.getNumJoints(robotId)

	# query joint states
	jointStates = p.getJointStates(robotId, range(numJoints))
	q, dq, reacTorque, appliedTorque = list(zip(*jointStates))

	# query robot position and orientation
	I_r_IR, Q_IR = p.getBasePositionAndOrientation(robotId)
	# query link state
	I_r_IB, Q_IB, F_r_FB, Q_FB, I_r_IF, Q_IF, I_v_B, I_w_B = p.getLinkState(
		robotId, linkIndex=linkIndex, computeLinkVelocity=1, computeForwardKinematics=0
	)
	# calculate jacobians (Js,Jr) of CoM (B) w.r.t. robot coordinates (R)
	B_r_FB = matrixFromQuat(Q_FB).T @ F_r_FB
	R_Js_B, R_Jr_B = p.calculateJacobian(
		robotId,
		linkIndex=linkIndex,
		localPosition=tuple(B_r_FB),
		objPositions=q,
		objVelocities=[0]*len(q),
		objAccelerations=[0]*len(q)
	)
	# transform to inertial frame description
	A_IR = matrixFromQuat(Q_IR)
	I_Js_B = A_IR @ R_Js_B
	I_Jr_B = A_IR @ R_Jr_B

	if debug:
		np.set_printoptions(precision=10, suppress=True,
							threshold=1000, linewidth=300)
		# sanity check:
		A_IF = matrixFromQuat(Q_IF)
		A_IB = matrixFromQuat(Q_IB)
		# sanity check: (A_IF.T @ A_IB - A_FB) must return a matrix of zeros
		A_FB = matrixFromQuat(Q_FB)
		print('A_FB: \n', A_FB)
		print(f'Transformation error (A_IF.T @ A_IB - A_FB): \n{A_IF.T @ A_IB - A_FB}')
		print(f'position error:                           {I_r_IB - (I_r_IF + A_IF @ F_r_FB)}')
		print(f'Orientation error (I_w_B - I_Jr_B @ dq): {I_w_B - I_Jr_B @ dq}')
		print(f'Velocity error (I_v_B - I_Js_B @ dq):    {I_v_B - I_Js_B @ dq}')

	return I_Js_B, I_Jr_B

def I_J_P(robotId, F_r_FP, linkIndex):
	''' Calculate jacobian of a arbitrary frame (P) on the link sepcified by linkIndex
	'''
	# get current link states
	I_r_IB, Q_IB, F_r_FB, Q_FB, I_r_IF, Q_IF = p.getLinkState(
		robotId, linkIndex=linkIndex, computeLinkVelocity=0, computeForwardKinematics=0
	)
	# calculate translational and rotational jacobians for the end effector link CoM frame (B)
	I_Js_B, I_Jr_B = I_J_B(robotId, linkIndex=linkIndex, debug=False)
	# calculate Jacobians for end effector (E)
	I_r_BP = matrixFromQuat(Q_IF) @ (F_r_FP - F_r_FB)
	I_Js_P = I_Js_B - skew(I_r_BP) @  I_Jr_B
	I_Jr_P = I_Jr_B
	return I_Js_P, I_Jr_P

# def test_I_J_B():
# 	''' Test function I_J_B '''
# 	pi = math.pi
# 	q = (np.random.rand(7)*0.5).tolist()  # [pi/2*0,0,0,0,0,0,0] #
# 	dq = (np.random.rand(7)*0.5).tolist()  # [1,0,0,0,0,0,0]      #
# 	numJoints=7
# 	for i in range(numJoints):
# 		p.resetJointState(kukaId, i, targetValue=q[i], targetVelocity=dq[i])
# 	np.set_printoptions(precision=4, suppress=True, threshold=1000, linewidth=300)
# 	for i in range(numJoints):
# 		I_Js_B, I_Jr_B = I_J_B(kukaId, linkIndex=i, debug=True)
# 	return

''' 
	-------------------------------- CONTROL CLASSES  ---------------------------------------- 
	------------------------------------------------------------------------------------------
'''

class InverseDynControl():

	def __init__(self, robotId: int,
				 endEffectorLinkIndex,
				 F_r_FE, Q_FE,
				 lowerLimits, upperLimits, jointRanges, restPoses,
				 wn_q: float = 10.0, xi_q: float = 1.0,
				 wn_s: float = 10.0, xi_s: float = 1.0,
				 wn_r: float = 10.0, xi_r: float = 1.0,
				 K_reg=None,
				 multiRigidBody=None):
		''' Defines a frame (E) in the end-effector body, given by 'endEffectorLinkIndex', to be controlled 

		Arguments:
			- wn, xi: controller natural frequency and damping ratio
			- endEffectorLinkIndex: index of end-effector link (B)
			- F_r_FE: displacement from frame (E) and frame (F) in (F) coordinates.
			- Q_FP: rotational orientation of the frame (E) with respect to the frame (F)  
		'''
		self.robotId = robotId
		self.numJoints = p.getNumJoints(robotId)
		self.endEffectorLinkIndex = endEffectorLinkIndex
		self.F_r_FE = np.array(F_r_FE)
		self.Q_FE = np.array(Q_FE)
		self.lowerLimits, self.upperLimits, self.jointRanges, self.restPoses = lowerLimits, upperLimits, jointRanges, restPoses
		self.jointDampCoeff = np.array([p.getJointInfo(robotId, jointIndex=i)[
									   6] for i in range(self.numJoints)])
		self.wn_q = wn_q
		self.xi_q = xi_q
		self.wn_s = wn_s
		self.xi_s = xi_s
		self.wn_r = wn_r
		self.xi_r = xi_r
		self.K_reg = K_reg
		self.mbd = multiRigidBody

	def getEndEffectorState(self, computeForwardKinematics=1):
		I_r_IB, Q_IB, F_r_FB, Q_FB, I_r_IF, Q_IF, I_v_B, I_w_B = p.getLinkState(
			self.robotId, linkIndex=self.endEffectorLinkIndex, computeLinkVelocity=1, computeForwardKinematics=0
		)
		I_r_IE = np.array(I_r_IF) + matrixFromQuat(Q_IF) @ np.array(self.F_r_FE)
		I_v_E = np.array(I_v_B) + np.cross(I_w_B, matrixFromQuat(Q_IF) @ np.array(self.F_r_FE - F_r_FB))
		Q_IE = (R.from_quat(Q_IF) * R.from_quat(self.Q_FE)).as_quat()
		I_w_E = np.array(I_w_B)
		return I_r_IE, I_v_E, Q_IE, I_w_E

	def getBiasAcceleration(self, q, dq):
		I_r_IB, Q_IB, F_r_FB, Q_FB, I_r_IF, Q_IF, I_v_B, I_w_B = p.getLinkState(
			self.robotId, linkIndex=self.endEffectorLinkIndex, computeLinkVelocity=1, computeForwardKinematics=0
		)
		eeLink = self.mbd.linkList[-1]
		B_r_BE = np.linalg.solve(matrixFromQuat(Q_FB), (self.F_r_FE - F_r_FB))
		I_Ss_E = eeLink.I_a_Q(B_r_BQ=B_r_BE)
		I_Sr_E = eeLink.I_dw_Q(B_r_BQ=B_r_BE)
		return I_Ss_E, I_Sr_E

	def inverseDiffKinematics(
		self, I_r_IEd, I_v_Ed, I_a_Ed, Q_IEd, I_w_Ed, I_dw_Ed,	# desired end-effector dynamic state
		useOrientation=False, useNullSpaceIK=False):
		''' Calculate a desired 
		'''
		# query current joint positions and accelerations
		jointStates = p.getJointStates(self.robotId, range(self.numJoints))
		q, dq, reacTorque, appliedTorque = np.array(list(zip(*jointStates)))
		# convert type object to type float
		q, dq = q.astype(np.float), dq.astype(np.float)

		# query current end effector configuration in inertial coordinates
		I_r_IE, I_v_E, Q_IE, I_w_E = self.getEndEffectorState()

		# calculate translational and rotational jacobians for the end effector (E)
		I_Js_E, I_Jr_E = I_J_P(self.robotId, F_r_FP=self.F_r_FE, linkIndex=self.endEffectorLinkIndex)

		# translational bias acceleration can not be calculate with Bullet... assuming zero
		I_Ss_E, I_Sr_E = self.getBiasAcceleration(q, dq)

		# get current link states
		I_r_IB, Q_IB, F_r_FB, Q_FB, I_r_IF, Q_IF = p.getLinkState(
			self.robotId, linkIndex=self.endEffectorLinkIndex, computeLinkVelocity=0, computeForwardKinematics=0
		)
		# find link frame configuration (Fd) that will make end effector (E) match the desired end-effector configuration (Ed)
		I_r_FE = matrixFromQuat(Q_IF) @ self.F_r_FE
		I_r_IFd = I_r_IEd - I_r_FE
		Q_IFd = (R.from_quat(Q_IEd) * R.from_quat(self.Q_FE).inv()).as_quat()

		IKparams = dict(ChainMap(*[
			{
				'bodyUniqueId': self.robotId,
				'endEffectorLinkIndex': self.endEffectorLinkIndex,
				# target link frame (F) position in inertial frame (I) coordinates
				'targetPosition': tuple(I_r_IFd)
			},
			{
				'lowerLimits': self.lowerLimits,
				'upperLimits': self.upperLimits,
				'jointRanges': self.jointRanges,
				'restPoses':   self.restPoses
			} if useNullSpaceIK else {},
			{
				'targetOrientation': tuple(Q_IFd)
			} if useOrientation else {}
		]))

		# desired joint positions obtained from inverse kinematics
		q_des = p.calculateInverseKinematics(**IKparams)

		# joint velocities obtained from inverse differential kinematics: [I_v_D;I_w_D] = [I_Js_B; I_Jr_B] @ dq
		if not useOrientation:
			# sanity check: np.array(I_Js_P) @ dq_des - I_v_P -> 0
			dq_des = lstsq(I_Js_E, I_v_Ed, rcond=None)[0]
			# sanity check: np.array(I_Js_P) @ dq_des - (I_a_P - I_Ss_B) -> 0
			ddq_des = lstsq(I_Js_E, I_a_Ed - I_Ss_E, rcond=None)[0]
		else:
			dq_des = lstsq(
				np.vstack((I_Js_E, I_Jr_E)),
				np.hstack((I_v_Ed, I_w_Ed)), rcond=None
			)[0]
			ddq_des = lstsq(
				np.vstack((I_Js_E, I_Jr_E)),
				np.hstack((I_a_Ed - I_Ss_E, I_dw_Ed - I_Sr_E)), rcond=None
			)[0]
		return q_des, dq_des, ddq_des


	def jointSpaceControl(self, q_des, dq_des, ddq_des):
		''' Inverse Dynamics Joint Space Control
		TODO:   
			- [ ] include bias acceleration I_Ss_B and I_Sr_B to better estimate ddq_des.
			Since bullet does not offers a ForwardDynamics function, this makes the computation harder  
		'''
		# query current joint positions and accelerations
		jointStates = p.getJointStates(self.robotId, range(self.numJoints))
		q, dq, reacTorque, appliedTorque = np.array(list(zip(*jointStates)))
		# convert type object to type float
		q, dq = q.astype(np.float), dq.astype(np.float)

		# controller gain matrices
		Kp = self.wn_q**2
		Kd = 2 * self.xi_q * self.wn_q

		# desired joint acceleration
		y = Kp * (q_des - q) + Kd * (dq_des - dq) + ddq_des

		# tau  =  M*y - f - g  =  M*(Kp @ (q_des - q) + Kd @ (dq_des - dq) + ddq_des) - f - g
		# note: pyBullet ignores joint/link damping effects when calculating inverse Dynamics by means of RNEA method
		tau = p.calculateInverseDynamics(
			self.robotId,
			objPositions=q.tolist(),
			objVelocities=dq.tolist(),
			objAccelerations=y.tolist()
		)
		# compensate joint damping
		tau += self.jointDampCoeff * dq
		# return joint torques
		return tau

	def taskSpaceControl(self, I_r_IEd, I_v_Ed, I_a_Ed, Q_IEd, I_w_Ed, I_dw_Ed, useOrientation=False): # desired end-effector dynamic state
		''' Inverse Dynamics Task Space Control
		'''
		# query current joint positions and accelerations
		jointStates = p.getJointStates(self.robotId, range(self.numJoints))
		q, dq, reacTorque, appliedTorque = np.array(list(zip(*jointStates)))
		# convert type object to type float
		q, dq = q.astype(np.float), dq.astype(np.float)

		# query current end effector configuration in inertial coordinates
		I_r_IE, I_v_E, Q_IE, I_w_E = self.getEndEffectorState()

		# calculate translational and rotational jacobians for the end effector (E)
		I_Js_E, I_Jr_E = I_J_P(self.robotId, F_r_FP=self.F_r_FE, linkIndex=self.endEffectorLinkIndex)
		# translational bias acceleration can not be calculate with Bullet... assuming zero
		I_Ss_E, I_Sr_E = self.getBiasAcceleration(q, dq)

		# controller gain matrices
		Kp_s = self.wn_s**2
		Kp_r = self.wn_r**2
		Kd_s = 2 * self.xi_s * self.wn_s
		Kd_r = 2 * self.xi_r * self.wn_r

		# query mass matrix
		M = p.calculateMassMatrix(self.robotId, tuple(q))

		# secondary goal
		ddq_reg = self.restPoses - q - 0.01*dq
		
		# tracking errors
		Q_EdE   = (R.from_quat(Q_IEd) * R.from_quat(Q_IE).inv()).as_quat()
		I_r_EEd = I_r_IEd - I_r_IE
		I_v_EEd = I_v_Ed - I_v_E
		I_w_EEd = I_w_Ed - I_w_E

		if useOrientation:
			I_J_E = np.vstack([I_Js_E, I_Jr_E])
			y = lsqr(
				I_J_E,
				np.hstack([
					Kp_s * I_r_EEd    + Kd_s * I_v_EEd + I_a_Ed - I_Ss_E,
					Kp_r * Q_EdE[0:3] + Kd_r * I_w_EEd + I_dw_Ed - I_Sr_E,
				]),
				damp=0.001
			)[0]

		else:
			I_J_E = I_Js_E
			y = lsqr(
				I_J_E,
				Kp_s * I_r_EEd + Kd_s * I_v_EEd + I_a_Ed - I_Ss_E,
				damp=0.001
			)[0]

		# add acceleration for secondary goal
		if not self.K_reg is None:
			# null space projector of I_J_E, s.t. for any w, I_J_E @ (NullSpaceProj @ w) = 0				
			NullSpaceProj = np.eye(self.numJoints) - lstsq(I_J_E, I_J_E,rcond=None)[0]
			y += NullSpaceProj @ (self.K_reg @ ddq_reg)

		# tau  =  M*y - f - g  =  M*(Kp @ (q_des - q) + Kd @ (dq_des - dq) + ddq_des) - f - g
		# note: pyBullet ignores joint/link damping effects when calculating inverse Dynamics by means of RNEA method
		tau = p.calculateInverseDynamics(
			self.robotId,
			objPositions=q.tolist(),
			objVelocities=dq.tolist(),
			objAccelerations=y.tolist()
		)
		# compensate joint damping
		tau += self.jointDampCoeff * dq
		# return joint torques
		return tau

