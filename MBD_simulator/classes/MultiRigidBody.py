'''
TODO:
    [ ] Implement function to read from URDF file
    [ ] Implement CRBA, RNEA and ABA to improve performance
    [x] implement a function to print the structure of the kinematic tree
    [ ] implement a function Step that calls a RK integrator
    [ ] implement getLinkStates, getJointsInfo, getLinksInfo
    [x] setup joint indices automatically after kin. tree has been defined
    [ ] define a sim. mode ['GUI','DIRECT], and avoid requiring vpython package
        in case the user does not want to animate simulation. (avoid importing also)
    [ ] Implement methods: inverseDynamics, getGravityForces, getNonlinearForce, getMassMatrix
'''

import numpy as np
from numpy import eye, array, ones, zeros, pi, arange, concatenate, append, diag, linspace, block, sum, vstack, hstack, size, all, sum
from scipy.linalg import solve
from scipy.linalg import expm
from scipy.spatial.transform import Rotation as R
from typing import List, Dict, Tuple

from .robotics_helpfuns import skew
from .RigidBody import RigidBody, Ground
from .RotationalJoint import RotationalJoint
from .TranslationalJoint import TranslationalJoint
from .SpringDamper import SpringDamper

from vpython import canvas, vector, color, rate

class MultiRigidBody():
    
    def __init__( self, ground:Ground , springDampers:List=[], bilateralConstraints:List=[], name=''):
        self.ground = ground
        self.springDampers = springDampers
        self.bilateralConstraints = bilateralConstraints
        self.name = name
        self.jointList = self._getJointList(self.ground)
        self.linkList  = self._getLinkList(self.ground)
        self.nq = sum([ j.dof for j in self.jointList ])

        # set automatically qIndex for joints that were not defined before by the user
        used_qIndices = concatenate([array(j.qIndex).reshape(-1) for j in self.jointList])
        free_qIndices = set(range(self.nq)) - set(used_qIndices)
        for joint in self.jointList:
            if joint.qIndex is None:
                joint.qIndex = [free_qIndices.pop() for _ in range(joint.dof)]
                print(f'Warning: qIndex of joint {joint.name} is None, setting it to {joint.qIndex}')        

    def _getJointList(self, currentBody):
        ''' Recursive method to get a list of all joints that belongs to the multi-body system
        '''
        joints = currentBody.childJoints.tolist()
        for childJoint in currentBody.childJoints:
            joints += self._getJointList(childJoint.sucBody)
        return joints
    
    @property
    def jointMap(self, currentBody):
        # Recursive method to get a dictionary {jointName : jointObj}
        return { obj.name:obj for obj in self.jointList}
    
    def _getLinkList(self, currentBody):
        ''' Recursive method to get a list of all links that belongs to the multi-body system
        '''
        links = [currentBody]
        for childJoint in currentBody.childJoints:
            links += self._getLinkList(childJoint.sucBody)
        return links
    
    @property
    def linkMap(self):
        # Recursive method to get a dictionary {linkName : linkObj}
        return { obj.name:obj for obj in self.linkList}
    
    def forwardKinematics(self):
        self.ground._recursiveForwardKinematics(nq=self.nq)

    def computationOfMfg(self):
        return self.ground._recursiveComputationOfMfg()
    
    def getJointFriction(self, qDot):
        # get joint friction genralized forces
        tauF = zeros(self.nq)
        for j in self.jointList:
            tauF[j.qIndex] = - j.jointDampCoeff * qDot[j.qIndex]
        return tauF

    def forwardDynamics( self, q, qDot, tau=0): # -> qDDot
        ''' Return body acceleration ddq given joints position q and velocities dq
        If the multi rigid body is subject to ideal constraints, it also solves finds
        the constraint foces (reduces DAE to ODE).
        '''

        # Set all joint accelerations to zero, so the subsequent call to _recursiveForwardKinematics 
        # will produce bias accelerations, not real accelerations
        self.setJointStates( q=q, qDot=qDot, qDDot=zeros(self.nq) )
        
        # calculate system matrices
        self.forwardKinematics()
        M, f, g = self.computationOfMfg() # ground._recursiveComputationOfMfg()
        
        # calculate generalized forces tau, due to springs and dampers
        tauSD = 0
        for springdamper in self.springDampers:
            tauSD += springdamper.computationOfTau()

        # joint friction forces
        tauF = self.getJointFriction(qDot)

        # calculate constraint matrices
        J_lambda, sigma_lambda = array([]),array([])
        for const in self.bilateralConstraints:
            J, sigma = const.getConstraintTerms()
            J_lambda = vstack( [J_lambda, J] ) if J_lambda.size else J
            sigma_lambda = hstack( [sigma_lambda, sigma] ) if sigma_lambda.size else sigma
        
        nc = sigma_lambda.size # number of constraints
        if nc:
            # first remove constraint forces that can not be determined (row of zeros in J_lambda)
            colkeep = np.any(J_lambda,axis=1)
            J_lambda = J_lambda[colkeep,:]
            sigma_lambda = sigma_lambda[colkeep]
            nc = colkeep.sum()  # new number of constraints
            # set DAE system
            A = block([
                [M,         -J_lambda.T], 
                [J_lambda,  zeros([nc,nc])] 
            ])
            b = concatenate(( 
                f + g + tau + tauSD + tauF,
                -sigma_lambda
            ))
            
            # solve for accelerations and ideal constraint forces
            x = solve(A,b)
            qDDot = x[0:self.nq].squeeze()
            lamb  = x[self.nq:].squeeze()

        else:
            # set ODE system
            A = M
            b = f + g + tau + tauSD + tauF
            qDDot = solve(A,b).squeeze()
            lamb  = None    # no constraint forces

        # # solve for accelerations
        # qDDot= solve(A,b)[0:self.nq].squeeze()

        return qDDot, lamb

    def setJointStates(self, q:np.ndarray=None, qDot:np.ndarray=None, qDDot:np.ndarray=None):
        if qDot is None:
            qDot = 0*q
        if qDDot is None:
            qDDot = 0*q
        for joint in self.jointList:
            joint.q = q[joint.qIndex]
            joint.qDot = qDot[joint.qIndex]
            joint.qDDot = qDDot[joint.qIndex]

    def getJointStates(self):
        q    = zeros(self.nq)
        qDot = zeros(self.nq)
        qDDot = zeros(self.nq)
        for joint in self.jointList:
            q[joint.qIndex] = joint.q
            qDot[joint.qIndex] = joint.qDot
            qDDot[joint.qIndex] = joint.qDDot
        return [q, qDot, qDDot]

    def printKinTree(self):
        ''' Print kinematic tree
        '''
        print(f'Kinematic Tree - Robot {self.name}:')
        self.ground._printKinTree(prefix='', level=1, last=True)

    def initGraphics(self, width:int=1200, height:int=800, range:int=1.5, title:str='Vpython animation', updaterate:int=60):
        canvas(width=width, height=height, range=range, background=color.white, title=title)
        self.ground._recursiveInitGraphicsVPython()
        for sd in self.springDampers:
            sd.initGraphics()
        for bc in self.bilateralConstraints:
            bc.initGraphics()
        self.updaterate = updaterate
    
    def updateGraphics(self):
        self.ground._recursiveUpdateGraphicsVPython()
        for sd in self.springDampers:
            sd.updateGraphics()
        for bc in self.bilateralConstraints:
            bc.updateGraphics()
        rate(self.updaterate)


    def fromURDF( robotFileName:str, basePosition=np.zeros(3), baseOrientation=np.eye(3), I_grav=np.array([0,0,-9.8]) ):
        ''' Load model from URDF file
        '''
        import os, sys, importlib
        sys.path.append( os.path.join( os.path.dirname(__file__), os.path.pardir ) )
        from urdf_parser_py import urdf

        def inertiaObj2Matrix(inertiaObj):
            matrix = np.eye(3)
            matrix[0,0] = inertiaObj.ixx
            matrix[0,1] = inertiaObj.ixy
            matrix[0,2] = inertiaObj.ixz
            matrix[1,1] = inertiaObj.iyy
            matrix[1,2] = inertiaObj.iyz
            matrix[2,2] = inertiaObj.izz
            return matrix

        def addJoint( rbt:urdf.Robot, mbdPredBody:RigidBody, joint:urdf.Joint, I_r_IDp, A_IDp, I_r_IP, A_IP ):

            sucBody = rbt.link_map[joint.child] # rbt.link_map[sucBodyName]
            mbdSucBody = RigidBody(
                m_B = sucBody.inertial.mass,
                B_I_B = inertiaObj2Matrix(sucBody.inertial.inertia),
                I_grav = I_grav,
                name = sucBody.name
            )

            # update global joint configuration
            I_r_IDp = I_r_IDs = I_r_IDp + A_IDp @ joint.origin.position
            A_IDp   = A_IDs   = A_IDp @ R.from_euler('xyz', joint.origin.rotation ).as_matrix()

            # query succesor body global configuration
            Ds_r_DsS = np.array( sucBody.inertial.origin.position )
            A_DsS    = R.from_euler('xyz', sucBody.inertial.origin.rotation ).as_matrix()
            I_r_IS   = I_r_IDs + A_IDs @ Ds_r_DsS
            A_IS     = A_IDs @ A_DsS

            if joint.joint_type == 'revolute' and np.all(joint.axis == [0, 0, 1]):
                mbdJoint = RotationalJoint(
                    mbdPredBody, mbdSucBody,
                    A_PDp   = A_IP.T  @ A_IDp,
                    A_SDs   = A_DsS.T,
                    P_r_PDp = A_IP.T  @ (I_r_IDp - I_r_IP),
                    S_r_SDs = A_DsS.T @ (- Ds_r_DsS),
                    name    = joint.name,
                    qIndex  = None, #[i for i,v in enumerate(rbt.joints) if joint == v][0]
                    jointDampCoeff = joint.dynamics.damping,
                )
            else:
                raise NotImplementedError('At the moment this MBD class can not read anything but revolute joints in the z-axis from urdf files')
            
            # recursively connect successor body to its child bodies
            mbdPredBody = mbdSucBody
            I_r_IP = I_r_IS
            A_IP   = A_IS
            if sucBody.name in rbt.child_map.keys():	# body is not leaf
                for jointName, sucBodyName in rbt.child_map[sucBody.name]:
                    joint = rbt.joint_map[jointName]
                    addJoint( rbt, mbdPredBody, joint, I_r_IDp, A_IDp, I_r_IP, A_IP )

        rbt = urdf.Robot.from_xml_file( robotFileName )
        PredBodyName = rbt.get_root()
        predBody = rbt.link_map[PredBodyName]
        mbdPredBody = ground = Ground()	
        
        # fake center of gravity of base body = center of inertial frame
        I_r_IP  = np.zeros(3)
        A_IP    = np.eye(3)
        # fake old joint configuration
        I_r_IDp = basePosition
        A_IDp   = baseOrientation

        for jointName, sucBodyName in rbt.child_map[PredBodyName]:
            joint = rbt.joint_map[jointName]
            addJoint( rbt, mbdPredBody, joint, I_r_IDp, A_IDp, I_r_IP, A_IP )
        return  MultiRigidBody(ground=ground)
