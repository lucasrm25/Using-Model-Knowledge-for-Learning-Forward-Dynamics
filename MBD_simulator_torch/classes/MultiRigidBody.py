'''
TODO:
    [x] Implement function to read from URDF file
    [ ] Implement CRBA, RNEA and ABA to improve performance
    [x] implement a function to print the structure of the kinematic tree
    [ ] implement a function Step that calls a RK integrator
    [x] implement getLinkStates, getJointsInfo, getLinksInfo
    [x] setup joint indices automatically after kin. tree has been defined
    [ ] define a sim. mode ['GUI','DIRECT], and avoid requiring vpython package
        in case the user does not want to animate simulation. (avoid importing also)
    [x] Implement methods: inverseDynamics, getGravityForces, getNonlinearForce, getMassMatrix
'''

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import List, Tuple
from .torch_utils import *
from .RigidBody import RigidBody, Ground
from .RotationalJoint import RotationalJoint
from .TranslationalJoint import TranslationalJoint
from .SpringDamper import SpringDamper

class MultiRigidBody(torch.nn.Module):
    
    def __init__( self, ground:Ground , springDampers:List=[], bilateralConstraints:List=[], name=''):
        super().__init__()
        self.ground = ground
        self.springDampers = torch.nn.ModuleList(springDampers)
        self.bilateralConstraints = torch.nn.ModuleList(bilateralConstraints)
        self.name = name
        self.jointList = torch.nn.ModuleList(self._getJointList(self.ground))
        self.linkList  = torch.nn.ModuleList(self._getLinkList(self.ground))
        self.nq = sum([ j.dof for j in self.jointList ])

        # set automatically qIndex for joints that were not defined before by the user
        used_qIndices = [ j.qIndex for j in self.jointList]
        free_qIndices = set(range(self.nq)) - set(used_qIndices)
        for joint in self.jointList:
            if joint.qIndex is None:
                joint.qIndex = torch.tensor([free_qIndices.pop() for _ in range(joint.dof)]).long()
                print(f'Warning: qIndex of joint {joint.name} is None, setting it to {joint.qIndex}')

    def _getJointList(self, currentBody):
        ''' Recursive method to get a list of all joints that belongs to the multi-body system
        '''
        joints = [*currentBody.getChildJoints()]
        for childJoint in currentBody.getChildJoints():
            joints += self._getJointList(childJoint.getSucBody())
        return joints
    
    @property
    def jointMap(self):
        # Recursive method to get a dictionary {jointName : jointObj}
        return { obj.name:obj for obj in self.jointList}
    
    def _getLinkList(self, currentBody):
        ''' Recursive method to get a list of all links that belongs to the multi-body system
        '''
        links = [currentBody]
        for childJoint in currentBody.getChildJoints():
            links += self._getLinkList(childJoint.getSucBody())
        return links
    
    @property
    def linkMap(self):
        # Recursive method to get a dictionary {linkName : linkObj}
        return { obj.name:obj for obj in self.linkList}
    
    def forwardKinematics(self, q:torch.Tensor, qDot:torch.Tensor=None, qDDot:torch.Tensor=None):
        self.setJointStates(q=q, qDot=qDot, qDDot=qDDot)
        device, batchSize = inputInfo(q)
        self.ground._recursiveForwardKinematics(nq=self.nq, batchSize=batchSize)

    def computationOfMfg(self):
        return self.ground._recursiveComputationOfMfg()
    
    def getJointFriction(self, qDot:torch.Tensor):
        device, batchSize = inputInfo(qDot)
        # get joint friction genralized forces
        tauF = torch.zeros((batchSize,self.nq), device=device)
        for j in self.jointList:
            tauF[:,j.qIndex] = - j.jointDampCoeff * qDot[:,j.qIndex]
        return tauF

    def forwardDynamics( self, q:torch.Tensor, qDot:torch.Tensor, tau:torch.Tensor=0): # -> qDDot
        ''' Return body acceleration ddq given joints position q and velocities dq
        If the multi rigid body is subject to ideal constraints, it also solves finds
        the constraint foces (reduces DAE to ODE).
        '''
        device, batchSize = inputInfo(q)

        # Set all joint accelerations to zero, so the subsequent call to _recursiveForwardKinematics 
        # will produce bias accelerations, not real accelerations
        self.forwardKinematics( q=q, qDot=qDot, qDDot=0*q )
        M, f, g = self.computationOfMfg() # ground._recursiveComputationOfMfg()
        
        # calculate generalized forces tau, due to springs and dampers
        tauSD = 0
        for springdamper in self.springDampers:
            tauSD += springdamper.computationOfTau()

        # joint friction forces
        tauF = self.getJointFriction(qDot)

        # calculate constraint matrices
        J_lambda, sigma_lambda = torch.tensor([]),torch.tensor([])
        for const in self.bilateralConstraints:
            J, sigma = const.getConstraintTerms()
            J_lambda = torch.cat( [J_lambda, J] ,dim=0) if len(J_lambda) else J
            sigma_lambda = torch.cat( [sigma_lambda, sigma] ) if len(sigma_lambda) else sigma
        
        nc = sigma_lambda.shape[-1] # number of constraints
        if nc:
            # set DAE system
            A = torch.cat([
                torch.cat([M,         - bT(J_lambda)], dim=-1),
                torch.cat([J_lambda,  torch.zeros((batchSize,nc,nc),device=device)], dim=-1)
            ],dim=1)
            b = torch.cat(( 
                f + g + tau + tauSD + tauF,
                -sigma_lambda
            ),dim=1)
            
            # solve for accelerations and ideal constraint forces
            x = blstsq(A,b)

            qDDot = x[:,:self.nq]
            lamb  = x[:,self.nq:]

        else:
            # set ODE system
            A = M
            b = f + g + tau + tauSD + tauF
            qDDot = blstsq(A,b)
            lamb  = None    # no constraint forces

        return qDDot, lamb

    def setJointStates(self, q:torch.Tensor, qDot:torch.Tensor=None, qDDot:torch.Tensor=None):        
        if qDot is None:
            qDot = 0*q
        if qDDot is None:
            qDDot = 0*q
        assert q.dim()==2 and qDot.dim()==2 and qDDot.dim()==2, 'Inputs dimension must be <batch_size,input_size>'
        assert isinstance(q,torch.Tensor) and isinstance(qDot,torch.Tensor) and isinstance(qDDot,torch.Tensor)
        for joint in self.jointList:
            joint.q = q[:,joint.qIndex]
            joint.qDot = qDot[:,joint.qIndex]
            joint.qDDot = qDDot[:,joint.qIndex]

    def getJointStates(self):
        # get batch and device info from an arbitrary joint position
        device, batchSize = inputInfo( self.jointList[0].q )
        # pre-allocate result tensors
        q     = torch.zeros( (batchSize,self.nq), device=device)
        qDot  = torch.zeros( (batchSize,self.nq), device=device)
        qDDot = torch.zeros( (batchSize,self.nq), device=device)
        for joint in self.jointList:
            q[:,joint.qIndex]    = joint.q
            qDot[:,joint.qIndex] = joint.qDot
            qDDot[:,joint.qIndex] = joint.qDDot
        return [q, qDot, qDDot]

    def printKinTree(self):
        ''' Print kinematic tree
        '''
        print(f'Kinematic Tree - Robot {self.name}:')
        self.ground.printKinTree(prefix='', level=1, last=True)

    def __str__(self):
        matrix2String = lambda m: m.__str__().replace("\n",",")
        classAndName = lambda obj: obj.__class__.__name__ + ":" + obj.name
        s  = f'{self.__class__.__name__}: {self.name}\n'
        s += f'\tlist of bodies: \n\t\t{[ classAndName(l) for l in self.linkList]}\n'
        s += f'\tlist of joints: \n\t\t{[ classAndName(l) for l in self.jointList]}\n'
        s += f'\tlist of spring/dampers: \n\t\t{[ classAndName(sd) for sd in self.springDampers]}\n'
        s += f'\tlist of bilateralConstraints: \n\t\t{[ classAndName(bc) for bc in self.bilateralConstraints]}\n'
        return s

    def __repr__(self):
        return self.__str__()

    def initGraphics(self, width:int=1200, height:int=800, range:int=1.5, title:str='Vpython animation', updaterate:int=60):
        from vpython import canvas, vector, color, rate

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

    @classmethod
    def fromURDF(self, robotFileName:str, basePosition=torch.zeros(3), baseOrientation=torch.eye(3), I_grav=torch.tensor([0,0,-9.8]) ):
        ''' Load model from URDF file
        '''
        import os, sys, importlib
        sys.path.append( os.path.join( os.path.dirname(__file__), os.path.pardir ) )
        from urdf_parser_py import urdf

        def inertiaObj2Matrix(inertiaObj):
            matrix = torch.eye(3)
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
                m_B = torch.tensor(sucBody.inertial.mass),
                B_I_B = inertiaObj2Matrix(sucBody.inertial.inertia),
                I_grav = I_grav,
                name = sucBody.name
            )

            # update global joint configuration
            I_r_IDp = I_r_IDs = I_r_IDp + A_IDp @ torch.tensor(joint.origin.position)
            A_IDp   = A_IDs   = A_IDp @ torch.tensor(R.from_euler('xyz', joint.origin.rotation ).as_matrix())

            # query succesor body global configuration
            Ds_r_DsS = torch.tensor( sucBody.inertial.origin.position )
            A_DsS    = torch.tensor( R.from_euler('xyz', sucBody.inertial.origin.rotation ).as_matrix() )
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

        # read urdf file using parser library
        rbt = urdf.Robot.from_xml_file( robotFileName )
        PredBodyName = rbt.get_root()
        predBody = rbt.link_map[PredBodyName]
        mbdPredBody = ground = Ground()	
        
        # first body is GROUND and therefore has a fake center of gravity = center of inertial frame
        # here we ignore the origin of the Ground body, since it does not matter for MBD
        I_r_IP  = torch.zeros(3)
        A_IP    = torch.eye(3)
        # first joint configuration
        I_r_IDp = basePosition
        A_IDp   = baseOrientation

        for jointName, sucBodyName in rbt.child_map[PredBodyName]:
            joint = rbt.joint_map[jointName]
            addJoint( rbt, mbdPredBody, joint, I_r_IDp, A_IDp, I_r_IP, A_IP )
        return  MultiRigidBody(ground=ground)
