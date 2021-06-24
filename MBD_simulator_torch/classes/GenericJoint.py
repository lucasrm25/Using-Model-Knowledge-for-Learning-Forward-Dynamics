'''
'''

# import numpy as np
# from numpy import append, array, size, ones, zeros, eye
# from numpy.linalg import matrix_power

import torch
from typing import List
from abc import ABC, abstractmethod
from .RigidBody import RigidBody
from .LinkedConstraintModule import LinkedConstraintModule
from .torch_utils import *

class LinkedJointModule(LinkedConstraintModule):
    def __init__(self, predBody:RigidBody, sucBody:RigidBody):
        super().__init__(predBody, sucBody)        
        # link pred and suc bodies to current joint 
        self.getPredBody().appendChildJoint(self)
        self.getSucBody().setParentJoint(self)

class GenericJoint(LinkedJointModule, ABC):
    ''' Abstract class that defines a generic Joint

    Defines a generic joint (without any specific motion) in a kinematic tree that is implemented as a set of linked selfects.  
    This class allows the computation of Positions/Orientations, Velocities, Accelerations, and Jacobians.
    '''

    def __init__(self, predBody:RigidBody, sucBody:RigidBody, 
                       A_PDp:torch.Tensor=torch.eye(3), A_SDs:torch.Tensor=torch.eye(3),
                       P_r_PDp:torch.Tensor=torch.zeros(3), S_r_SDs:torch.Tensor=torch.zeros(3), 
                       name:str='', dof:int=0, qIndex:List[int]=None, jointDampCoeff:float=0.0):
                       
        super().__init__(predBody, sucBody)
        
        # init params
        self.A_PDp   = param(A_PDp)
        self.A_SDs   = param(A_SDs)
        self.P_r_PDp = param(P_r_PDp)
        self.S_r_SDs = ConstrainedParameter(S_r_SDs, Interval(S_r_SDs-0.1, S_r_SDs+0.1), requires_grad=False)
        self.jointDampCoeff = param(jointDampCoeff)

        # init abstract properties
        self.dof     = dof
        self.q       = torch.zeros(self.dof)
        self.qDot    = torch.zeros(self.dof)
        self.qDDot   = torch.zeros(self.dof)
        self.qIndex  = qIndex    # generalized coordinates indices
        self.name    = name
        
        # assert qIndex is None or size(sef.qIndex)==self.dof, f'Expected size(qIndex={sef.qIndex})==self.dof of joint {self.name}'

    @property
    def qIndex(self): 
        return self._qIndex
    
    @qIndex.setter
    def qIndex(self, val):
        if val is None:
            self._qIndex = val
        else: 
            # force variable to be ndarray with dimension==1
            self._qIndex = torch.tensor(val, dtype=torch.int).reshape(-1).long()
            assert self._qIndex.ndim == 1

    @abstractmethod
    def JointJacobian(self, q, qIndex, nq): # -> [S, R]
        pass

    @abstractmethod
    def JointFunction(self, q): # -> [Dp_r_DpDs, A_DpDs]
        pass

    @abstractmethod
    def JointVelocity(self, q, qDot): # -> [Dp_rDot_DpDs, Dp_omega_DpDs]
        pass

    @abstractmethod
    def JointAcceleration(self, q, qDot, qDDot): # -> [Dp_rDDot_DpDs, Dp_omegaDot_DpDs]
        pass

    def _recursiveForwardKinematics(self, nq, batchSize, P_r_IP, A_IP, P_omega_P, P_v_P, P_omega_dot_P, P_a_P, P_Js_P, P_Jr_P):
        '''
        Given predecessor (P) body kinematics:
            - P_r_IP, A_IP, P_omega_P, P_v_P, P_omega_dot_P, P_a_P, P_Js_P, P_Jr_P
        Calculate kinematics for the succesor body:
            - S_r_IS, A_IS, S_omega_S, S_v_S, S_omegaDot_S, S_a_S, S_Js_S, S_Jr_S
        based on current joint configuration
        '''
        # Rotation and displacement about the joint:
        Dp_r_DpDs, A_DpDs = self.JointFunction(self.q)
        # Angular rate and translational velocity accross the joint:
        Dp_rDot_DpDs, Dp_omega_DpDs = self.JointVelocity(self.q, self.qDot)
        # Angular andtranslational acceleration accross the joint:
        Dp_rDDot_DpDs, Dp_omegaDot_DpDs = self.JointAcceleration(self.q, self.qDot, self.qDDot)
        # Compute the matrices that define the velocity accross the joint as a linear function of q_dot:
        # nq = size(P_Js_P,1)
        S, R = self.JointJacobian( self.q, self.qIndex, nq)

        # Compute the position,qIndex velocity, and acceleration of each successing coordinate system:
        A_IDp           = bmm( A_IP , self.A_PDp )
        Dp_r_IDp        = bmv( bT(self.A_PDp) , P_r_IP + self.P_r_PDp ) 
        Dp_omega_Dp     = bmv( bT(self.A_PDp) , P_omega_P + 0 ) 
        Dp_v_Dp         = bmv( bT(self.A_PDp) , P_v_P + 0 + bmv(batch(skew,P_omega_P) , self.P_r_PDp) ) 
        Dp_omegaDot_Dp  = bmv( bT(self.A_PDp) , P_omega_dot_P + 0 + 0 ) 
        Dp_a_Dp         = bmv( bT(self.A_PDp) , P_a_P + 0 + 0 + bmv( batch(skew,P_omega_dot_P) + torch.matrix_power(batch(skew,P_omega_P),2) , self.P_r_PDp) ) 

        A_IDs           = bmm( A_IDp , A_DpDs )
        Ds_r_IDs        = bmv( bT( A_DpDs ) , (Dp_r_IDp + Dp_r_DpDs) )

        Ds_omega_Ds     = bmv( bT(A_DpDs) , Dp_omega_Dp + Dp_omega_DpDs )
        Ds_v_Ds         = bmv( bT(A_DpDs) , Dp_v_Dp + Dp_rDot_DpDs + bmv(batch(skew,Dp_omega_Dp) , Dp_r_DpDs) )
        Ds_omegaDot_Ds  = bmv( bT(A_DpDs) , Dp_omegaDot_Dp + Dp_omegaDot_DpDs + bmv(batch(skew,Dp_omega_Dp) , Dp_omega_DpDs) )
        Ds_a_Ds         = bmv( bT(A_DpDs) , Dp_a_Dp + Dp_rDDot_DpDs + 2 * bmv(batch(skew,Dp_omega_Dp) , Dp_rDot_DpDs) + bmv(batch(skew,Dp_omegaDot_Dp) + torch.matrix_power(batch(skew,Dp_omega_Dp),2) , Dp_r_DpDs ) )

        A_IS            = bmm( A_IDs , bT(self.A_SDs) )
        S_r_IS          = bmv( self.A_SDs , Ds_r_IDs ) - self.S_r_SDs()
        S_omega_S       = bmv( self.A_SDs , Ds_omega_Ds + 0 )
        S_v_S           = bmv( self.A_SDs , Ds_v_Ds + 0 ) - bmv( batch(skew,S_omega_S) , self.S_r_SDs() )
        S_omegaDot_S    = bmv( self.A_SDs , Ds_omegaDot_Ds + 0 + 0 )
        S_a_S           = bmv( self.A_SDs , Ds_a_Ds + 0 + 0 ) - bmv(batch(skew,S_omegaDot_S) + torch.matrix_power(batch(skew,S_omega_S),2) , self.S_r_SDs() )

        # Compute the displacement and orientation of the successor body:
        # Compute the overall rotation first:
        A_PS = bmm( self.A_PDp , bmm(A_DpDs , bT(self.A_SDs)) )
        # Compute the rotational Jacobian of the successor body:
        S_Jr_S = bmm( bT(A_PS) ,  P_Jr_P + bmm( self.A_PDp , R) )
        # Compute the translational Jacobian of the successor body:
        S_Js_S = bmm( bT(A_PS) , P_Js_P + bmm(self.A_PDp,S) + bmm(bT(batch(skew, self.P_r_PDp + bmv(self.A_PDp,Dp_r_DpDs))) , P_Jr_P) ) - bmm(bT(batch(skew,self.S_r_SDs())) , S_Jr_S)

        # Pass this information on to the successor body:
        self.getSucBody()._recursiveForwardKinematics(nq, batchSize, S_r_IS, A_IS, S_omega_S, S_v_S, S_omegaDot_S, S_a_S, S_Js_S, S_Jr_S)

    def printKinTree(self, prefix='', level=1, last=False):
        print(f"{prefix}{'└──' if last else '├──'} ● {self.name} ({self.__class__.__name__}) qIdx={self.qIndex}")
        prefix += '    ' if last else '|   '
        self.getSucBody().printKinTree(prefix, level+1, last=True)