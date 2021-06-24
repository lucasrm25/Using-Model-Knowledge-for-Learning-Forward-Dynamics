from numpy import array, ones, zeros, eye, cos, sin, asscalar
from .GenericJoint import GenericJoint
from .robotics_helpfuns import skew

class TranslationalJoint(GenericJoint):

    def __init__(self, predBody, sucBody,
                       A_PDp = eye(3), A_SDs = eye(3),
                       P_r_PDp = zeros(3), S_r_SDs = zeros(3), 
                       name='', qIndex=None, jointDampCoeff=0.0):

        super().__init__(
            predBody=predBody, sucBody=sucBody, 
            A_PDp=A_PDp, A_SDs=A_SDs, 
            P_r_PDp=P_r_PDp, S_r_SDs=S_r_SDs, 
            name=name, dof=1, qIndex=qIndex,jointDampCoeff=jointDampCoeff
        )

    def JointFunction(self, q): # -> [Dp_r_DpDs, A_DpDs]
        deltaX = asscalar(q)
        Dp_r_DpDs = array([deltaX,0,0])
        A_DpDs    = eye(3)
        return [Dp_r_DpDs, A_DpDs]
        
    def JointVelocity(self, q, qDot): # -> [Dp_rDot_DpDs, Dp_omega_DpDs]
        # Overwrite generic JointVelocity:
        Dp_rDot_DpDs  = array([qDot,0,0])
        Dp_omega_DpDs = zeros(3)
        return [Dp_rDot_DpDs, Dp_omega_DpDs]
    
    def JointAcceleration(self, q, qDot, qDDot): # -> [Dp_rDDot_DpDs, Dp_omegaDot_DpDs]
        # Overwrite generic JointAcceleration:
        Dp_rDDot_DpDs    = array([qDDot,0,0])
        Dp_omegaDot_DpDs = zeros(3)
        return [Dp_rDDot_DpDs, Dp_omegaDot_DpDs]
    
    def JointJacobian(self, q, qIndex, nq): # -> [S, R]
        # Overwrite generic JointJacobian:
        S = zeros([3,nq])
        R = zeros([3,nq])
        S[:,qIndex] = array([[1,0,0]]).T
        R[:,qIndex] = array([[0,0,0]]).T
        return [S, R]