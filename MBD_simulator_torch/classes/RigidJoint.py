'''
TODO:
    [ ] This file needs some fixes. I am not sure how the dynamics will work with this class.
        Second problem is how to handle the generalized coordinates.
        Should we skip this gen. coordinate? 
'''

# from numpy import array, ones, zeros, eye, cos, sin, asscalar
# from .GenericJoint import GenericJoint
# from .torch_utils import skew

# class RigidJoint(GenericJoint):
#     ''' Defines a joint that is rigid, i.e. allows no displacement or rotation between bodies
#     '''

#     def __init__(self, predBody, sucBody,
#                        A_PDp = eye(3), A_SDs = eye(3),
#                        P_r_PDp = zeros(3), S_r_SDs = zeros(3), 
#                        name='', qIndex=None, jointDampCoeff=0.0):

#         super().__init__(
#             predBody=predBody, sucBody=sucBody, 
#             A_PDp=A_PDp, A_SDs=A_SDs, 
#             P_r_PDp=P_r_PDp, S_r_SDs=S_r_SDs, 
#             name=name, qIndex=qIndex,jointDampCoeff=jointDampCoeff
#         )
#         # init generalized coordinates -> joint angle
#         self.q = self.qDot = self.qDDot = array([0])
#         self.dof = 0

#     def JointFunction(self, q): # -> [Dp_r_DpDs, A_DpDs]
#         Dp_r_DpDs = zeros(3)
#         A_DpDs    = eye(3)
#         return [Dp_r_DpDs, A_DpDs]
        
#     def JointVelocity(self, q, qDot): # -> [Dp_rDot_DpDs, Dp_omega_DpDs]
#         # Overwrite generic JointVelocity:
#         Dp_rDot_DpDs  = zeros(3)
#         Dp_omega_DpDs = zeros(3)
#         return [Dp_rDot_DpDs, Dp_omega_DpDs]
    
#     def JointAcceleration(self, q, qDot, qDDot): # -> [Dp_rDDot_DpDs, Dp_omegaDot_DpDs]
#         # Overwrite generic JointAcceleration:
#         Dp_rDDot_DpDs    = zeros(3)
#         Dp_omegaDot_DpDs = zeros(3)
#         return [Dp_rDDot_DpDs, Dp_omegaDot_DpDs]
    
#     def JointJacobian(self, q, qIndex, nq): # -> [S, R]
#         # Overwrite generic JointJacobian:
#         S = zeros([3,nq])
#         R = zeros([3,nq])
#         return [S, R]