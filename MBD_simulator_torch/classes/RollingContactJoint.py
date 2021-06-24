import torch
from .GenericJoint import GenericJoint
from .torch_utils import skew

class RotationalJoint(GenericJoint):

    def __init__(self, predBody, sucBody,
                       A_PDp = torch.eye(3), A_SDs = torch.eye(3),
                       P_r_PDp = torch.zeros(3), S_r_SDs = torch.zeros(3), r=0, 
                       name='', qIndex=None, jointDampCoeff=0.0):
                       
        super().__init__(
            predBody=predBody, sucBody=sucBody, 
            A_PDp=A_PDp, A_SDs=A_SDs, 
            P_r_PDp=P_r_PDp, S_r_SDs=S_r_SDs, 
            name=name, dof=1, qIndex=qIndex, jointDampCoeff=torch.tensor(jointDampCoeff)
        )
        # rolling contact radius
        self.r = r

    def JointFunction(self, q): # -> [Dp_r_DpDs, A_DpDs]
        device, batchSize = inputInfo(q)
        Dp_r_DpDs = torch.zeros((batchSize, 3), device=device)
        Dp_r_DpDs[:,0] = - self.r * q.squeeze()
        A_DpDs = batch(rotZ, q)
        return [Dp_r_DpDs, A_DpDs]
        
    def JointVelocity(self, q, qDot): # -> [Dp_rDot_DpDs, Dp_omega_DpDs]
        device, batchSize = inputInfo(q)
        # Overwrite generic JointVelocity:
        Dp_rDot_DpDs = torch.zeros((batchSize, 3), device=device)
        Dp_rDot_DpDs[:,0] = - self.r * qDot.squeeze()
        Dp_omega_DpDs = torch.zeros((batchSize, 3), device=device)
        Dp_omega_DpDs[:,2] = qDot.squeeze()
        return [Dp_rDot_DpDs, Dp_omega_DpDs]
    
    def JointAcceleration(self, q, qDot, qDDot): # -> [Dp_rDDot_DpDs, Dp_omegaDot_DpDs]
        device, batchSize = inputInfo(q)
        # Overwrite generic JointAcceleration:

        Dp_rDDot_DpDs = torch.zeros((batchSize, 3), device=device)
        Dp_rDDot_DpDs[:,0] = - self.r * qDDot.squeeze()
        Dp_omegaDot_DpDs = torch.zeros((batchSize, 3), device=device)
        Dp_omegaDot_DpDs[:,2] = qDDot.squeeze()
        return [Dp_rDDot_DpDs, Dp_omegaDot_DpDs]
    
    def JointJacobian(self, q, qIndex, nq): # -> [S, R]
        device, batchSize = inputInfo(q)
        # Overwrite generic JointJacobian:
        S = torch.zeros((batchSize,3,nq), device=q.device)
        R = torch.zeros((batchSize,3,nq), device=q.device)
        S[:,:,qIndex] = torch.tensor([[-self.r,0,0]], device=q.device).T
        R[:,:,qIndex] = torch.tensor([[0,0,1]], device=q.device).T
        return [S, R]