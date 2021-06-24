'''
    Defines Spring/Damper component that interacts with the multi-body system
'''

# from numpy import append, array, size, ones, zeros, eye, ndarray, absolute
# from numpy.linalg import matrix_power, norm
import torch
from .RigidBody import RigidBody
from .LinkedConstraintModule import LinkedConstraintModule
from .torch_utils import *

class SpringDamper(LinkedConstraintModule):
    '''
    Abstract Generic Joint class
    '''
    def __init__(self, predBody:RigidBody, sucBody:RigidBody, 
                       A_PDp = torch.eye(3), A_SDs = torch.eye(3),
                       P_r_PDp = torch.zeros(3), S_r_SDs = torch.zeros(3),
                       d0 = 0, K=100, D=5,
                       radius=0.05, coils=20,
                       name=''):
        super().__init__(predBody,sucBody)
        
        # init parameters
        self.A_PDp   = param(A_PDp)
        self.A_SDs   = param(A_SDs)
        self.P_r_PDp = param(P_r_PDp)
        self.S_r_SDs = param(S_r_SDs)
        self.d0      = param(d0)
        self.K       = param(K)
        self.D       = param(D)
        self.name    = name

        # store graphic properties
        self.radius = radius
        self.coils = coils

    def computationOfTau(self) -> torch.tensor:
        '''
            Compute generalized force
            TODO: Check from which body the update is coming and calculate:
                    - A_IDp, Dp_Js_Dp, Dp_Jr_Dp and A_IDs, Ds_Js_Ds, Ds_Jr_Ds
                
                  such that I_Js_D = I_Js_Dp - I_Js_Ds
        '''
        # calculate displacement and velocity vectors between attaching points of the spring/damper
        I_r_DpDs = self.getPredBody().I_r_IQ(B_r_BQ=self.P_r_PDp) - self.getSucBody().I_r_IQ (B_r_BQ = self.S_r_SDs )
        I_v_DpDs = self.getPredBody().I_v_Q(B_r_BQ=self.P_r_PDp) - self.getSucBody().I_v_Q (B_r_BQ = self.S_r_SDs )
        
        # calculate jacobian I_Js_D, such that: dDot = I_Js_D * qDot, which is the ratio of the spring/damper 
        # displacement to the generalized coordinates
        I_Js_D = self.getPredBody().I_Js_Q( B_r_BQ=self.P_r_PDp ) - self.getSucBody().I_Js_Q( B_r_BQ=self.S_r_SDs )
        
        # calculate generalized forces
        tau = bmv( bT(I_Js_D) , - self.K * I_r_DpDs * (1-self.d0/torch.norm(I_r_DpDs, dim=1)).view(-1,1) - self.D * I_v_DpDs )
        return tau



    ''' -------------------- GRAPHICS ------------------- '''


    def initGraphics(self):
        from vpython import canvas, vector, color, rate, helix
        from .vpython_ext import vellipsoid
        self.helix = helix(pos=vector(0,0,0), axis=vector(1,0,0), radius=self.radius, coils=self.coils)
    
    def updateGraphics(self):
        origin = self.getPredBody().A_IB @ (self.getPredBody().B_r_IB + self.P_r_PDp )
        target = self.getSucBody().A_IB @ (self.getSucBody().B_r_IB + self.S_r_SDs )
        self.helix.pos = vector( *(origin) )
        self.helix.axis = vector( *(target-origin) )

