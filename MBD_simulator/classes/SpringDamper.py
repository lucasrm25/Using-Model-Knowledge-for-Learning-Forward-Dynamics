'''
    Defines Spring/Damper component that interacts with the multi-body system
'''

from numpy import append, array, size, ones, zeros, eye, ndarray, absolute
from numpy.linalg import matrix_power, norm
from .robotics_helpfuns import skew
from .RigidBody import RigidBody

from vpython import canvas, vector, color, rate, helix
from .vpython_ext import vellipsoid

class SpringDamper():
    '''
    Abstract Generic Joint class
    '''

    def __init__(self, predBody:RigidBody, sucBody:RigidBody, 
                       A_PDp = eye(3), A_SDs = eye(3),
                       P_r_PDp = zeros(3), S_r_SDs = zeros(3),
                       d0 = 0, K=100, D=5,
                       radius=0.05, coils=20):
        # link pred and suc bodies to current joint
        self.predBody = predBody
        self.sucBody  = sucBody
        
        # store static properties
        self.A_PDp   = A_PDp
        self.A_SDs   = A_SDs
        self.P_r_PDp = P_r_PDp
        self.S_r_SDs = S_r_SDs
        self.d0      = d0
        self.K       = K
        self.D       = D

        # store graphic properties
        self.radius = radius
        self.coils = coils

    def computationOfTau(self) -> ndarray:
        '''
            Compute generalized force
            TODO: Check from which body the update is coming and calculate:
                    - A_IDp, Dp_Js_Dp, Dp_Jr_Dp and A_IDs, Ds_Js_Ds, Ds_Jr_Ds
                
                  such that I_Js_D = I_Js_Dp - I_Js_Ds
        '''
        # calculate displacement and velocity vectors between attaching points of the spring/damper
        I_r_DpDs = self.predBody.I_r_IQ(B_r_BQ=self.P_r_PDp) - self.sucBody.I_r_IQ (B_r_BQ = self.S_r_SDs )
        I_v_DpDs = self.predBody.I_v_Q(B_r_BQ=self.P_r_PDp) - self.sucBody.I_v_Q (B_r_BQ = self.S_r_SDs )
        
        # calculate jacobian I_Js_D, such that: dDot = I_Js_D * qDot, which is the ratio of the spring/damper 
        # displacement to the generalized coordinates
        I_Js_D = self.predBody.I_Js_Q( B_r_BQ=self.P_r_PDp ) - self.sucBody.I_Js_Q( B_r_BQ=self.S_r_SDs )
        
        # calculate generalized forces
        tau = I_Js_D.T @ ( - self.K * I_r_DpDs * (1-self.d0/norm(I_r_DpDs)) - self.D * I_v_DpDs)
        return tau



    ''' -------------------- GRAPHICS ------------------- '''


    def initGraphics(self):
        self.helix = helix(pos=vector(0,0,0), axis=vector(1,0,0), radius=self.radius, coils=self.coils)
    
    def updateGraphics(self):
        origin = self.predBody.A_IB @ (self.predBody.B_r_IB + self.P_r_PDp )
        target = self.sucBody.A_IB @ (self.sucBody.B_r_IB + self.S_r_SDs )
        self.helix.pos = vector( *(origin) )
        self.helix.axis = vector( *(target-origin) )

