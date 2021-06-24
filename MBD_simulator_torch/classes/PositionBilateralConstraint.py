'''
'''

# from numpy import append, array, size, ones, zeros, eye, ndarray, absolute
# from numpy.linalg import matrix_power, norm
import torch
from classes.RigidBody import RigidBody
from classes.LinkedConstraintModule import LinkedConstraintModule
from classes.torch_utils import *

from vpython import canvas, vector, color, rate, helix, arrow
from classes.vpython_ext import vellipsoid

class PositionBilateralConstraint(LinkedConstraintModule):
    ''' Bilateral Constraint making 2 points in different bodies coincide (one might be ground)
    '''

    def __init__(self, predBody:RigidBody, sucBody:RigidBody, 
                       A_PDp = torch.eye(3), A_SDs = torch.eye(3),
                       P_r_PDp = torch.zeros(3), S_r_SDs = torch.zeros(3),
                       wn=10, ksi=1,
                       name=''):
        super().__init__(predBody, sucBody)
        
        # init param
        self.A_PDp   = param(A_PDp)
        self.A_SDs   = param(A_SDs)
        self.P_r_PDp = param(P_r_PDp)
        self.S_r_SDs = param(S_r_SDs)
        self.K       = param(wn**2)       # Baumgarte proportional stabilization constant
        self.D       = param(2*ksi*wn)    # Baumgarte derivative stabilization constant
        self.name    = name

    def I_c(self):
        ''' Return constraint violation'''
        return self.getPredBody().I_r_IQ(B_r_BQ=self.P_r_PDp) - self.getSucBody().I_r_IQ( B_r_BQ = self.S_r_SDs )
    
    def I_cDot(self):
        ''' Return constraint violation velocity'''
        return self.getPredBody().I_v_Q(B_r_BQ=self.P_r_PDp)  - self.getSucBody().I_v_Q(  B_r_BQ = self.S_r_SDs )       

    def getConstraintTerms(self, useBaumgarteStab=True):
        ''' Compute constraint terms

            I_c      = predbody.I_r_IDp - sucbody.I_r_IDs
                     = 0
            I_cDot   = I_v_IDp - I_v_IDs
                     = ( I_Js_Dp - I_Js_Dp ) @ dq
                    :=      J_lambda           @ dq 
                     = 0
            I_cDDot := J_lambda * ddq + sigma_lambda 
                     = 0
        '''
        
        # calculate I_cDDot matrices
        # NOTE: calling I_a_Q should return the bias acceleration and not the acceleration when ddq is set to zero,
        # which is the case when getConstraintTerms is being called
        J_lambda     = self.getPredBody().I_Js_Q( B_r_BQ=self.P_r_PDp ) - self.getSucBody().I_Js_Q( B_r_BQ=self.S_r_SDs )
        sigma_lambda = self.getPredBody().I_a_Q( B_r_BQ=self.P_r_PDp )  - self.getSucBody().I_a_Q( B_r_BQ=self.S_r_SDs )

        if useBaumgarteStab:
            # add Baumgarte stabilization
            nc = sigma_lambda.shape[-1]
            I = torch.eye(nc, device=sigma_lambda.device)
            sigma_lambda += bmv(I*self.D, self.I_cDot()) + bmv(I*self.K, self.I_c())

        return [J_lambda, sigma_lambda]
        

    ''' -------------------- GRAPHICS ------------------- '''


    def initGraphics(self):
        self.arrow = arrow(pos=vector(0,0,0), axis=vector(1,0,0)) # , shaftwidth=shaftwidth
    
    def updateGraphics(self):
        origin = self.getPredBody().A_IB @ (self.getPredBody().B_r_IB + self.P_r_PDp )
        target = self.getSucBody().A_IB @ (self.getSucBody().B_r_IB + self.S_r_SDs )
        self.arrow.pos = vector( *(origin) )
        self.arrow.axis = vector( *(target-origin) )

