'''
'''

from numpy import append, array, size, ones, zeros, eye, ndarray, absolute
from numpy.linalg import matrix_power, norm
from .robotics_helpfuns import skew
from .RigidBody import RigidBody

from vpython import canvas, vector, color, rate, helix, arrow
from classes.vpython_ext import vellipsoid

class PositionBilateralConstraint():
    ''' Bilateral Constraint making 2 points in different bodies coincide (one might be ground)
    '''

    def __init__(self, predBody:RigidBody, sucBody:RigidBody, 
                       A_PDp = eye(3), A_SDs = eye(3),
                       P_r_PDp = zeros(3), S_r_SDs = zeros(3),
                       wn=10, ksi=1):
        # link pred and suc bodies to current joint
        self.predBody = predBody
        self.sucBody  = sucBody
        
        # store static properties
        self.A_PDp   = A_PDp
        self.A_SDs   = A_SDs
        self.P_r_PDp = P_r_PDp
        self.S_r_SDs = S_r_SDs
        self.K       = wn**2       # Baumgarte proportional stabilization constant
        self.D       = 2*ksi*wn    # Baumgarte derivative stabilization constant


    def getConstraintTerms(self):
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

        # calculate displacement and velocity constraint violation
        I_c    = self.predBody.I_r_IQ(B_r_BQ=self.P_r_PDp) - self.sucBody.I_r_IQ( B_r_BQ = self.S_r_SDs )
        I_cDot = self.predBody.I_v_Q(B_r_BQ=self.P_r_PDp)  - self.sucBody.I_v_Q(  B_r_BQ = self.S_r_SDs )
        
        # calculate I_cDDot matrices
        # NOTE: calling I_a_Q should return the bias acceleration and not the acceleration when ddq is set to zero,
        # which is the case when getConstraintTerms is being called
        J_lambda     = self.predBody.I_Js_Q( B_r_BQ=self.P_r_PDp ) - self.sucBody.I_Js_Q( B_r_BQ=self.S_r_SDs )
        sigma_lambda = self.predBody.I_a_Q( B_r_BQ=self.P_r_PDp )   - self.sucBody.I_a_Q( B_r_BQ=self.S_r_SDs )

        # add Baumgarte stabilization
        nc = sigma_lambda.size
        sigma_lambda += eye(nc)*self.D @ I_cDot + eye(nc)*self.K @ I_c

        return [J_lambda, sigma_lambda]


    ''' -------------------- GRAPHICS ------------------- '''


    def initGraphics(self):
        self.arrow = arrow(pos=vector(0,0,0), axis=vector(1,0,0)) # , shaftwidth=shaftwidth
    
    def updateGraphics(self):
        origin = self.predBody.A_IB @ (self.predBody.B_r_IB + self.P_r_PDp )
        target = self.sucBody.A_IB @ (self.sucBody.B_r_IB + self.S_r_SDs )
        self.arrow.pos = vector( *(origin) )
        self.arrow.axis = vector( *(target-origin) )

