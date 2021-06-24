'''
'''

from numpy import append, array, size, ones, zeros, eye, ndarray, absolute, meshgrid
from numpy.linalg import matrix_power, norm
from .robotics_helpfuns import skew
from .RigidBody import RigidBody


class BodyOnSurfaceBilateralConstraint():
    ''' Class that represents the constraint of a point in a body sliding on a surface
    '''

    def __init__(self, predBody:RigidBody, A_PDp=eye(3),  P_r_PDp=zeros(3),
                       surface_fun   = None,
                       surface_fun_J = None,
                       surface_fun_H = None,
                       wn=10, ksi=1):
        '''
        Args:
            - predBody: body, whose point of interest will be constrained
            - A_PDp, P_r_PDp: rotation matrix and position of the point of interest of the body predBody
            - surface_fun: function handle that represents the surface equation f(I_r_IDp) = 0
            - surface_fun_J: Jacobian of f(I_r_IDp) w.r.t. I_r_IDp
            - surface_fun_H: Hessian of f(I_r_IDp) w.r.t. I_r_IDp
            - wn, ksi: Baumgarte stabilization parameters
        '''
        # link pred and suc bodies to current joint
        self.predBody = predBody        
        # store static properties
        self.A_PDp   = A_PDp
        self.P_r_PDp = P_r_PDp
        # store surface information handles
        self.surface_fun = surface_fun
        self.surface_fun_J = surface_fun_J
        self.surface_fun_H = surface_fun_H
        # Baumgarte stabilization
        self.K = wn**2       # Baumgarte proportional stabilization constant
        self.D = 2*ksi*wn    # Baumgarte derivative stabilization constant

        assert  (
            self.surface_fun(zeros(3)).ndim == 1 and
            self.surface_fun_J(zeros(3)).ndim == 2 and
            self.surface_fun_H(zeros(3)).ndim == 2 
        ),'Surface function have wrong dimension'

    def getConstraintTerms(self):
        ''' Compute constraint terms

            I_c      = f(I_r_IDp) 
                      0
            
            I_cDot   = Jf(I_r_IDp) @      I_v_Dp 
                     = Jf(I_r_IDp) @ ( I_Js_Dp @ dq )
                     = 0

            I_cDDot := I_v_Dp.T @ Hf(I_r_IDp) @ I_v_Dp + Jf(I_r_IDp) @             I_a_IDp
                     = I_v_Dp.T @ Hf(I_r_IDp) @ I_v_Dp + Jf(I_r_IDp) @ ( I_Js_Dp @ ddq + I_sigmas_Dp )
                     rearanging
                     = ( Jf(I_r_IDp) @ I_Js_Dp ) @ ddq + I_v_Dp.T @ Hf(I_r_IDp) @ I_v_Dp + Jf(I_r_IDp) @ I_sigmas_Dp
                     =           J_lambda         @ ddq +                     sigma_lambda
                     = 0
        '''
        # inertial position of point of interest in inertial coordinate
        I_r_IDp = self.predBody.I_r_IQ (B_r_BQ=self.P_r_PDp)
        I_v_Dp = self.predBody.I_v_Q  (B_r_BQ=self.P_r_PDp)
        f  = self.surface_fun  ( I_r_IDp )
        Jf = self.surface_fun_J( I_r_IDp )
        Hf = self.surface_fun_H( I_r_IDp )

        # calculate displacement and velocity constraint violation
        I_c    = f
        I_cDot = Jf @ I_v_Dp
        
        # calculate I_cDDot matrices
        J_lambda = Jf @ self.predBody.I_Js_Q( B_r_BQ=self.P_r_PDp )
        sigma_lambda = I_v_Dp.T @ Hf @ I_v_Dp + Jf @ self.predBody.I_a_Q( B_r_BQ=self.P_r_PDp )

        # add Baumgarte stabilization
        nc = I_c.size
        sigma_lambda += eye(nc)*self.D @ I_cDot + eye(nc)*self.K @ I_c

        return [J_lambda, sigma_lambda]


    ''' -------------------- GRAPHICS ------------------- '''


    def initGraphics(self):
        from vpython import canvas, vector, color, rate, helix, arrow, vertex, quad
        from .vpython_ext import vellipsoid
        import numpy as np
        from scipy.optimize import fsolve

        n = 10
        x = np.linspace(-2,2,n)
        z = np.linspace(-2,2,n)
        xv, zv = np.meshgrid(x,z)
        yv = 0*zv
        vertices = np.ndarray((n,n), dtype=object)
        for i in range(n):
            for j in range(n):
                yv[i,j] = fsolve( lambda y: self.surface_fun([xv[i,j],y,zv[i,j]]).item(), 0 )
                vertices[i,j] = vertex(pos=vector(xv[i,j],yv[i,j],zv[i,j]))
        
        quads = []
        for i in range(n-1):
            for j in range(n-1):
                quads.append( quad( vs=[vertices[i,j],vertices[i+1,j],vertices[i+1,j+1],vertices[i,j+1]] ) )

    
    def updateGraphics(self):
        return
        # origin = self.predBody.A_IB @ (self.predBody.B_r_IB + self.P_r_PDp )
        # target = self.sucBody.A_IB @ (self.sucBody.B_r_IB + self.S_r_SDs() )
        # self.arrow.pos = vector( *(origin) )
        # self.arrow.axis = vector( *(target-origin) )

