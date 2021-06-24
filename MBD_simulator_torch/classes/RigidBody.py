# import numpy as np
# from numpy import array, zeros, eye, size, sqrt, pi, diag, shape
# from numpy.linalg import inv, eig, matrix_power
# from scipy.linalg import expm
import torch
from .torch_utils import *
import math

torch.set_default_tensor_type(torch.DoubleTensor)

class LinkedBodyModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # link body to joints
        self._childJoints = []     # torch.nn.ModuleList()
        self._parentJoint = [None]     # storing this variable in a list protects it from pytorch recursion when registering this Module

    def getChildJoints(self):
        ''' returns a list with child joints'''
        return self._childJoints

    def appendChildJoint(self, child):
        self._childJoints.append(child)

    def getParentJoint(self):
        return self._parentJoint[0]

    def setParentJoint(self, parent):
        self._parentJoint = [parent]

    def nChildren(self):
        return len(self.getChildJoints())
    
    def isLeaf(self):
        return self.nChildren() == 0
    
    def isRoot(self):
        return self.getParentJoint() is None
        # return self.parentJoint is None

class RigidBody(LinkedBodyModule):
    
    def __init__( self, m_B:torch.Tensor=torch.tensor(1.), B_I_B:torch.Tensor=torch.eye(3), I_grav:torch.Tensor=torch.tensor([0,0,-9.81]), name:str=''):
        super().__init__()
        
        # basic type asserts
        assert isinstance(m_B,torch.Tensor) and isinstance(B_I_B,torch.Tensor) and isinstance(I_grav,torch.Tensor), 'Wrong types'
        assert B_I_B.size()==(3,3) and I_grav.size()==(3,) , 'Wrong argument dimensions'

        # body properties
        self.name   = name
        self.m_B    = ConstrainedParameter(m_B, Interval(m_B-1., m_B+1.), requires_grad=False)
        self.B_I_B  = param(B_I_B)
        # gravity vector
        self.I_grav = param(I_grav)

        # body state
        self.A_IB    = None # torch.eye(3)      # The rotational orientation of the body B with respect to the inertial frame
        self.B_w_B   = None # torch.zeros(3)    # The absolute angular velocity [rad/s]
        self.B_dw_B  = None # torch.zeros(3)    # The absolute angular acceleration  [rad/s^2]
        self.B_r_IB  = None # torch.zeros(3)    # The displacement of the body's COG [m]
        self.B_v_B   = None # torch.zeros(3)    # The absolute velocity of the body's COG [m/s]
        self.B_a_B   = None # torch.zeros(3)    # The absolute acceleration of the body's COG [m/s^2]

        # print(list(self.buffers()))

    # def to(self, device):
    #     self.device = device
    #     return super().to(device)

    def integrationStep( self, delta_t=0.001 ):
        # Using the M = skew(w) function which is defined below, we compute the 
        # skew symmetric matrices of omega_B in I and in B-coordinates: 
        B_omega_IB = skew(self.B_w_B)
        I_omega_IB = skew(self.A_IB @ self.B_w_B)
        
        # Doing one-step Euler forward integration for linear motion
        # while taking into account that we do so in a moving coordinate system:  
        self.B_r_IB = self.B_r_IB + delta_t * (self.B_v_B - B_omega_IB @ self.B_r_IB)
        self.B_v_B  = self.B_v_B  + delta_t * (self.B_a_B - B_omega_IB @ self.B_v_B)
        # Using the matrix-exponential to compute A_IB exactly over the course of one integration time-step.
        self.A_IB   = torch.matrix_exp(delta_t*I_omega_IB) @ self.A_IB
        # Doing one-step Euler forward integration for angular velocity:
        self.B_w_B  = self.B_w_B + delta_t * (self.B_dw_B - 0)

    def I_r_IQ( self, B_r_BQ ):
        ''' return position of point of interest Q in inertial coordinates I
        '''
        return bmv( self.A_IB , self.B_r_IB + B_r_BQ )
    
    def I_v_Q( self, B_r_BQ ):
        ''' return velocity of point of interest Q in inertial coordinates I
        '''
        B_omega_IB = batch(skew,self.B_w_B)
        return bmv( self.A_IB , self.B_v_B + bmv( B_omega_IB , B_r_BQ) )
    
    def I_a_Q( self, B_r_BQ ): # -> I_a_Q
        ''' return acceleration of point of interest Q in inertial coordinates I
        '''
        B_omega_IB = batch(skew,self.B_w_B)
        B_omegaDot_IB = batch(skew,self.B_dw_B)
        return bmv( self.A_IB , self.B_a_B + bmv(B_omegaDot_IB + torch.matrix_power(B_omega_IB,2) , B_r_BQ) )

    def I_w_Q( self, B_r_BQ ):
        ''' return angular velocity of point of interest Q in inertial coordinates I
        '''
        return self.A_IB @ self.B_w_B
    
    def I_dw_Q( self, B_r_BQ ): # -> I_a_Q
        ''' return angular acceleration of point of interest Q in inertial coordinates I
        '''
        return self.A_IB @ self.B_dw_B

    def B_Js_Q( self, B_r_BQ ):
        ''' return translation jacobian of a point of interest Q in the body B, expressed in body coordinates B
            NOTE:
                B_v_Q =   B_v_B + skew(B_w_B) @ B_r_BQ
                      = ( B_Js_B - skew(B_r_BQ)    @ B_Jr_B   ) @ dq 
                     :=               B_Js_Q                 @ dq
        '''
        return self.B_Js_B - bmm( batch(skew,B_r_BQ) , self.B_Jr_B )
    
    def I_Js_Q( self, B_r_BQ ):
        ''' return translation jacobian of a point of interest Q in the body B, expressed in inertial coordinates I
            NOTE:
                I_v_Q = I_Js_Q * dq
        '''
        return bmm( self.A_IB , self.B_Js_Q(B_r_BQ) )

    # def computeNaturalDynamics( self ):
    #     # Since no external forces or moments are acting, the change of
    #     # angular momentum and linear moment is zero:
    #     B_pDot   = torch.zeros(3)
    #     B_LDot_B = torch.zeros(3)
    #     # Compute the current angular momentum and the skew symmetric matrix of B_w_B
    #     B_L_B = self.B_I_B @ self.B_w_B
    #     B_omega_IB = skew(self.B_w_B)
    #     # Compute accelerations from the equations of motion of a rigid
    #     # body. 
    #     self.B_a_B  = B_pDot / self.m_B()
    #     self.B_dw_B = torch.solve( (B_LDot_B - B_omega_IB @ B_L_B) , self.B_I_B )[0]
    
    def _recursiveForwardKinematics( self, nq, batchSize=1, B_r_IB=[], A_IB=[], B_w_B=[], B_v_B=[], B_dw_B=[], B_a_B=[], B_Js_B=[], B_Jr_B=[] ):
        '''
            Position and orientation, as well as velocities and accelerations are given by the parent 
            joint and passed in its call of 'recursiveForwardKinematics' 
        '''
        # root is the ground and has no dynamics
        if self.isRoot():
            device = self.m_B.device
            self.A_IB    = beye(batchSize, 3, device=device)
            self.B_w_B   = torch.zeros( (batchSize,3),      device=device)
            self.B_dw_B  = torch.zeros( (batchSize,3),      device=device)
            self.B_r_IB  = torch.zeros( (batchSize,3),      device=device) 
            self.B_v_B   = torch.zeros( (batchSize,3),      device=device)
            self.B_a_B   = torch.zeros( (batchSize,3),      device=device)
            self.B_Js_B  = torch.zeros( (batchSize, 3,nq),  device=device)
            self.B_Jr_B  = torch.zeros( (batchSize, 3,nq),  device=device)
        else:
            self.A_IB    = A_IB
            self.B_w_B   = B_w_B
            self.B_dw_B  = B_dw_B
            self.B_r_IB  = B_r_IB
            self.B_v_B   = B_v_B
            self.B_a_B   = B_a_B
            self.B_Js_B  = B_Js_B
            self.B_Jr_B  = B_Jr_B
        
        for childJoint in self.getChildJoints():
            childJoint._recursiveForwardKinematics(nq, batchSize, self.B_r_IB, self.A_IB, self.B_w_B, self.B_v_B, self.B_dw_B, self.B_a_B,  self.B_Js_B, self.B_Jr_B)

    def _recursiveComputationOfMfg( self ): # -> [M, f, g]
        '''
            This method requires a model update with all generalized accelerations set to zero
            such that B_a_B and B_dw_B represent bias accelerations and not real accelerations
        '''
        # Compute the components for this body:
        M =   bmm( bT(self.B_Js_B) * self.m_B()  , self.B_Js_B ) + \
              bmm( bmm( bT(self.B_Jr_B) , self.B_I_B ) , self.B_Jr_B )  
        f = - bmv( bT(self.B_Js_B) * self.m_B()  , self.B_a_B ) + \
            - bmv( bT(self.B_Jr_B) , bmv(self.B_I_B, self.B_dw_B) + bmv(bmm(batch(skew,self.B_w_B), self.B_I_B ), self.B_w_B) )
        g =   bmv( bmm(bT(self.B_Js_B), bT(self.A_IB)), self.I_grav ) * self.m_B() #+ \
            #   self.B_Jr_B.T @ self.A_IB.T @ torch.zeros(3)

        for childJoint in self.getChildJoints():
            M_part, f_part, g_part = childJoint.getSucBody()._recursiveComputationOfMfg()
            M += M_part
            f += f_part
            g += g_part
        return [M, f, g]

    def printKinTree(self, prefix='', level=1, last=True):
        print(f"{prefix}{'└──' if last else '├──'} ▒ {self.name} ({self.__class__.__name__})")
        prefix += '    ' if last else '|   '
        # print(f"{'  '*level}{level}. Link {self.__class__.__name__}")
        for i,childJoint in enumerate(self.getChildJoints()):
            childJoint.printKinTree(prefix, level+1, last=i==(self.nChildren()-1))

    ''' -------------------- GRAPHICS ------------------- '''

    def _recursiveInitGraphicsVPython(self):
        from vpython import canvas, vector, color, rate, cylinder
        from .vpython_ext import vellipsoid

        if not self.isRoot():   # for now, ground does not need a graphics representation
            # Inertia ellipse and principal axes
            self.ellsize, self.A_BP = self.getInertiaEllipsoid()
            # create Ellipse object in OPENGL
            self.ellipsoid = vellipsoid(pos=vector(0,0,0), color=color.orange, size=vector(*(self.ellsize*2)))
            # recursive call to other objects in the tree
        for childJoint in self.getChildJoints():
            childJoint.getSucBody()._recursiveInitGraphicsVPython()

    def _recursiveUpdateGraphicsVPython(self):
        if not self.isRoot():   # for now, ground does not need a graphics representation
            self.ellipsoid.pos = self.A_IB @ self.B_r_IB
            self.ellipsoid.orientation = self.A_IB @ self.A_BP
        # recursive call to other objects in the tree
        for childJoint in self.getChildJoints():
            childJoint.getSucBody()._recursiveUpdateGraphicsVPython()

    def getInertiaEllipsoid(self): # -> []
        '''
            returns:
                - A_BP: rotation matrix from principal axes to body coordinates
                - ellsize: vector with the 3 ellipse principal radius corresponding to the Inertia matrix
        '''
        # Compute the inertia axis:
        D, V = eig(self.B_I_B)

        A_BP = V

        I1, I2, I3 = D
        # Define the main axis of the ellipsoid:
        a = torch.sqrt(2.5/self.m_B()*(- I1 + I2 + I3))
        b = torch.sqrt(2.5/self.m_B()*(+ I1 - I2 + I3))
        c = torch.sqrt(2.5/self.m_B()*(+ I1 + I2 - I3))
        ellsize = torch.tensor([a,b,c])

        return [ellsize, A_BP]

    def __str__(self):
        matrix2String = lambda m: m.__str__().replace("\n",",")
        s  = f'{self.__class__.__name__}: {self.name}\n'
        s += f'   children: {[j.getSucBody().name for j in self.getChildJoints()]}\n'
        s += f'   m_B:      {self.m_B()}\n'
        s += f'   B_I_B:    {matrix2String(self.B_I_B)}\n'
        s += f'   A_IB:     {matrix2String(self.A_IB)}\n'
        s += f'   B_r_IB:   {self.B_r_IB}\n'
        # print(s)
        return s

    def __repr__(self):
        return self.__str__()


class Ground(RigidBody):
    def __init__(self):
        super().__init__(m_B=torch.tensor(0.), B_I_B=torch.zeros([3,3]), name='Ground')


class Rod(RigidBody):
    def __init__( self, length=1, radius_o=0.01, radius_i=0, density=8000, I_grav=torch.tensor([0,0,-9.81]), name='' ):
        self.length = length
        self.radius_i = radius_i
        self.radius_o = radius_o
        self.density = density
        volume = math.pi * (radius_o**2 - radius_i**2) * length
        mass = density * volume
        inertia = mass * torch.diag(torch.tensor([0.5*(radius_o**2 + radius_i**2) , 0.25*(radius_o**2 + radius_i**2 + length**2/3), 0.25*(radius_o**2 + radius_i**2 + length**2/3) ]))
        super().__init__( m_B=mass, B_I_B=inertia, I_grav=I_grav, name=name )
    
    def _recursiveInitGraphicsVPython(self):
        from vpython import canvas, vector, color, rate, cylinder
        from .vpython_ext import vellipsoid
        
        if not self.isRoot():   # for now, ground does not need a graphics representation
            # create Ellipse object in OPENGL
            self.cylinder = cylinder(pos=vector(*(self.A_IB @ self.B_r_IB)), color=color.orange, axis=vector(self.length,0,0), radius=self.radius_o)
            # recursive call to other objects in the tree
        for childJoint in self.getChildJoints():
            childJoint.getSucBody()._recursiveInitGraphicsVPython()

    def _recursiveUpdateGraphicsVPython(self):
        if not self.isRoot():   # for now, ground does not need a graphics representation
            self.cylinder.pos  = vector( *(self.A_IB @ (self.B_r_IB - array([self.length/2,0,0])) ) )
            self.cylinder.axis = vector( *( self.A_IB[:,0] * self.length ) )
        # recursive call to other objects in the tree
        for childJoint in self.getChildJoints():
            childJoint.getSucBody()._recursiveUpdateGraphicsVPython()


class Ellipsoid(RigidBody):
    def __init__( self, rx=1, ry=0.01, rz=0, density=8000, I_grav=torch.tensor([0,0,-9.81]), name=''):
        # ellipsoid principal diameters
        self.rx = rx
        self.ry = ry
        self.rz = rz
        volume = 4/3 * pi * rx * ry * rz
        mass = density * volume
        inertia = mass/5 * diag([ry**2+rz**2, rx**2+rz**2, rx**2+ry**2])
        super().__init__(m_B=mass, B_I_B=inertia, I_grav=I_grav,  name=name)