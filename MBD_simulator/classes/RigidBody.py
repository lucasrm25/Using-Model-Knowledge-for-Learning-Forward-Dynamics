import numpy as np
from numpy import array, zeros, eye, size, sqrt, pi, diag, shape
from numpy.linalg import inv, eig, matrix_power
from scipy.linalg import expm
from .robotics_helpfuns import skew

from vpython import canvas, vector, color, rate, cylinder
from .vpython_ext import vellipsoid

class RigidBody():
    
    def __init__( self, m_B=1, B_I_B=eye(3), I_grav=array([0,0,-9.81]), name:str=''):
        assert np.size(m_B)==1 and np.shape(B_I_B)==(3,3) and np.shape(I_grav)==(3,), 'Wrong argument dimensions'
        
        # link body to joints
        self.childJoints = array([])
        self.parentJoint = array([])
        
        # body properties
        self.name   = name
        self.m_B    = float(m_B)
        self.B_I_B  = array(B_I_B)

        # gravity vector
        self.I_grav = array(I_grav)
        
        # body state
        self.A_IB    = eye(3)      # The rotational orientation of the body B with respect to the inertial frame
        self.B_w_B   = zeros(3)    # The absolute angular velocity [rad/s]
        self.B_dw_B  = zeros(3)    # The absolute angular acceleration  [rad/s^2]
        self.B_r_IB  = zeros(3)    # The displacement of the body's COG [m]
        self.B_v_B   = zeros(3)    # The absolute velocity of the body's COG [m/s]
        self.B_a_B   = zeros(3)    # The absolute acceleration of the body's COG [m/s^2]

    @property
    def nChildren(self):
        return size(self.childJoints)
    
    @property
    def isLeaf(self):
        return self.nChildren == 0
    
    @property
    def isRoot(self):
        return size(self.parentJoint) == 0

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
        self.A_IB   = expm(delta_t*I_omega_IB) @ self.A_IB
        # Doing one-step Euler forward integration for angular velocity:
        self.B_w_B  = self.B_w_B + delta_t * (self.B_dw_B - 0)


    def I_r_IQ( self, B_r_BQ ):
        ''' return position of point of interest Q in inertial coordinates I
        '''
        return self.A_IB @ (self.B_r_IB + B_r_BQ)
    
    def I_v_Q( self, B_r_BQ ):
        ''' return velocity of point of interest Q in inertial coordinates I
        '''
        B_omega_IB = skew(self.B_w_B)
        return self.A_IB @ (self.B_v_B + B_omega_IB @ B_r_BQ)
    
    def I_a_Q( self, B_r_BQ ): # -> I_a_Q
        ''' return acceleration of point of interest Q in inertial coordinates I
        '''
        B_omega_IB = skew(self.B_w_B)
        B_omegaDot_IB = skew(self.B_dw_B)
        return self.A_IB @ (self.B_a_B + (B_omegaDot_IB + matrix_power(B_omega_IB,2) ) @ B_r_BQ)

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
        return self.B_Js_B - skew(B_r_BQ) @ self.B_Jr_B
    
    def I_Js_Q( self, B_r_BQ ):
        ''' return translation jacobian of a point of interest Q in the body B, expressed in inertial coordinates I
            NOTE:
                I_v_Q = I_Js_Q * dq
        '''
        return self.A_IB @ self.B_Js_Q( B_r_BQ )

    def computeNaturalDynamics( self ):
        # Since no external forces or moments are acting, the change of
        # angular momentum and linear moment is zero:
        B_pDot   = zeros(3)
        B_LDot_B = zeros(3)
        # Compute the current angular momentum and the skew symmetric
        # matrix of B_w_B
        B_L_B = self.B_I_B @ self.B_w_B
        B_omega_IB = skew(self.B_w_B)
        # Compute accelerations from the equations of motion of a rigid
        # body.  Note that instead of using inv(B_I_B), we're using the
        # matrix 'devision' '\' that Matlab implements ("...X = A\B is
        # the solution to the equation A*X = B..."):   
        self.B_a_B         = B_pDot / self.m_B
        self.B_dw_B  = inv(self.B_I_B) @ (B_LDot_B - B_omega_IB @ B_L_B)
    

    def _recursiveForwardKinematics( self, nq, B_r_IB=[], A_IB=[], B_w_B=[], B_v_B=[], B_dw_B=[], B_a_B=[], B_Js_B=[], B_Jr_B=[] ):
        '''
            Position and orientation, as well as velocities and accelerations are given by the parent 
            joint and passed in its call of 'recursiveForwardKinematics' 
        '''
        # root is the ground and has no dynamics
        if self.isRoot:
            self.A_IB    = eye(3)
            self.B_w_B   = zeros(3)
            self.B_dw_B  = zeros(3)
            self.B_r_IB  = zeros(3) 
            self.B_v_B   = zeros(3)
            self.B_a_B   = zeros(3)
            self.B_Js_B  = zeros([3,nq])
            self.B_Jr_B  = zeros([3,nq])
        else:
            self.A_IB    = A_IB
            self.B_w_B   = B_w_B
            self.B_dw_B  = B_dw_B
            self.B_r_IB  = B_r_IB
            self.B_v_B   = B_v_B
            self.B_a_B   = B_a_B
            self.B_Js_B  = B_Js_B
            self.B_Jr_B  = B_Jr_B
        
        for childJoint in self.childJoints:
            childJoint._recursiveForwardKinematics(nq, self.B_r_IB, self.A_IB, self.B_w_B, self.B_v_B, self.B_dw_B, self.B_a_B,  self.B_Js_B, self.B_Jr_B)


    def _recursiveComputationOfMfg( self ): # -> [M, f, g]
        '''
            This method requires a model update with all generalized accelerations set to zero
            such that B_a_B and B_dw_B represent bias accelerations and not real accelerations
        '''
        # Compute the components for this body:
        M =   self.B_Js_B.T * self.m_B    @ self.B_Js_B + \
              self.B_Jr_B.T @ self.B_I_B  @ self.B_Jr_B  
        f = - self.B_Js_B.T * self.m_B    @ self.B_a_B - \
              self.B_Jr_B.T @ (self.B_I_B @ self.B_dw_B + skew(self.B_w_B) @ self.B_I_B @ self.B_w_B)
        g =   self.B_Js_B.T @ self.A_IB.T @ self.I_grav * self.m_B + \
              self.B_Jr_B.T @ self.A_IB.T @ zeros(3)

        for childJoint in self.childJoints:
            M_part, f_part, g_part = childJoint.sucBody._recursiveComputationOfMfg()
            M += M_part
            f += f_part
            g += g_part
        return [M, f, g]

    def _printKinTree(self, prefix='', level=1, last=True):
        print(f"{prefix}{'└──' if last else '├──'} ▒ {self.name} ({self.__class__.__name__})")
        prefix += '    ' if last else '|   '
        # print(f"{'  '*level}{level}. Link {self.__class__.__name__}")
        for i,childJoint in enumerate(self.childJoints):
            childJoint._printKinTree(prefix, level+1, last=i==(self.nChildren-1))

    ''' -------------------- GRAPHICS ------------------- '''

    def _recursiveInitGraphicsVPython(self):
        if not self.isRoot:   # for now, ground does not need a graphics representation
            # Inertia ellipse and principal axes
            self.ellsize, self.A_BP = self.getInertiaEllipsoid()
            # create Ellipse object in OPENGL
            self.ellipsoid = vellipsoid(pos=vector(0,0,0), color=color.orange, size=vector(*(self.ellsize*2)))
            # recursive call to other objects in the tree
        for childJoint in self.childJoints:
            childJoint.sucBody._recursiveInitGraphicsVPython()

    def _recursiveUpdateGraphicsVPython(self):
        if not self.isRoot:   # for now, ground does not need a graphics representation
            self.ellipsoid.pos = self.A_IB @ self.B_r_IB
            self.ellipsoid.orientation = self.A_IB @ self.A_BP
        # recursive call to other objects in the tree
        for childJoint in self.childJoints:
            childJoint.sucBody._recursiveUpdateGraphicsVPython()

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
        a = sqrt(2.5/self.m_B*(- I1 + I2 + I3))
        b = sqrt(2.5/self.m_B*(+ I1 - I2 + I3))
        c = sqrt(2.5/self.m_B*(+ I1 + I2 - I3))
        ellsize = array([a,b,c])

        return [ellsize, A_BP]

    def __str__(self):
        matrix2String = lambda m: m.__str__().replace("\n",",")
        s = f'{self.__class__.__name__}: {self.name}\n'
        s += f'   children: {[j.sucBody.name for j in self.childJoints]}\n'
        s += f'   mass:     {self.m_B}\n'
        s += f'   B_I_B:    {matrix2String(self.B_I_B)}\n'
        s += f'   A_IB:     {matrix2String(self.A_IB)}\n'
        s += f'   B_r_IB:   {self.B_r_IB}\n'
        # print(s)
        return s

    def __repr__(self):
        return self.__str__()



class Ground(RigidBody):
    def __init__(self):
        super().__init__(m_B=0, B_I_B=zeros([3,3]), name='Ground')


class Rod(RigidBody):
    def __init__( self, length=1, radius_o=0.01, radius_i=0, density=8000, I_grav=array([0,0,-9.81]), name='' ):
        self.length = length
        self.radius_i = radius_i
        self.radius_o = radius_o
        self.density = density
        volume = pi * (radius_o**2 - radius_i**2) * length
        mass = density * volume
        inertia = mass * diag([0.5*(radius_o**2 + radius_i**2) , 0.25*(radius_o**2 + radius_i**2 + length**2/3), 0.25*(radius_o**2 + radius_i**2 + length**2/3) ])
        super().__init__( m_B=mass, B_I_B=inertia, I_grav=I_grav, name=name )
    
    def _recursiveInitGraphicsVPython(self):        
        if not self.isRoot:   # for now, ground does not need a graphics representation
            # create Ellipse object in OPENGL
            self.cylinder = cylinder(pos=vector(*(self.A_IB @ self.B_r_IB)), color=color.orange, axis=vector(self.length,0,0), radius=self.radius_o)
            # recursive call to other objects in the tree
        for childJoint in self.childJoints:
            childJoint.sucBody._recursiveInitGraphicsVPython()

    def _recursiveUpdateGraphicsVPython(self):
        if not self.isRoot:   # for now, ground does not need a graphics representation
            self.cylinder.pos  = vector( *(self.A_IB @ (self.B_r_IB - array([self.length/2,0,0])) ) )
            self.cylinder.axis = vector( *( self.A_IB[:,0] * self.length ) )
        # recursive call to other objects in the tree
        for childJoint in self.childJoints:
            childJoint.sucBody._recursiveUpdateGraphicsVPython()


class Ellipsoid(RigidBody):
    def __init__( self, rx=1, ry=0.01, rz=0, density=8000, I_grav=array([0,0,-9.81]), name=''):
        # ellipsoid principal diameters
        self.rx = rx
        self.ry = ry
        self.rz = rz
        volume = 4/3 * pi * rx * ry * rz
        mass = density * volume
        inertia = mass/5 * diag([ry**2+rz**2, rx**2+rz**2, rx**2+ry**2])
        super().__init__(m_B=mass, B_I_B=inertia, I_grav=I_grav,  name=name)