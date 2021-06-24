import sys, os
sys.path.append( os.path.join( os.path.dirname(__file__), os.path.pardir ) )

import torch
import numpy as np
import math
from classes.RigidBody import RigidBody, Ground, Rod
from classes.MultiRigidBody import MultiRigidBody
from classes.RotationalJoint import RotationalJoint
from classes.SpringDamper import SpringDamper
from classes.BodyOnSurfaceBilateralConstraint import BodyOnSurfaceBilateralConstraint
from classes.ODEtorch import ODEtorch

''' ------------------------------------------------------------------------
Setup PyTorch
------------------------------------------------------------------------ '''

torch.set_printoptions(precision=4,threshold=1000,linewidth=500)

# ! VERY IMPORTANT: change torch to double precision
torch.set_default_tensor_type(torch.DoubleTensor)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'\nUsing device: {device}\n')


''' ------------------------------------------------------------------------
Setup MBD system
------------------------------------------------------------------------ '''

I_grav = torch.tensor([0,-9.81,0], device=device)
ground = Ground()

link1 = Rod(length=1, radius_o=0.02, radius_i=0., I_grav=I_grav)
link2 = Rod(length=1, radius_o=0.02, radius_i=0., I_grav=I_grav)

joint1 = RotationalJoint(ground,link1, A_PDp=torch.eye(3), A_SDs=torch.eye(3), P_r_PDp=torch.tensor([0.,0,0]),   S_r_SDs= torch.tensor([-0.5,0,0]))
joint2 = RotationalJoint(link1, link2, A_PDp=torch.eye(3), A_SDs=torch.eye(3), P_r_PDp=torch.tensor([0.5,0,0]), S_r_SDs= torch.tensor([-0.5,0,0]))

surface_fun   = lambda r: r[:,1:2]+1   # c(r) = y + 1 = 0
surface_fun_J = lambda r: torch.tensor([0,1.,0], device=device).repeat(r.shape[0],1)
surface_fun_H = lambda r: torch.zeros((3,3), device=device).repeat(r.shape[0],1,1)
surfaceConstraint = BodyOnSurfaceBilateralConstraint(
    link2, 
    P_r_PDp=torch.tensor([0.5,0,0]), 
    surface_fun=surface_fun,
    surface_fun_J=surface_fun_J,
    surface_fun_H=surface_fun_H 
)

# set generalized coordinate indices
joint1.qIndex = 0
joint2.qIndex = 1

pendulum = MultiRigidBody(
    ground=ground, 
    bilateralConstraints=[surfaceConstraint]
).to(device)

print(pendulum)

nq = pendulum.nq


''' ------------------------------------------------------------------------
Set Initial Position
------------------------------------------------------------------------ '''

with torch.no_grad():
    # set initial conditions
    q = torch.tensor([[0,-90]]).to(device) * math.pi/180
    dq = 0*q
    ddq = 0*q
    pendulum.forwardKinematics(q=q, qDot=dq, qDDot=ddq)
    

''' ------------------------------------------------------------------------
Simulate
------------------------------------------------------------------------ '''

def odefun(t, y):
    q, qDot = y[:,:nq], y[:,nq:]
    qDDot,_ = pendulum.forwardDynamics( q=q, qDot=qDot )
    return torch.cat((qDot,qDDot),dim=-1)


# initial conditions
q0, dq0, ddq0  = pendulum.getJointStates()

# simulate
tf = 20
fps = 60
# odesol = solve_ivp( odefun, t_span=[0,tf], t_eval=arange(0,tf,1/fps), y0=concatenate((q0,dq0)).squeeze(), method='RK45', dense_output=True, events=None )
odesol = ODEtorch(device=device)

with torch.no_grad():
    t, x = odesol.ode2( f=odefun, x0=torch.cat((q0,dq0),-1), t_end=tf, dt=0.01 )

t = t.cpu().numpy()
x = x.cpu().numpy()

from matplotlib import pyplot as plt
plt.figure()
batchIdx = 0
plt.plot(t, x[batchIdx,0,:]*180/np.pi, label='joint1.q')
plt.plot(t, x[batchIdx,1,:]*180/np.pi, label='joint2.q')
plt.legend()
plt.grid(True)
plt.show()

