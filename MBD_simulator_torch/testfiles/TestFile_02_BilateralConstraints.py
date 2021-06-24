import sys, os
sys.path.append( os.path.join( os.path.dirname(__file__), os.path.pardir ) )

import math
import torch
import numpy as np
from classes.RigidBody import RigidBody, Ground, Rod
from classes.MultiRigidBody import MultiRigidBody
from classes.RotationalJoint import RotationalJoint
from classes.SpringDamper import SpringDamper
from classes.PositionBilateralConstraint import PositionBilateralConstraint
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

link1 = Rod(name='l1', length=1, radius_o=0.02, radius_i=0.01, I_grav=I_grav)
link2 = Rod(name='l2', length=1, radius_o=0.02, radius_i=0.01, I_grav=I_grav)
link3 = Rod(name='l3', length=1, radius_o=0.02, radius_i=0.01, I_grav=I_grav)
link4 = Rod(name='l4', length=1, radius_o=0.02, radius_i=0.01, I_grav=I_grav)

joint1 = RotationalJoint(ground, link1, name='g-l1',  A_PDp=torch.eye(3), A_SDs=torch.eye(3), P_r_PDp=torch.tensor([0.,0.,0.]),S_r_SDs= torch.tensor([-0.5,0,0]))
joint2 = RotationalJoint(link1,  link2, name='l1-l2', A_PDp=torch.eye(3), A_SDs=torch.eye(3), P_r_PDp=torch.tensor([0.5,0,0]), S_r_SDs= torch.tensor([-0.5,0,0]))
joint3 = RotationalJoint(link2,  link3, name='l2-l3', A_PDp=torch.eye(3), A_SDs=torch.eye(3), P_r_PDp=torch.tensor([0.5,0,0]), S_r_SDs= torch.tensor([-0.5,0,0]))
joint4 = RotationalJoint(link1,  link4, name='l1-l4', A_PDp=torch.eye(3), A_SDs=torch.eye(3), P_r_PDp=torch.tensor([0.5,0,0]), S_r_SDs= torch.tensor([-0.5,0,0]))

springDamper1 = SpringDamper(ground, link4, P_r_PDp=torch.tensor([1.,1.,0.]), S_r_SDs=torch.tensor([0.5,0,0]), K=50, D=5, d0=0)

constraint = PositionBilateralConstraint(link3, ground, P_r_PDp=torch.tensor([0.5,0,0]), S_r_SDs=torch.tensor([2.,0,0]))

# set generalized coordinate indices
joint1.qIndex = 0
joint2.qIndex = 1
joint3.qIndex = 2
joint4.qIndex = 3

# create multi-rigid-body object
pendulum = MultiRigidBody(
    name='Pendulum', 
    ground=ground, 
    springDampers=[springDamper1], 
    bilateralConstraints=[constraint]
).to(device)

print(pendulum)

nq = pendulum.nq


''' ------------------------------------------------------------------------
Test library
------------------------------------------------------------------------ '''

with torch.no_grad():
    # set initial conditions
    q = torch.tensor([[60,-60,-60,-60-90], [50,-50,-50,-50-80]]).to(device) * math.pi/180
    dq = q / 10
    ddq = q / 20
    pendulum.forwardKinematics(q=q, qDot=dq, qDDot=ddq)

pendulum.linkList[-1].B_r_IB
pendulum.linkList[-1].A_IB

# pendulum.linkList[-1].B_r_IB
# tensor([[-0.3660,  0.5000,  0.0000],
#         [-0.1428,  0.7660,  0.0000]], device='cuda:0')
# pendulum.linkList[-1].A_IB
# tensor([[[-1.2905e-07,  1.0000e+00,  0.0000e+00],
#          [-1.0000e+00, -1.2320e-07,  0.0000e+00],
#          [ 0.0000e+00,  0.0000e+00,  1.0000e+00]],

#         [[ 1.7365e-01,  9.8481e-01,  0.0000e+00],
#          [-9.8481e-01,  1.7365e-01,  0.0000e+00],
#          [ 0.0000e+00,  0.0000e+00,  1.0000e+00]]], device='cuda:0')


''' ------------------------------------------------------------------------
Set Initial Position
------------------------------------------------------------------------ '''

with torch.no_grad():
    # set initial conditions
    q = torch.tensor([[60,-60,-60,-60-90], [50,-50,-50,-50-80]]).to(device) * math.pi/180
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
plt.plot(t, x[batchIdx,2,:]*180/np.pi, label='joint3.q')
plt.legend()
plt.grid(True)
plt.show()

