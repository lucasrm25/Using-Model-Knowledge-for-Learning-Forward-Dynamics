'''  
    Runge-Kutta ODE fixed-step solvers that runs 100% on Pytorch, in which device={'cpu','cuda'} can also be chosen
'''

import torch
import numpy as np
import math
from tqdm import tqdm
from classes.torch_utils import *

class ODEtorch():
    def __init__(self, device='cpu', dtype=torch.float):
        self.device = device
        self.dtype = dtype
        self.RK4 = torch.tensor( 
                [[0,0,0,0,0],
                [.5,.5,0,0,0],
                [.5,0,.5,0,0],
                [1,0,0,1,0],
                [0,1/6,1/3,1/3,1/6]] , device=device)
        self.RK2 = torch.tensor( 
               [[0,0,0], 
               [.5,.5,0], 
               [0,0,1]] , device=device)
        self.RK1 = torch.tensor( 
               [[0,0], 
               [0,1.]] , device=device)

    def ode1(self, f, x0, t_end, dt):
        return self.RK(self.RK1, f, x0, t_end, dt);

    def ode2(self, f, x0, t_end, dt):
        return self.RK(self.RK2, f, x0, t_end, dt);

    def ode4(self, f, x0, t_end, dt):
        return self.RK(self.RK4, f, x0, t_end, dt);

    def RK(self, ButcherT, f, x0, t_end, dt):
        ''' ButcherT: <n,n> Butcher tableau of order n-1
        f: <@(t,x)> ode function, e.g. f = @(t,x) lambda*x;
        '''
        batchSize, dim = x0.shape
        ord = np.size(ButcherT,0)-1
        N = math.ceil(t_end/dt)
        t = torch.zeros(N, device=self.device)
        x = torch.zeros((batchSize,dim,N), device=self.device)
        x[:,:,0] = x0
        for it in tqdm(range(1,N)):
            t[it] = t[it-1] + dt
            K = torch.zeros( (batchSize, ord, dim), device=self.device)
            for ki in range(ord):
                K[:,ki,:] = f( t[it-1] + ButcherT[ki,0]*dt , x[:,:,it-1] + dt * bvm(ButcherT[ki,1:], K) )
            # x[:,:,it] = x[:,:,it-1] + dt * (ButcherT[-1,1:] @ K).T
            x[:,:,it] = x[:,:,it-1] + dt * bvm(ButcherT[-1,1:] , K)
        t = t[1:]
        x = x[:,:,1:]
        return t, x




if __name__ == "__main__":
    device = 'cuda'

    f = lambda t,x: torch.tensor([-0.5*x[0], -0.3*x[0]])
    x0 = torch.tensor([3,3], device=device)
    t_end = 10
    dt = 0.01
    
    ode = ODEtorch(device='cuda')
    
    t,x = ode.ode4(f,x0,t_end,dt)

    t = t.cpu()
    x = x.cpu()

    import matplotlib.pyplot as plt
    plt.plot(t,x)
    plt.show()