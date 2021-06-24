import numpy as np
import matplotlib.pyplot as plt

''' 
    -------------------------------- TRAJECTORY PLANNING ------------------------------------- 
    ------------------------------------------------------------------------------------------
'''

class TrajectoryPlanning():
    ''' Trajectory planning with trapezoidal velocity profile
    '''
    def __init__(self, ddx_max):
        '''
        Arguments:
            - ddx_max: desired maximum acceleration in each direction
        '''
        self.ddx_max = ddx_max
        self.ti = np.array([])
        self.tf = np.array([])
        self.xi = np.array([]).reshape(0,3)
        self.xf = np.array([]).reshape(0,3)
        self.xfun= np.array([])

    def add(self,ti:float, tf:float, xi:np.ndarray, xf:np.ndarray):
        ''' Add point to the trajectory
        '''
        dt = tf - ti
        # minimum acceleration required to move from xi to xf in dt seconds
        ddx_min_req = 4*np.abs(xf-xi)/dt**2 + 1e-10
        # correct maximum acceleration if this value is below the minimum
        ddx_n = [np.max((ddx_min_i,self.ddx_max)) for ddx_min_i in ddx_min_req]
        # fix numpy.sign. By definition we want sign(0)=1, otherwise we get an error in the algorithm
        ddx_n = np.array([i if i!=0.0 else 1 for i in ddx_n * np.sign(xf-xi)])

        # time at the end of the parabolic segment
        tc = dt/2 - 0.5*np.sqrt( (dt**2*ddx_n - 4*(xf-xi))/ddx_n )
        tc[np.isnan(tc)] = 0

        # lambda function that evaluates [x,dx,ddx](t)
        xfun = lambda t: np.array([
            (          t-ti <= 0       ) * [xi_i,                              0,                                      0]          +
            (0       < t-ti <= tc_i    ) * [xi_i+0.5*ddx_n_i*(t-ti)**2,        ddx_n_i*(t-ti),                         ddx_n_i]    +
            (tc_i    < t-ti <= dt-tc_i ) * [xi_i+ddx_n_i*tc_i*((t-ti)-tc_i/2), ddx_n_i*tc_i,                           0]          +
            (dt-tc_i < t-ti <= dt      ) * [xf_i-0.5*ddx_n_i*(dt-(t-ti))**2,   ddx_n_i*tc_i-ddx_n_i*((t-ti)-dt+tc_i), -ddx_n_i]    + 
            (     dt < t-ti            ) * [xf_i,                              0,                                      0]
            for tc_i, xi_i, xf_i, ddx_n_i in zip(tc, xi, xf, ddx_n)
        ]).T

        self.ti   = np.concatenate((self.ti,[ti]))
        self.tf   = np.concatenate((self.tf,[tf])) 
        self.xi   = np.vstack((self.xi,xi)) 
        self.xf   = np.vstack((self.xf,xf)) 
        self.xfun = np.concatenate((self.xfun,[xfun])) 

    def __call__(self,t):
        idx = 0 if t<self.ti[0] else np.where(t>=self.ti)[0][-1]
        return self.xfun[idx](t)
    
    def plot_trajectory(self ):
        tv = np.linspace(np.min(self.ti), np.max(self.tf), int((self.tf[-1]-self.ti[0])*40) )
        # evaluate trajectory for time vector
        xdata = np.array([self(t) for t in tv])
        # plot trajectories
        fig = plt.figure(figsize=(10,6))
        #
        plt.subplot(311); plt.grid(True)
        plotpos = plt.plot(tv, xdata[:,0,:])
        plt.gca().set_prop_cycle(None)	# reset color cycle
        plt.scatter( self.ti, self.xi[:,0], 60, edgecolors='k', zorder=100) # facecolor='r',
        plt.scatter( self.ti, self.xi[:,1], 60, edgecolors='k', zorder=100) # facecolor='r',
        plt.scatter( self.ti, self.xi[:,2], 60, edgecolors='k', zorder=100) # facecolor='r',
        plt.gca().set_prop_cycle(None)	# reset color cycle
        plt.scatter( self.tf[-1], self.xf[-1,0], 60, edgecolors='k', zorder=100) # facecolor='r',
        plt.scatter( self.tf[-1], self.xf[-1,1], 60, edgecolors='k', zorder=100) # facecolor='r',
        plt.scatter( self.tf[-1], self.xf[-1,2], 60, edgecolors='k', zorder=100) # facecolor='r',
        # plt.scatter( *np.stack([(ti,xi_c) for ti,xi in zip(self.ti,self.xi) for xi_c in xi]).T , c='k')
        # plt.scatter( *np.stack([(tf,xf_c) for tf,xf in zip(self.tf,self.xf) for xf_c in xf]).T , c='k')
        plt.ylabel('pos [m]')
        plt.legend(plotpos, ['$x$','$y$','$z$']).set_zorder(1000)
        #
        plt.subplot(312); plt.grid(True)
        plotvel = plt.plot(tv, xdata[:,1,:])
        plt.ylabel('vel [m/2]')
        plt.legend(plotvel, ['$\dot x$','$\dot y$','$\dot z$'])
        #
        plt.subplot(313); plt.grid(True)
        plotacc = plt.plot(tv, xdata[:,2,:])
        plt.ylabel('acc [m/s^2]'); plt.xlabel('time [s]')
        plt.legend(plotacc, ['$\ddot x$','$\ddot y$','$\ddot z$'])
        #
        plt.suptitle('Task-space Trajetory Planning')
        plt.tight_layout(rect=[0, 0.0, 1, 0.95])
        # plt.show()
        return fig

def __test_TrajectoryPlanning():
    tp = TrajectoryPlanning(ddx_max=1)
    xrand = np.random.rand(4,3)
    t = [[1,3],[3,5],[10,12]]
    [tp.add(t[i][0],t[i][1],xrand[i],xrand[i+1]) for i in range(3)]
    tp.plot_trajectory()
    plt.show()
