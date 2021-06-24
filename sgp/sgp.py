'''
This file contains the implementation of the Structured-GP as well as a vanilla GP.
For more details, please refer to the paper: Using Model Knowledge for Learning Forward Dynamics.

Inspired on the paper "Learning Constrained Dynamics with Gauss’ Principle adhering Gaussian Processes" - A. Rene Geist and Sebastian Trimpe
'''

import os, sys
sys.path.append( os.path.join( os.path.dirname(__file__), os.path.pardir ) )

import gc
import math
import abc
from enum import Flag, auto
from typing import *
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import dill
import gpytorch
import torch
from gpytorch import Module, settings, add_jitter
from gpytorch.kernels import IndexKernel, Kernel, MultitaskKernel, RBFKernel, ScaleKernel, MaternKernel, LinearKernel
from gpytorch.means import ZeroMean, ConstantMean, Mean, MultitaskMean
from gpytorch.constraints import *
from gpytorch.models import GP, ExactGP
from utils.StructArray import StructTorchArray, StructNumpyArray
from MBD_simulator_torch.classes.RigidBody import *
from MBD_simulator_torch.classes.BodyOnSurfaceBilateralConstraint import *
from MBD_simulator_torch.classes.MultiRigidBody import *
from MBD_simulator_torch.classes.RotationalJoint import *
from addict import Dict

torch.set_printoptions(precision=4,threshold=1000,linewidth=500)

'''--------------
help functions
--------------'''

def beye(batchSize, n, device):
    ''' batch eye matrix 
    '''
    return torch.eye(n,device=device).repeat(batchSize,1,1)

def beye_like(tensor):
    ''' tensor must have the size <batchSize, n, n>
    '''
    assert tensor.dim() == 3 and tensor.shape[-1] == tensor.shape[-1]
    batchSize, n, device = tensor.shape[0], tensor.shape[-1], tensor.device
    return beye(batchSize, n, device)

def cond_number_np(cov):
    eigvals = np.linalg.eig(cov.cpu().detach())[0]
    print(f'eig_max:{eigvals.max()}  eig_min:{eigvals.min()}  cond_numb: {(eigvals.max() / eigvals.min()).item()}')

def cond_number(cov):
    eigvals = cov.symeig().eigenvalues
    if eigvals.dim() > 1:
        return print( eigvals.max(dim=1).values / eigvals.min(dim=1).values )
    else:
        return print(f'eig_max:{eigvals.max()}  eig_min:{eigvals.min()}  cond_numb: {(eigvals.max() / eigvals.min()).item()}')

def cleanGPUcache():
    # clean GPU cache
    gc.collect()
    torch.cuda.empty_cache()

def printParameterList(model, sformat = '{:60s} {:8s} {:20s} {:30s} {:50s}'):
    ''' Print model parameters
    '''
    print('\nParameter list:')
    print(sformat.format('Name','Type','Size','True Value', 'Constraint'))
    print(sformat.format('-'*40,'-'*6,'-'*15,'-'*20,'-'*40))
    pretty = lambda list_: [f"{element:.4f}" for element in list_.flatten()]
    for name, param, constraint in model.named_parameters_and_constraints():
        # if param.requires_grad:
        print(sformat.format(
            name, 
            type(param.data).__name__, 
            list(param.size()).__str__(), 
            ' '.join( pretty(param if constraint is None else constraint.transform(param)) ),
            constraint.__str__()
        )) 
    print('\n')

def assertDataset(dataset:StructTorchArray):
    ''' Check wheter the dataset keys that are relevant for S-GP have correct dimensions
    '''
    N  = len(dataset)
    nq = dataset.q.shape[1] 
    nc = dataset.A.shape[1]
    datasetSizes = {
        't': (N,1), 
        'k': (N,1),
        'q': (N,nq),
        'dq': (N,nq),
        'tau':(N,nq),
        'ddq':(N,nq),
        'qn': (N,nq),
        'dqn': (N,nq),
        'M': (N,nq,nq),
        'f': (N,nq),
        'g': (N,nq),
        'A': (N,nc,nq),
        'b': (N,nc),
        'Ml':(N,nc,nc),
        'L': (N,nq,nc),
        'T': (N,nq,nq),
        'contacts': (N,1)
    }
    for k,v in dataset.items():
        assert isinstance(v,torch.Tensor), 'Dataset keys are not Pytorch Tensors'
        if k in datasetSizes.keys(): # assert k in datasetSizes.keys(), f'Key {k} missing in the Dataset'
            assert v.size() == datasetSizes[k], f'Key {k} is of size {v.size()}, expected {datasetSizes[k]}'

def unflatten_MultiTaskCovarMatrix(covar, d=1):
    N = covar.shape[0] // d
    return covar.reshape(N, d, N, d).permute(0,2,1,3)

def flatten_MultiTaskCovarMatrix(covar):
    ''' Reshape augmented multitask covariance matrix of shape <N1,N2,D1,D2> to <N1*D1,N2*D2>
    '''
    N1, N2, D1, D2 = covar.shape        
    return covar.permute(0,2,1,3).contiguous().view(D1*N1,D2*N2)

def flatten(fun):
    ''' Decorator for flatenning multi-task covariance matrices 
    '''
    # @functools.wraps(fun)
    def wrapper(*args, **kwargs): 
        return flatten_MultiTaskCovarMatrix( covar = fun(*args, **kwargs) )
    return wrapper

@flatten 
def kronProd(mat1,mat2): 
    return torch.einsum('ij,bc->ijbc', mat1, mat2)

def plotMultiTaskKernel(covar, nq):
    lik = unflatten_MultiTaskCovarMatrix(covar, nq).cpu().detach()
    fig, axs = plt.subplots(nq, nq, figsize=(8,8), gridspec_kw = {'wspace':0, 'hspace':0})
    for i in range(nq):
        for j in range(nq):
            axs[i,j].imshow( lik[:,:,i,j] )
            axs[i,j].axis('off')
            axs[i,j].set_xticklabels([])
            axs[i,j].set_yticklabels([])
            # axs[i,j].set_aspect('equal')
    plt.tight_layout( pad=0.)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
      
'''--------------
GP Core Classes
--------------'''

class MultitaskMultivariateNormal():
    def __init__(self, mean, covar):
        assert mean.shape[0]*mean.shape[1] == covar.shape[0] == covar.shape[1], 'Mean and covar shapes do not match'
        self.mean = mean
        self.covariance_matrix = covar
        self.num_tasks = mean.shape[1]
        self.X0 = None
    
    @property
    def variance(self):
        return torch.diag(self.covariance_matrix).view(-1,self.num_tasks)
    
    def confidence_region(self, level:float=2):
        std = torch.sqrt(self.variance)
        return self.mean - level*std, self.mean + level*std

    def sample(self, size:int=1):
        return np.random.multivariate_normal( self.mean.squeeze() , self.covariance_matrix , size ).T

    def loglikelihood(self, value, verbose=False):
        err = (value - self.mean).view(-1,1).contiguous()
        covar = gpytorch.add_jitter( self.covariance_matrix , jitter_val=1*1e-8)
        quatterm = err.T @ torch.solve(err, covar).solution
        logdetcovar = torch.logdet(covar)
        ll = - 0.5*quatterm - 0.5*logdetcovar - 0.5*len(covar)*math.log(2*math.pi)
        if verbose:
            print(f'log|K|: {logdetcovar.item():10.2f} {" ":5s} |err|^2_{{K^-1}}: {quatterm.item():10.2f} {" ":5s}',end='')
        return ll

    def to(self, device):
        self.mean = self.mean.to(device)
        self.covariance_matrix = self.covariance_matrix.to(device)
        return self

class MultiTaskLikelihood(Module):
    def __init__(self, num_tasks:int, noise_constraint=gpytorch.constraints.GreaterThan(1e-4), requires_grad:bool=True):
        super().__init__()
        self.num_tasks = num_tasks
        self.register_parameter(
            name = 'raw_noise', 
            parameter = torch.nn.Parameter(torch.zeros(num_tasks), requires_grad=requires_grad)
        )
        self.register_constraint(
            'raw_noise',
            noise_constraint
        )
        # self.raw_noise = torch.nn.Parameter(torch.zeros(num_tasks), requires_grad=True)
        # self.raw_noise_constraint = noise_constraint

    @property
    def noise(self):
        return self.raw_noise_constraint.transform(self.raw_noise)

    @noise.setter
    def noise(self, value):
        self.raw_noise[:] = self.raw_noise_constraint.inverse_transform(torch.tensor(value))

    def __call__(self, dist:MultitaskMultivariateNormal):
        noise = self.raw_noise_constraint.transform(self.raw_noise)
        N = int(dist.mean.shape[0])
        dist.covariance_matrix = dist.covariance_matrix + torch.diag_embed(noise.repeat(N))
        return dist

class QUANT(Flag):
    # inputs
    q     = auto()
    dq    = auto()
    tau   = auto()
    # outputs for constrained dynamics
    ddq   = auto()
    ddqs  = auto()
    dqn   = auto()
    qn    = auto()
    # outputs for unconstrained dynamics
    ddqa  = auto()
    ddqas = auto()
    dqan  = auto()
    qan   = auto()
    # extra quantities
    lamb  = auto()
    # extra variables for simple GP regression example
    x     = auto()
    y     = auto()

class NormStandDist():
    ''' Normalize/Standardize multi-task Gaussian distributions
    '''
    def __init__(self, data, normData=True, stdData=True):
        '''
        Arguments:
            - data: <N,nq> array-like structure. N is the number of points and nq the number of dimension 
        '''
        self.normData = normData
        self.stdData  = stdData
        self.mean = data.mean(dim=-2)       if self.normData else torch.zeros(data.shape[1], device=data.device)
        self.std  = data.std(dim=-2, unbiased=False) + 1e-6 if self.stdData  else torch.ones(data.shape[1], device=data.device)
        self.nq = self.mean.shape[0]

    def apply_mean(self, mean):
        return ((mean - self.mean) / self.std)

    def reverse_mean(self, mean):
        return (mean * self.std + self.mean)

    def apply_covar(self, covar):
        N1, N2 = int(covar.shape[0]/self.nq), int(covar.shape[1]/self.nq)
        return (1/self.std.squeeze()).repeat(N1).reshape(-1,1) * covar * (1/self.std.squeeze()).repeat(N2)
        # stdTransform1 = (1/self.std[0]).repeat(N1).diag_embed()
        # stdTransform2 = (1/self.std[0]).repeat(N2).diag_embed()
        # return stdTransform1 @ covar @ stdTransform2.T

    def reverse_covar(self, covar):
        N1, N2 = int(covar.shape[0]/self.nq), int(covar.shape[1]/self.nq)
        return (self.std.squeeze()).repeat(N1).reshape(-1,1) * covar * (self.std.squeeze()).repeat(N2)
        # stdTransform1 = (self.std[0]).repeat(N1).diag_embed()
        # stdTransform2 = (self.std[0]).repeat(N2).diag_embed()
        # return stdTransform1 @ covar @ stdTransform2.T

    def apply_dist(self, dist:MultitaskMultivariateNormal):
        return MultitaskMultivariateNormal(
            self.apply_mean(dist.mean),
            self.apply_covar(dist.covariance_matrix)
        )

    def reverse_dist(self, dist:MultitaskMultivariateNormal):
        return MultitaskMultivariateNormal(
            self.reverse_mean(dist.mean),
            self.reverse_covar(dist.covariance_matrix)
        )

    @classmethod
    def transform(cls, data, normData=True, stdData=True):
        obj = cls(data, normData=normData, stdData=stdData)
        return obj.apply_mean(data), obj

class GPprediction():
    def __init__(self, likelihood:MultitaskMultivariateNormal, train_targets, USE_LU=True):
        self.USE_LU = USE_LU # if set to False, use Cholesky factorization
        err = (train_targets - likelihood.mean).view(-1,1).contiguous()
        if self.USE_LU:
            self.LU = torch.lu(likelihood.covariance_matrix)
            self.alpha = torch.lu_solve(err, *self.LU )
        else:  
            self.L = torch.cholesky(likelihood.covariance_matrix, upper=False)
            self.alpha = torch.triangular_solve( torch.triangular_solve(err, self.L, upper=False).solution, self.L.T, upper=True).solution

    def predict(self, test_mean, train_test_cov, test_test_cov):
        if self.USE_LU:
            predictive_mean  = test_mean + ( train_test_cov.T @ self.alpha ).view(*test_mean.shape)
            predictive_covar = test_test_cov - train_test_cov.T @ torch.lu_solve( train_test_cov, *self.LU )
        else:
            v = torch.triangular_solve(train_test_cov, self.L, upper=False).solution
            predictive_mean  = test_mean + ( train_test_cov.T @ self.alpha ).view(*test_mean.shape)
            predictive_covar = test_test_cov - v.T @ v
        return MultitaskMultivariateNormal(predictive_mean, predictive_covar)

'''--------------
Kernel and Mean Classes
--------------'''

class MTRBFKernel(Kernel):
    ''' This class defines a multi-task (RBF) kernel. Each output dimension is treated as an 
    independent square-exponential kernel.
    Calling this class results in a <N1,N2,D1,D2> covariance matrix
    '''
    def __init__(self, num_tasks=1, ard_num_dims=1, lengthscale_constraint=None, outputscale_constraint=None):
        super().__init__()
        self.SE_covar_module = torch.nn.ModuleList([
            ScaleKernel(
                RBFKernel(
                    ard_num_dims = ard_num_dims, 
                    lengthscale_constraint = lengthscale_constraint
                ),
                outputscale_constraint = outputscale_constraint
            )
            for _ in range(num_tasks)
        ])

    def __call__(self, feat1, feat2):
        ''' Evaluates the multi-task kernel
        '''
        return torch.diag_embed( torch.stack( [
            d.forward( feat1, feat2) 
            for d in self.SE_covar_module
        ], dim=-1) )

class SGPKernel(Kernel, abc.ABC):
    ''' GP-squared (Gauss' principle adhering GP) Kernel class
    
        Constrained acceleration (Gauss principle adhering GP)
        h(x) = ddq(x) = L(x)*b(x) + T(x)*M(x)^-1*(Fa(x) + Fz(x))
        h(x) ~ GP( L(x)*b(x) + T(x)*mu_abar(x) , T(x)*K_abar(x,x')*T(x') )
    '''
    def __init__(self, nq:int, dt):
        super().__init__()
        self.dt = dt
        self.nq = nq

        self.linearOperators_from_F = {
            # unconstrained outputs
            QUANT.ddqa:  lambda dataset:                dataset.Minv,  #beye_like(dataset.M),
            QUANT.ddqas: lambda dataset: dataset.ScFa @ dataset.Minv,
            QUANT.dqan:  lambda dataset: self.dt      * dataset.Minv, # beye_like(dataset.M) ,
            QUANT.qan:   lambda dataset: self.dt**2   * dataset.Minv, # beye_like(dataset.M) ,
            # constrained outputs
            QUANT.ddq:   lambda dataset:                dataset.T @ dataset.Minv,
            QUANT.ddqs:  lambda dataset: dataset.ScFa @ dataset.T @ dataset.Minv,
            QUANT.dqn:   lambda dataset: self.dt      * dataset.T @ dataset.Minv,
            QUANT.qn:    lambda dataset: self.dt**2   * dataset.T @ dataset.Minv,
            # extra quantitities:
            QUANT.lamb:  lambda dataset: - dataset.Ml @ dataset.A @ dataset.Minv
        }

    # @abc.abstractclassmethod
    def cov_Fa_Fa(self, dataset1, dataset2):
        raise NotImplementedError()

    # @abc.abstractclassmethod
    def cov_Fz_Fz(self, dataset1, dataset2):
        raise NotImplementedError()

    def cov_F_F(self, dataset1, dataset2):
        return self.cov_Fa_Fa(dataset1, dataset2) + self.cov_Fz_Fz(dataset1, dataset2)

    @flatten
    def __call__(self, quant1:QUANT, quant2:QUANT, dataset1, dataset2):
        ''' evaluate covariance between quant1 and quant2 
        '''
        linearoperator1 = self.linearOperators_from_F[quant1](dataset1)
        linearoperator2 = self.linearOperators_from_F[quant2](dataset2)

        if (quant1 | quant2) & (QUANT.ddqas | QUANT.ddqa | QUANT.dqan | QUANT.qan):
            covar_F_F = self.cov_Fa_Fa(dataset1, dataset2)
        else: # quant1 & (QUANT.ddqs | QUANT.ddq | QUANT.dqn | QUANT.qn) and quant2 & (QUANT.ddqs | QUANT.ddq | QUANT.dqn | QUANT.qn)
            covar_F_F = self.cov_F_F(dataset1, dataset2)
        
        # Mulitask covariance linear transformation  -->   T1 * Cov * T2^T
        cov_quant1_quant2 = torch.einsum('iab,ijbc,jdc->ijad', 
            linearoperator1,
            covar_F_F, 
            linearoperator2
        )
        return cov_quant1_quant2

class SGPKernel_Fa_SE(SGPKernel):
    ''' S-GP Kernel with prior on generalized forces 'Fa' and squared-exponential (SE) kernel
    ''' 
    def __init__(self, nq:int, dt):
        super().__init__(nq=nq, dt=dt)

        # ! general kernel for learning the generalized forces F
        self.SE_covar_module = torch.nn.ModuleList([
            ScaleKernel(
                RBFKernel( # RBFKernel(    MaternKernel(nu=3/2, 
                    ard_num_dims = nq*3, 
                    lengthscale_constraint = None
                ),
                outputscale_constraint = None
            )
            for _ in range(nq)
        ])
        # init hyperparameters
        for j in range(nq):
            self.SE_covar_module[j].base_kernel.lengthscale = 0.1
            self.SE_covar_module[j].outputscale             = 0.1

    def cov_F_F(self, dataset1, dataset2):
        ''' Returns augmented <N1,N2,D1,D2> covariance matrix COV(F(feat1_n1)_d1, F(feat2_n2)_d2)
        '''
        covar_F_F = torch.diag_embed( torch.stack( [
            d.forward(
                torch.cat((dataset1.q_sn, dataset1.dq_sn, dataset1.tau_s), axis=-1), 
                torch.cat((dataset2.q_sn, dataset2.dq_sn, dataset2.tau_s), axis=-1)
            ) for d in self.SE_covar_module
        ], dim=-1) )
        return covar_F_F

class SGPMean(Mean):
    ''' GP-squared (Gauss' principle adhering GP) Mean class
    
        Constrained acceleration (Gauss principle adhering GP)
        h(x) = ddq(x) = L(x)*b(x) + T(x)*abar(x)
        h(x) ~ GP( L(x)*b(x) + T(x)*mu_abar(x) , T(x)*K_abar(x,x')*T(x') )
    '''

    def __init__(self, nq:int, dt, use_Fa_mean=False):
        super().__init__( )
        self.dt = dt
        self.use_Fa_mean = use_Fa_mean
        self.F_mean_module =  MultitaskMean(
            ZeroMean(), num_tasks=nq             # ZeroMean, ConstantMean
        )

    def mean_F(self, dataset):
        feat =  torch.cat((dataset.q_sn, dataset.dq_sn, dataset.tau_s), axis=-1)
        mu_F = self.F_mean_module(feat)
        if self.use_Fa_mean:
            mu_F += dataset.f + dataset.g + dataset.tau
        return mu_F
    
    def mean_ddqa(self, dataset):
        return torch.einsum('nab,nb->na', dataset.Minv, self.mean_F(dataset) )

    def __call__(self, quant:QUANT, dataset):
        fun_name = f'mean_{quant.name}'
        return self.__getattr__(fun_name)(dataset)

    ''' Unconstrained dynamics '''

    def mean_ddqas(self, dataset):
        return torch.einsum('nab,nb->na', dataset.ScFa, self.mean_ddqa(dataset) )
    
    def mean_dqan(self, dataset):
        # foward Euler
        return dataset.dq + self.dt * self.mean_ddqa(dataset)
    
    def mean_qan(self, dataset):
        # backward Euler
        return dataset.q + self.dt * self.mean_dqan(dataset)

    ''' Constrained dynamics '''

    def mean_ddq(self, dataset):
        # Udwadia-Kalaba equation
        return torch.einsum('nab,nb->na', dataset.L, dataset.b) + torch.einsum('nab,nb->na', dataset.T, self.mean_ddqa(dataset))

    def mean_ddqs(self, dataset):
        return torch.einsum('nab,nb->na', dataset.ScFa , self.mean_ddq(dataset))

    def mean_dqn(self, dataset):
        # foward Euler
        return dataset.dq + self.dt * self.mean_ddq(dataset)

    def mean_qn(self, dataset):
        # backward Euler
        return dataset.q + self.dt * self.mean_dqn(dataset)

    def mean_lamb(self, dataset):
        return torch.einsum('nab,nb->na', dataset.Ml, dataset.b) - torch.einsum('nab,nbc,nc->na', dataset.Ml, dataset.A, self.mean_ddqa(dataset))

'''--------------
Main Classes
--------------'''

class MultiTaskGP(GP, abc.ABC):
    ''' Abstract class for working with Multi-Task GPs.
        This class implementes all the main functionalities of GPs, like calculating log of the marginal
        likelihood and performing prediction.
        In addition, this class can normalize/standardize inputs and output enhance learning performance
    '''

    def __init__( 
        self, dataset_train:StructTorchArray, nq:int, inputs_quant:QUANT, 
        train_target_quant:QUANT, standardize:bool=True, standardize_out:bool=False
    ):
        super().__init__()
        self.GPpred = None
        self.standardize_in  = standardize
        self.standardize_out = standardize_out
        self.nq = nq
        self.dataset_train = dataset_train
        self.likelihood = MultiTaskLikelihood(num_tasks=nq)
        self.assertDataset(self.dataset_train)

        self.input_names        = inputs_quant
        self.train_target_quant = train_target_quant
        self.input_names        = [d.name for d in QUANT if bool(d & inputs_quant)]
        self.train_target_names = [d.name for d in QUANT if bool(d & train_target_quant)]
        
        # merge the target keys, add it to the dataset as 'target'
        self.dataset_train.targets = torch.cat( [self.dataset_train[k] for k in self.train_target_names], dim=1 )
        # add normalized inputs and target keys to the dataset. Store transforms
        self.addNormalizedDatasetKeys_train(self.dataset_train, self.input_names + ['targets'])

        # * USER HAS TO IMPLEMENT THIS
        self.mean_module  = None
        self.covar_module = None

    @abc.abstractmethod
    def train_mean_fun(self, dataset):
        pass # raise NotImplementedError()

    @abc.abstractmethod
    def train_train_covar_fun(self, dataset1, dataset2):
        pass # raise NotImplementedError()
    
    def addNormalizedDatasetKeys_train(self, dataset:StructTorchArray, keys:List[str]) -> List[NormStandDist]:
        ''' Includes in the dataset new keys with a prefix '_sn' and _s (e.g. 'tau' -> 'tau_sn') that indicates
        the (standardized s amd normalized n) dataset entries. 
        '''
        if not hasattr(self, 'transforms_StdNorm'):
            self.transforms_StdNorm = Dict()
        if not hasattr(self, 'transforms_Std'):
            self.transforms_Std = Dict()
        for key in keys:
            keyStdNorm = key + '_sn'
            keyStd = key + '_s'
            if self.standardize_in:
                dataset[keyStdNorm], self.transforms_StdNorm[key] = NormStandDist.transform(dataset[key], normData=True,  stdData=True)
                dataset[keyStd],     self.transforms_Std[key]     = NormStandDist.transform(dataset[key], normData=False, stdData=True)
            else:
                dataset[keyStdNorm] = dataset[key]
                dataset[keyStd]     = dataset[key]
        # return transforms_Std, transforms_StdNorm

    def addNormalizedDatasetKeys_test(self, dataset, keys:List[str]):
        'Standardize and normalize entries of the test dataset based on statistical measures from the train dataset'
        for key in keys:
            keyStdNorm = key + '_sn'
            keyStd = key + '_s'
            dataset[keyStdNorm] = self.transforms_StdNorm[key].apply_mean(dataset[key]) if self.standardize_in else dataset[key]
            dataset[keyStd]     = self.transforms_Std[key].apply_mean(dataset[key])     if self.standardize_in else dataset[key]

    def assertDataset(self, dataset):
        return True

    def train(self, mode=True):
        if mode: 
            self.GPpred = None
        super().train(mode)
    
    def __call__(self, dataset_test=None):
        ''' user should always call the function representing the quantity wanted to infer. However, if calling __call__,
        this class will evaluate the same quantity used for training
        '''
        return self._evaluate( self.train_mean_fun, self.train_train_covar_fun, self.train_train_covar_fun, dataset_test )

    def train_dist_norm(self):
        ''' returns the (normalized) train distribution
        '''
        train_dist = MultitaskMultivariateNormal(
            self.train_mean_fun(self.dataset_train), 
            self.train_train_covar_fun(self.dataset_train, self.dataset_train)
        )
        if self.standardize_out:
            train_dist = self.transforms_StdNorm.targets.apply_dist(train_dist)
        return train_dist

    def marginalLogLikelihood(self, verbose=False):
        ''' calculates the marginal log likelihood of the normalized observations
        '''
        if self.standardize_out: 
            return self.likelihood(self.train_dist_norm()).loglikelihood(self.dataset_train.targets_sn, verbose=verbose)
        else:
            return self.likelihood(self.train_dist_norm()).loglikelihood(self.dataset_train.targets, verbose=verbose)

    def _evaluate(self, test_mean_fun=None, train_test_covar_fun=None, test_test_covar_fun=None, dataset_test=None):     
        '''
        Args:
            test_mean_fun        = lambda dataset
            train_test_covar_fun = lambda dataset1, dataset2
            test_test_covar_fun  = lambda dataset1, dataset2
            dataset_test         = StructTorchArray
        '''
        # with torch.no_grad():

        # Prior mode
        if settings.prior_mode.on():
            self.addNormalizedDatasetKeys_test( dataset_test, self.input_names)           
            return MultitaskMultivariateNormal(
                test_mean_fun(dataset_test), 
                test_test_covar_fun(dataset_test, dataset_test)
            )

        # Posterior mode (returns the Gaussian predictive posterior distribution)
        else:
            self.assertDataset(dataset_test)
            # precalculate GP terms that only deppend on training data
            if self.GPpred is None:
                likelihood_norm = self.likelihood(self.train_dist_norm())
                if self.standardize_out:
                    likelihood = self.transforms_StdNorm.targets.reverse_dist(likelihood_norm)
                else:
                    likelihood = likelihood_norm
                self.GPpred = GPprediction( likelihood, self.dataset_train.targets )              
            # normalize test inputs based on statistics obtained from training data
            self.addNormalizedDatasetKeys_test( dataset_test, self.input_names)
            # evaluate test-deppendent terms
            test_mean      = test_mean_fun(dataset_test)
            train_test_cov = train_test_covar_fun(self.dataset_train, dataset_test)  
            test_test_cov  = test_test_covar_fun(dataset_test, dataset_test)                
            # infer test points
            prediction = self.GPpred.predict(test_mean, train_test_cov, test_test_cov)
            return prediction

class SGPModel(MultiTaskGP):
    ''' Class for GP-squared (Gauss' principle adhering GP) model
        Uses SGPMean and SGPKernel modules
    '''    
    def __init__( 
        self, dataset_train:StructTorchArray, nq:int, dt:float, train_target_quant:QUANT, 
        standardize:bool=True, standardize_out:bool=False, use_Fa_mean:bool=False, 
        multiRigidBody:MultiRigidBody=None, trainMBD=False
    ):
        super().__init__(
            dataset_train = dataset_train, 
            nq = nq, 
            inputs_quant = QUANT.q | QUANT.dq | QUANT.tau, 
            train_target_quant = train_target_quant, 
            standardize = standardize
        )
        # init layers
        self.mean_module = SGPMean(nq=nq, dt=dt, use_Fa_mean=use_Fa_mean)
        self.covar_module = SGPKernel_Fa_SE(nq=nq, dt=dt)
        # store parameters
        self.dt = dt
        self.multiRigidBody = multiRigidBody
        self.trainMBD = trainMBD

    def train_mean_fun(self, dataset):
        return self.mean_module( quant=self.train_target_quant, dataset=dataset )

    def train_train_covar_fun(self, dataset1, dataset2):
        return self.covar_module( quant1=self.train_target_quant, quant2=self.train_target_quant, dataset1=dataset1, dataset2=dataset2 )

    def assertDataset(self, dataset):
        return assertDataset(dataset)

    def marginalLogLikelihood(self, verbose=False):
        if self.trainMBD:
            # Update dataset by performing forward-propagation through the muli-body dynamical model (MBD)
            self.dataset_train = get_KUKA_SGPMatrices_from_MDB(self.multiRigidBody, self.dataset_train)            
            # Here, we need to update the target values because they might deppend on the MBD. This occurs for example, when we
            # scale the outputs based on some hyper-parameter deppendent values
            #    1. merge the target keys, add it to the dataset as 'target'
            self.dataset_train.targets = torch.cat( [self.dataset_train[k] for k in self.train_target_names], dim=1 )
            #    2. add normalized inputs and target keys to the dataset. Store transforms
            self.addNormalizedDatasetKeys_train(self.dataset_train, ['targets'])

        return super(SGPModel,self).marginalLogLikelihood(verbose)

    def __call__(self, quant:QUANT, dataset_test):
        # If not provided, we need to calculate SGP extra matrices as function of the 
        # inputs (q,dq,tau) using the MBD model
        if not {'A','b','M','L','T'}.issubset(dataset_test.keys()):
            dataset_test = get_KUKA_SGPMatrices_from_MDB(self.multiRigidBody, dataset_test)

        return super()._evaluate(
            test_mean_fun        = lambda dataset1:           self.mean_module(  quant,                          dataset1 ),
            train_test_covar_fun = lambda dataset1, dataset2: self.covar_module( self.train_target_quant, quant, dataset1, dataset2),
            test_test_covar_fun  = lambda dataset1, dataset2: self.covar_module( quant,                   quant, dataset1, dataset2),
            dataset_test         = dataset_test
        )

    def ddq(self, dataset_test=None):
        return self( QUANT.ddq, dataset_test )

    def dqn(self, dataset_test=None):
        return self( QUANT.dqn, dataset_test )

    def qn(self, dataset_test=None):
        return self( QUANT.qn, dataset_test )

class MultitaskGPModel(MultiTaskGP):
    ''' Multitask GP model with independent outputs
    '''
    def __init__(self, dataset_train:StructTorchArray, nq:int, dt, train_target_quant:QUANT=QUANT.ddq, standardize:bool=True, standardize_out:bool=False):

        super().__init__(
            dataset_train=dataset_train, 
            nq=nq, 
            inputs_quant=QUANT.q | QUANT.dq | QUANT.tau, 
            train_target_quant=train_target_quant, 
            standardize=standardize,
            standardize_out=standardize_out
        )
        self.dt = dt
        self.mean_module  = MultitaskMean( ZeroMean(), num_tasks=nq )

        # self.data_covar_module = RBFKernel(ard_num_dims=nq*3)
        # self.task_covar_module = IndexKernel( num_tasks=nq, rank=0 )
        self.covar_module = MTRBFKernel(num_tasks=nq, ard_num_dims=3*nq)

    def train_mean_fun(self, dataset):
        feat = torch.cat((dataset.q_sn, dataset.dq_sn, dataset.tau_sn), axis=-1)
        return self.mean_module(feat)

    @flatten
    def train_train_covar_fun(self, dataset1, dataset2):
        feat1 = torch.cat((dataset1.q_sn, dataset1.dq_sn, dataset1.tau_sn), axis=-1)
        feat2 = torch.cat((dataset2.q_sn, dataset2.dq_sn, dataset2.tau_sn), axis=-1)
        # data_covar = self.data_covar_module.forward(feat1, feat2)
        # task_covar = self.task_covar_module.covar_matrix.evaluate()
        # return kronProd(data_covar, task_covar)
        return self.covar_module(feat1,feat2)

    def assertDataset(self, dataset):
        return True
        # return assertDataset(dataset)

    def __call__(self, quant:QUANT, dataset_test):
        if bool(quant & QUANT.ddq) or bool(quant & QUANT.ddqa):
            return self.ddq(dataset_test)
        elif bool(quant & QUANT.dqn) or bool(quant & QUANT.dqan):
            return self.dqn(dataset_test)
        elif bool(quant & QUANT.qn) or bool(quant & QUANT.qan):
            return self.qn(dataset_test)
        else:
            raise Exception('This class can not evaluate this function')

    def ddq(self, dataset_test=None):
        return super()._evaluate(
            test_mean_fun        = self.train_mean_fun,
            train_test_covar_fun = self.train_train_covar_fun,
            test_test_covar_fun  = self.train_train_covar_fun,
            dataset_test         = dataset_test
        )

    def dqn(self, dataset):
        ddq_dist = self.ddq(dataset)
        return MultitaskMultivariateNormal(
            dataset.dq + ddq_dist.mean * self.dt, 
            ddq_dist.covariance_matrix * self.dt**2
        )
    
    def qn(self, dataset):
        dqn_dist = self.dqn(dataset)
        return MultitaskMultivariateNormal(
            dataset.q + dqn_dist.mean * self.dt, 
            dqn_dist.covariance_matrix * self.dt**2
        )

class StateSpaceGPModel(MultiTaskGP):
    ''' State Space GP model
    '''
    def __init__(self, dataset_train:StructTorchArray, nq:int, dt, train_target_quant=(QUANT.qn|QUANT.dqn), standardize:bool=True, standardize_out:bool=True):
        super().__init__(
            dataset_train=dataset_train, 
            nq= 2*nq, 
            inputs_quant=QUANT.q | QUANT.dq | QUANT.tau, 
            train_target_quant = train_target_quant,
            standardize=standardize,
            standardize_out=standardize_out
        )
        self.dt = dt
        self.mean_module  = MultitaskMean( ZeroMean(), num_tasks=2*nq )
        self.covar_module = MTRBFKernel(ard_num_dims=3*nq, num_tasks=2*nq)    

    def train_mean_fun(self, dataset):
        ''' Use current state as prior mean to the prediction of next state
        '''
        feat = torch.cat((dataset.q_sn, dataset.dq_sn, dataset.tau_s), axis=-1)
        return torch.cat((dataset.q, dataset.dq),axis=1) + self.mean_module(feat)

    @flatten
    def train_train_covar_fun(self, dataset1, dataset2):
        feat1 = torch.cat((dataset1.q_sn, dataset1.dq_sn, dataset1.tau_s), axis=-1)
        feat2 = torch.cat((dataset2.q_sn, dataset2.dq_sn, dataset2.tau_s), axis=-1)
        return self.covar_module(feat1,feat2)

    def assertDataset(self, dataset):
        return True
        # return assertDataset(dataset)

    def next_state(self, dataset_test=None):
        ''' returns state space prediction distribution of qn and dqn
        '''
        dist_qn_dqn = super()._evaluate(
            test_mean_fun        = self.train_mean_fun,
            train_test_covar_fun = self.train_train_covar_fun,
            test_test_covar_fun  = self.train_train_covar_fun,
            dataset_test         = dataset_test
        )
        # split multi-task mean
        mean_qn = dist_qn_dqn.mean[:,:dist_qn_dqn.num_tasks//2]
        mean_dqn = dist_qn_dqn.mean[:,dist_qn_dqn.num_tasks//2:]

        # split multi-task covariance matrix
        cov_ext = unflatten_MultiTaskCovarMatrix(dist_qn_dqn.covariance_matrix, d=dist_qn_dqn.num_tasks)
        cov_qn_qn   = flatten_MultiTaskCovarMatrix( cov_ext[:,:,:dist_qn_dqn.num_tasks//2,:dist_qn_dqn.num_tasks//2] )
        cov_dqn_dqn = flatten_MultiTaskCovarMatrix( cov_ext[:,:,dist_qn_dqn.num_tasks//2:,dist_qn_dqn.num_tasks//2:] )
        
        return (
            MultitaskMultivariateNormal(mean_qn,cov_qn_qn),
            MultitaskMultivariateNormal(mean_dqn,cov_dqn_dqn)
        )
    
    def qn(self, dataset_test=None):
        return self.next_state(dataset_test)[1]

    def dqn(self, dataset_test=None):
        return self.next_state(dataset_test)[0]
    
    def ddq(self, dataset_test=None):
        ''' Approximate acceleration using finite-differences
        '''
        dqn_dist = self.dqn(dataset_test)
        return MultitaskMultivariateNormal( 
            (dqn_dist.mean - dataset_test.dq) / self.dt,
            dqn_dist.covariance_matrix / self.dt**2
        )

    def __call__(self, quant:QUANT, dataset_test):
        if bool(quant & QUANT.ddq) or bool(quant & QUANT.ddqa):
            return self.ddq(dataset_test)
        elif bool(quant & QUANT.dqn) or bool(quant & QUANT.dqan):
            return self.dqn(dataset_test)
        elif bool(quant & QUANT.qn) or bool(quant & QUANT.qan):
            return self.qn(dataset_test)
        else:
            raise Exception('This class can not evaluate this function')

'''--------------
Help functions
--------------'''

def loadKUKA(
        urdfPath, basePos, baseOrn, 
        F_r_FE, gravity, 
        surf_fun, surf_fun_J, surf_fun_H, 
        endEffectorName='lbr_iiwa_link_7', baumgarte_wn=5., baumgarte_ksi=1.
    ) -> MultiRigidBody:
    ''' Create a MBD model for the KUKA experiment (end effector touching the surface)'''

    mbdKuka = MultiRigidBody.fromURDF(
        robotFileName   = urdfPath, 
        basePosition    = torch.tensor(basePos),
        baseOrientation = torch.from_numpy(R.from_quat(baseOrn).as_matrix()),
        I_grav			= torch.tensor(gravity) 
    )
    mbdEE = mbdKuka.linkMap[endEffectorName]
    surfaceConstraint = BodyOnSurfaceBilateralConstraint(
        predBody	  = mbdEE,
        P_r_PDp 	  = torch.tensor(F_r_FE) - mbdEE.getParentJoint().A_SDs.T @ (-mbdEE.getParentJoint().S_r_SDs()),
        surface_fun   = surf_fun,
        surface_fun_J = surf_fun_J,
        surface_fun_H = surf_fun_H,
        wn=baumgarte_wn, ksi=baumgarte_ksi 	# Baumgarte stabilization
    )
    mbdKuka.bilateralConstraints.append(surfaceConstraint)
    return mbdKuka

def get_KUKA_SGPMatrices_from_MDB(mbd:MultiRigidBody, dataset:StructTorchArray):
    ''' Given a KUKA model and configuration, get matrices needed for SGP and store into dataset
    '''
    # with torch.no_grad():

    # update model wit current position and velocity
    mbd.forwardKinematics(q=dataset.q , qDot=dataset.dq, qDDot=0*dataset.q)

    M, f, g = mbd.computationOfMfg()
    C = mbd.getJointFriction(dataset.dq)
    J_lambda, sigma_lambda = mbd.bilateralConstraints[0].getConstraintTerms()
    I_c = mbd.bilateralConstraints[0].I_c()
    A, b = J_lambda, -sigma_lambda
    I = torch.eye(mbd.nq, device=M.device)
    Ml = (A @ torch.solve(bT(A),M)[0]).inverse()
    L = torch.solve(bT(A),M)[0] @ Ml
    T = I - L @ A
    Minv = torch.solve(I,M)[0]
    ScFa = torch.cholesky(Minv)

    dataset.addColumns(
        M=M, f=f, g=g, C=C, A=A, b=b, Ml=Ml, L=L, T=T, Minv=Minv, ScFa=ScFa, I_c=I_c
    )
    if 'ddq' in dataset.keys():
        dataset.addColumns( ddqs = bmv(ScFa, dataset.ddq) )
    if 'ddqa' in dataset.keys():
        dataset.addColumns( ddqas = bmv(ScFa, dataset.ddqa) )
    
    return dataset
