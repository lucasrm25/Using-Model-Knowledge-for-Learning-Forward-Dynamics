import torch
from torch.nn.parameter import Parameter
from typing import Callable, List, Union # , Dict
import abc


def param(tensor:torch.Tensor, requires_grad=False):
    return Parameter(tensor , requires_grad=requires_grad)


''' -------------- 
Bounding transformations 
-------------- '''

def sigmoid(x):
    return torch.sigmoid(x)

def inv_sigmoid(x):
    return torch.log(x) - torch.log(1 - x)

def softplus(x):
    return torch.nn.functional.softplus(x)

def inv_softplus(x):
    return x + torch.log(-torch.expm1(-x))

''' -------------- 
Constraints 
-------------- '''

class Constraint(abc.ABC, torch.nn.Module):
    def __init__(self, transform:Callable=None, inv_transform:Callable=None, lower_bound=0., upper_bound=1.):
        super().__init__()
        self._transform = transform
        self._inv_transform = inv_transform
        # we make bounds as Parameters instances so they are stored in state_dict 
        # (storing only the raw Parameter value is not enough, since the transformed value to be used in the algorithm deppends on the bound values)
        # but since we do never want to learn them, we set requires_grad=False
        self.lower_bound = Parameter(torch.tensor(lower_bound), requires_grad=False)  
        self.upper_bound = Parameter(torch.tensor(upper_bound), requires_grad=False)
    @abc.abstractmethod
    def transform(self, x): 
        pass
    @abc.abstractmethod
    def inv_transform(self, x): 
        pass
    def __str__(self):
        return f'{self.__class__.__name__} ({self.lower_bound.tolist()},{self.upper_bound.tolist()})'
    def __repr__(self):
        return self.__str__()

class DummyConstraint(Constraint):
    def __init__(self):
        super().__init__()
    def transform(self, x): return x
    def inv_transform(self, x): return x 

class Interval(Constraint):
    def __init__(self, lower_bound=0., upper_bound=1.):
        super().__init__(transform=sigmoid, inv_transform=inv_sigmoid, lower_bound=lower_bound, upper_bound=upper_bound)
    def transform(self, x):
        return (self._transform(x) * (self.upper_bound - self.lower_bound)) + self.lower_bound
    def inv_transform(self, x):
        return self._inv_transform((x - self.lower_bound) / (self.upper_bound - self.lower_bound))

class GreaterThan(Constraint):
    def __init__(self, lower_bound=0.):
        super().__init__(transfor=softplus, inv_transform=inv_softplus, lower_bound=lower_bound)
    def transform(self, x):
        return self._transform(x) + self.lower_bound
    def inv_transform(self, x):
        return self._inv_transform(x - self.lower_bound)

class LessThan(Constraint):
    def __init__(self, upper_bound=0.0):
        super().__init__(transform=softplus, inv_transform=inv_softplus, upper_bound=upper_bound)
    def transform(self, x):
        return -self._transform(-x) + self.upper_bound
    def inv_transform(self, x):
        return -self._inv_transform(-(x - self.upper_bound)) 

class ConstrainedParameter(torch.nn.Module):
    def __init__(self, tensor:torch.Tensor, constraint:Constraint=DummyConstraint(), requires_grad=False):
        super().__init__()
        self.constraint = constraint.to(tensor.device)
        self.raw = Parameter(self.constraint.inv_transform(tensor), requires_grad=requires_grad)  # inverse of sigmoid
    @property
    def device(self):
        return self.raw.device
    def __call__(self):
        return self.constraint.transform(self.raw)
    def __str__(self):
        return f'Constrained Parameter:\n\tTensor:     {self().data.__str__()}\n\tConstraint: {self.constraint.__str__()}'
    def __repr__(self):
        return self.__str__()

# # testing
# a = ConstrainedParameter(torch.tensor(1.), constraint=Interval(0.,5.))
# b = ConstrainedParameter(torch.tensor([1.,2.]), constraint=Interval(0.,5.))
# c = ConstrainedParameter(torch.tensor([1.,2.]), constraint=Interval([0.,1.],[2.,5.]))


''' -----
Tensor operations that are done with batched vectors <b,i> and batched matrices <b.i.j>
----- '''

def skew(vec:torch.Tensor):
    ''' 
    Generates a skew-symmetric matrix given a vector w
    '''
    S = torch.zeros([3,3], device=vec.device)

    S[0,1] = -vec[2]
    S[0,2] =  vec[1]
    S[1,2] = -vec[0]
    
    S[1,0] =  vec[2]
    S[2,0] = -vec[1]
    S[2,1] =  vec[0]
    return S

def batch(fun, vec:torch.Tensor):
    ''' Execute fun along the vec first dimension. 
    Returns the stacked batch of results
    '''
    # add extra dimension if batch dimension is missing
    bvec = vec.unsqueeze(0) if vec.dim()==1 else vec
    # stack batch results into the first dimension (the batch dimension)
    return torch.stack( [fun(v) for v in bvec] )

def rotZ(angle):
    return torch.tensor([
        [torch.cos(angle),-torch.sin(angle), 0],
        [torch.sin(angle), torch.cos(angle), 0],
        [0,                0,                1]
    ], device=angle.device)

def inputInfo(q:torch.Tensor):
    device, batchSize = q.device, q.shape[0]
    return device, batchSize

def bmm(m1:torch.Tensor, m2:torch.Tensor):
    ''' batch matrix-matrix multiplication
    '''
    return m1 @ m2

def bmv(m:torch.Tensor, v:torch.Tensor):
    ''' batch matrix-vector multiplication
    '''
    if m.dim()==3 and v.dim()==2:
        return torch.einsum('bij,bj->bi',m,v)
    elif m.dim()==2 and v.dim()==2:
        return torch.einsum('ij,bj->bi',m,v)
    elif m.dim()==3 and v.dim()==1:
        return torch.einsum('bij,j->bi',m,v)
    else:
        raise InvalidOperation(f'invalid matrix-vector multiplication: m.shape:{m.shape} and v.shape:{v.shape}')

def bvm(v:torch.Tensor, m:torch.Tensor):
    ''' batch matrix-vector multiplication
    '''
    if m.dim()==3 and v.dim()==2:
        return torch.einsum('bi,bij->bj',v,m)
    elif m.dim()==2 and v.dim()==2:
        return torch.einsum('bi,ij->bj',v,m)
    elif m.dim()==3 and v.dim()==1:
        return torch.einsum('i,bij->bj',v,m)
    else:
        raise InvalidOperation(f'invalid vector-matrix multiplication: m.shape:{m.shape} and v.shape:{v.shape}')

def binner(v1:torch.Tensor, v2:torch.Tensor):
    ''' batch inner product between vectors
    '''
    if v1.dim()==2 and v2.dim()==2:
        return torch.einsum('bi,bi->b',v1,v2).unsqueeze(-1)
    elif v1.dim()==2 and v2.dim()==1:
        return torch.einsum('bi,i->b',v1,v2).unsqueeze(-1)
    elif v1.dim()==1 and v2.dim()==2:
        return torch.einsum('i,bi->b',v1,v2).unsqueeze(-1)
    else:
        raise InvalidOperation(f'invalid inner product: v1.shape:{v1.shape} and v2.shape:{v2.shape}')

def bT(m:torch.Tensor):
    ''' batch transpose of matrix (transpose the last two dimensions)
    '''
    # assert m.dim() > 2, 'batch transpose requires input dimension >= 3 (1 batch dimension + 2 matrix dimensions)'
    return m.transpose(-1,-2)

def blstsq(A,b):
    ''' solves batch of least squares problem  x = pinv(A) @ b
    Args:
        A: <b,m,m>, b:<b,m>
    Returns:
        x: <b,m>
    '''
    return bmv( torch.pinverse(A) , b )

def bsolve(A,b):
    ''' solves batch of least squares problem  x = pinv(A) @ b
    Args:
        A: <b,m,m>, b:<b,m> or <b,m,k>
    Returns:
        x: <b,m> or <b,m,k>
    '''
    assert A.shape[0] == b.shape[0] and A.shape[1] == A.shape[2] == b.shape[1] and A.dim()==3 and b.dim() in [2,3], 'input shapes are wrong'
    if b.dim()==2:
        return torch.solve( b.unsqueeze(-1), A )[0].squeeze(-1)
    if b.dim()==3:
        return torch.solve( b, A )[0]      

def beye(batchSize, n, device):
    ''' batch eye matrix 
    '''
    return torch.eye(n,device=device).repeat(batchSize,1,1)

def beye_like(tensor):
    ''' tensor must have the size <batchSize, n, n>
    '''
    assert tensor.dim == 3 and tensor.shape[-1] == tensor.shape[-1]
    batchSize, n, device = tensor.shape[0], tensor.tensor.shape[-1], tensor.device
    return beye(batchSize, n, device)