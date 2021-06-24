'''
This file defines a data structure, that is similar to a panda dataframe. 
Each cell can however store also arrays.

'''
# from __future__ import annotations
import numpy as np
import torch
from typing import List, Union


class StructTorchArray():
    ''' This class defines a struct array datastructure
    Check __main__ to see an example 
    '''

    def __init__(self, **kwargs):
        self._dict = {}
        self.cat(**kwargs)

    def append(self, **kwargs):
        ''' Append one single data
        '''
        return self.cat(extradim=True, **kwargs)

    def cat(self, extradim:bool=False, **kwargs):
        ''' Append dataset to this object. All keys must be provided
        '''
        if len(kwargs):
            assert len(self)==0 or set(self.keys()) == set(kwargs.keys()), 'All keys must be provided'
            for key, v in kwargs.items():
                v = v if isinstance(v,torch.Tensor) else torch.from_numpy(np.array(v))
                if extradim:
                    v = v.unsqueeze(0)
                if v.dim()==1:
                    v = v.unsqueeze(-1)
                # v = v.float()
                assert not key in self._dict.keys() or self._dict[key].shape[1:] == v.shape[1:], f'Key {key} has shape {list(v.shape[1:])}, expected {list(self._dict[key].shape[1:])}'
                self._dict[key] = torch.cat((self._dict[key], v),dim=0) if key in self._dict.keys() else v
            assert len(set([v.shape[0] for v in self._dict.values()]))==1, 'Something went wrong, features have different lengths\n' + self.__str__()
            assert all([v.dim() > 1 for v in self._dict.values()] ), 'Something went wrong, not all features have at least two dimensions'
        return self

    @classmethod
    def merge(cls, structArrays:List['StructTorchArray']):
        merged = StructTorchArray()
        for k in structArrays[0].keys():
            merged._dict[k] = torch.cat( [v._dict[k] for v in structArrays] , dim=0)
        return merged

    def to(self, device, dtype=None):
        for k,v in self._dict.items():
            if not dtype is None:
                self._dict[k] = self._dict[k].type(dtype)
            self._dict[k] = self._dict[k].to(device)
        return self

    def keys(self):
        return self._dict.keys()
    
    def items(self):
        return self._dict.items()

    def addColumn(self, newKey:str, array:Union[np.ndarray,torch.Tensor]):
        assert len(self) == len(array), f'Length Missmatch: Array length is {len(array)} but current StructArray length is {len(self)}'
        self._dict[newKey] = array

    def addColumns(self, **kwargs):
        for k,v in kwargs.items():
            self.addColumn(newKey=k, array=v)

    def __setattr__(self, key, array:Union[np.ndarray,torch.Tensor]):
        if key == '_dict':
            self.__dict__[key] = array
        # user wants to add a new column
        else:
            if not isinstance(array,torch.Tensor):
                array = torch.tensor(array).float()
            self.addColumn(newKey=key, array=array)

    def __getattr__(self, key):
        if key.startswith('__') and key.endswith('__'): 
            raise AttributeError  # this solves a bug with pickle and dill
        return self._dict[key]
        # if len(self) == 1:
        #     return self._dict[key][0]
        # else:
        #     return self._dict[key]

    def __setitem__(self, key, array:Union[np.ndarray,torch.Tensor]):
        assert isinstance(key,str), f'Expected key: {key} to be a string'
        return self.__setattr__(key, array)

    def __getitem__(self, idx):
        if isinstance(idx,str):
            return self.__getattr__(idx)
        newdict = { 
            # slice StructArray and preserve the array size even when slicing one single element
            k:  self._dict[k][idx,:].reshape( -1, *self._dict[k].shape[1:] )
                # self._dict[k][None,idx]
                # if isinstance(idx,int) or (isinstance(idx,np.ndarray) and idx.ndim==0) # isinstance(idx,slice) or len(idx) > 1
                # else self._dict[k][idx,:] 
            for k,v in self.items()
        }
        newObj = StructTorchArray()         # TODO: change to self() ... check if it works
        newObj._dict = newdict
        return newObj
    
    def __len__(self):
        # empty struct array
        if not self._dict:
            return 0
        else:
            firstkey = list(self.keys())[0]
            return len(self._dict[firstkey])

    def __str__(self):
        s = f'StructArray with the following keys:\n'
        for k,v in self.items():
            s += f'{k}: {list(v.shape)}\n'
        s += f'and {len(self)} elements'
        return s
        # return f'Struct array with the following keys:\n{list(self.keys())}\nand {len(self)} elements'

    def __repr__(self):
        return self.__str__() # f'Struct array with the following keys:\n{list(self.keys())}\nand {len(self)} elements'


class StructNumpyArray(StructTorchArray):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __setattr__(self, key, array):
        if key == '_dict':
            self.__dict__[key] = array
        else: # user wants to add a new column
            super().addColumn(newKey=key, array=np.array(array)) 

    @classmethod
    def merge(cls, structArrays:List['StructNumpyArray']):
        merged = StructNumpyArray()
        for k in structArrays[0].keys():
            merged._dict[k] = np.concatenate( [v._dict[k] for v in structArrays], axis=0)
        return merged
    
    def cat(self, extradim:bool=False, **kwargs):
        ''' Concatenate dataset
        '''
        if len(kwargs):
            assert len(self)==0 or set(self.keys()) == set(kwargs.keys()), 'All keys must be provided'
            for key, v in kwargs.items():
                v = np.array(v)
                if extradim:
                    v = np.expand_dims(v, axis=0)
                if v.ndim==1:
                    v = np.expand_dims(v, axis=-1)
                assert not key in self._dict.keys() or self._dict[key].shape[1:] == v.shape[1:], f'Key {key} has shape {list(v.shape[1:])}, expected {list(self._dict[key].shape[1:])}'
                self._dict[key] = np.concatenate((self._dict[key], v), axis=0) if key in self._dict.keys() else v
            assert len(set([v.shape[0] for v in self._dict.values()]))==1, 'Something went wrong, features have different lengths'
            assert all( [v.ndim > 1 for v in self._dict.values()] ), 'Something went wrong, not all features have at least two dimensions'
        return self



if __name__ == "__main__":
    
    ''' ---------------- Test StructArray ------------------- '''
    
    M = torch.ones(2,3)
    N = [1,1]
    O = np.zeros((2,4,5))

    a = StructTorchArray(M=M, N=N, O=O)
    a.cat(M=M, N=N, O=O)
    a.cat(M=M, N=N, O=O)

    print(a)

    a.append(M=torch.zeros(3), N=[1], O=np.ones((4,5)))

    b = StructTorchArray(M=M, N=N, O=O)
    c = StructTorchArray.merge([a,b])

    d = StructTorchArray.merge([a,b,c])
    n = len(d)
    d.R = torch.ones(n,8,1)
    d.S = torch.ones(n,2,3)
    d.Q = np.ones((n,3))

    print(d)

    print(d[2:5])
    print(d[4])
    print(d[ torch.arange(len(d))%2==0 ])


    ''' ---------------- Test StructNumpyArray ------------------- '''

    a = StructNumpyArray(M=M, N=N, O=O)
    a.cat(M=M, N=N, O=O)
    a.cat(M=M, N=N, O=O)

    print(a)

    a.append(M=torch.zeros(3), N=[1], O=np.ones((4,5)))

    b = StructNumpyArray(M=M, N=N, O=O)
    c = StructNumpyArray.merge([a,b])

    d = StructNumpyArray.merge([a,b,c])
    n = len(d)
    d.R = torch.ones(n,8,1)
    d.S = torch.ones(n,2,3)
    d.Q = np.ones((n,3))

    print(f'M: {d.M.shape} N: {d.N.shape} O: {d.O.shape} R: {d.R.shape} ')

    print(d[2:5])
    print(d[4])
    print(d[ torch.arange(len(d))%2==0 ])