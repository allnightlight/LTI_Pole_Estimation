'''
Created on 2020/07/20

@author: ukai
'''

from pole_agent import PoleAgent
from pole_batch_data_agent import PoleBatchDataAgent
from pole_batch_data_environment import PoleBatchDataEnvironment
import numpy as np
import torch
import torch.nn as nn


class PoleAgent003(PoleAgent, nn.Module):
    '''
    classdocs
    '''


    def __init__(self, Ny, Nu, Nhidden, dampingConstantInitial):
        super(PoleAgent003, self).__init__()
        
        self.Ny, self.Nu, self.Nhidden = Ny, Nu, Nhidden 

        assert dampingConstantInitial < 1.
        lmbd_cont_real = (-np.log(dampingConstantInitial)) * np.ones(Nhidden//2) # (*, Nhidden//2)
        lmbd_cont_imag = np.random.rand(Nhidden//2) # (*, Nhidden//2)
        B = np.random.randn(Nhidden, Nu)/np.sqrt(Nu)

        self._lmbd_cont_real = nn.Parameter(torch.from_numpy(lmbd_cont_real.astype(np.float32)))
        self._lmbd_cont_imag = nn.Parameter(torch.from_numpy(lmbd_cont_imag.astype(np.float32)))
        self._B = nn.Parameter(torch.from_numpy(B.astype(np.float32)))

        self.y2x = nn.Linear(Ny, Nhidden)
        self.x2y = nn.Linear(Nhidden, Ny)


    def forward(self, batchDataIn):                
        assert isinstance(batchDataIn, PoleBatchDataEnvironment)
        
        _y0 = batchDataIn._y0 # (*, Ny)
        _U = batchDataIn._U # (Nhrz, *, Nu)
        
        Nhrz = _U.shape[0]

        X = []

        _r = torch.exp(-torch.abs(self._lmbd_cont_real)) # (Nx//2,)
        _theta = np.pi/2 * self._lmbd_cont_imag # (Nx//2,)
        _lmbd_real = _r * torch.cos(_theta) # (Nx//2,)
        _lmbd_imag = _r * torch.sin(_theta) # (Nx//2,)

        _normalized_factor_tmp = torch.sqrt(1 - torch.exp(-2*torch.abs(self._lmbd_cont_real))) # (Nx//2,)
        _normalized_factor = torch.cat((_normalized_factor_tmp, _normalized_factor_tmp)) # (Nx,)
        _Bu = torch.matmul(_U, self._B.t())  * _normalized_factor # (Nhrz, *, Nx)

        _A11 = torch.diag(_lmbd_real) # = A22, (Nx//2, Nx//2)
        _A21 = torch.diag(_lmbd_imag) # = A21, (Nx//2, Nx//2)
        _A = torch.cat((torch.cat((_A11, -_A21), dim=1), 
            torch.cat((_A21, _A11), dim=1)), dim=0) #(Nx, Nx)

        _x = self.y2x(_y0) # (*, Nhidden)
        X.append(_x)
        for k1 in range(Nhrz):
            _x = torch.matmul(_x, _A.t()) + _Bu[k1,:]
            X.append(_x)
        _X = torch.stack(X, dim=0) # (Nhrz+1, *, Nhidden)
        _Y = self.x2y(_X) # (Nhrz, *, Ny)

        batchDataOut = PoleBatchDataAgent(_Y)

        return batchDataOut

    def get_eig(self):
        lmbd_cont_real = self._lmbd_cont_real.data.numpy()
        lmbd_cont_imag = self._lmbd_cont_imag.data.numpy()
        
        r = np.exp(-np.abs(lmbd_cont_real))
        theta = np.pi/2 * lmbd_cont_imag
        
        lmbd_real = r * np.cos(theta)
        lmbd_imag = r * np.sin(theta)
        
        eig_hat = np.concatenate((lmbd_real + 1j * lmbd_imag, lmbd_real - 1j * lmbd_imag))
        return eig_hat
