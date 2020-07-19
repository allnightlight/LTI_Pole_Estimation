'''
Created on 2020/07/16

@author: ukai
'''

import torch
import torch.nn as nn
import numpy as np
from pole_batch_data_agent import PoleBatchDataAgent
from pole_batch_data_environment import PoleBatchDataEnvironment
from pole_agent import PoleAgent

class PoleAgent001(PoleAgent, nn.Module):
    '''
    classdocs
    '''

    def __init__(self, Ny, Nu, Nhidden):
        super(PoleAgent001, self).__init__()
        
        self.Ny, self.Nu, self.Nhidden = Ny, Nu, Nhidden 
        self.y2x = nn.Linear(Ny, Nhidden)
        self.xu2x = nn.Linear(Nhidden+Nu, Nhidden)
        self.x2y = nn.Linear(Nhidden, Ny)


    def forward(self, batchDataIn):        
        assert isinstance(batchDataIn, PoleBatchDataEnvironment)
        
        _y0 = batchDataIn._y0 # (*, Ny)
        _U = batchDataIn._U # (Nhrz, *, Nu)
        
        Nhrz = _U.shape[0]
 
        X = []
        _x = self.y2x(_y0) # (*, Nhidden)
        X.append(_x)
        for k1 in range(Nhrz):
            _u = _U[k1,:] # (*, Nu)
            _x = self.xu2x(torch.cat((_x, _u), dim=1)) # (*, Nhidden)
            X.append(_x)
        _X = torch.stack(X, dim=0) # (Nhrz+1, *, Nhidden)
        _Y = self.x2y(_X) # (Nhrz+1, *, Ny)
        
        batchDataOut = PoleBatchDataAgent(_Y)

        return batchDataOut

    def get_eig(self):
        _weight = self.xu2x.weight
        weight = _weight.data.numpy()
        A_hat = weight[:, :self.Nhidden]
        eig_hat, _ = np.linalg.eig(A_hat)
        return eig_hat
