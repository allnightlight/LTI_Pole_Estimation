'''
Created on 2020/07/16

@author: ukai
'''

import torch
from torch.optim import Adam

from pole_batch_data_agent import PoleBatchDataAgent
from pole_batch_data_environment import PoleBatchDataEnvironment
from sl_trainer import SlTrainer
from builtins import isinstance
from pole_agent import PoleAgent
from pole_environment import PoleEnvironment


class PoleTrainer(SlTrainer):
    '''
    classdocs
    '''


    def __init__(self, agent, environment):
        SlTrainer.__init__(self, agent, environment)
        
        assert isinstance(agent, PoleAgent)
        assert isinstance(environment, PoleEnvironment)
        
        self.optimizer = Adam(agent.parameters())
        
        
    def update(self, batchDataIn, batchDataOut):
        
        assert isinstance(batchDataIn, PoleBatchDataEnvironment)
        assert isinstance(batchDataOut, PoleBatchDataAgent)
        
        _Y = batchDataIn._Y # (Nhrz+1, *, Ny)
        _Yhat = batchDataOut._Yhat # (Nhrz+1, * , Ny)
        
        _loss = torch.mean((_Y - _Yhat)**2)
        
        self.optimizer.zero_grad()
        _loss.backward()
        self.optimizer.step() 