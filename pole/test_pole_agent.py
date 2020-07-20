'''
Created on 2020/07/16

@author: ukai
'''
import unittest

import numpy as np
from pole_agent import PoleAgent
from pole_agent001 import PoleAgent001
from pole_agent002 import PoleAgent002
from pole_agent003 import PoleAgent003
from pole_agent004 import PoleAgent004
from pole_batch_data_agent import PoleBatchDataAgent
from pole_environment import PoleEnvironment


class Test(unittest.TestCase):


    def test001(self):
        
        Ny = 2
        Nu = 3
        
        environment = PoleEnvironment(Nhidden=2**2, Ntrain=2**10, T0=2**3, T1=2**4, Ny=Ny, Nu=Nu, Nbatch=2**5, Nhrz=2**3, seed = 1)
        
        params = dict(Ny=Ny, Nu=Nu, Nhidden=2**3)
        
        for agent in (PoleAgent001(**params)
                      , PoleAgent002(**params, dampingConstantInitial=0.99)
                      , PoleAgent003(**params, dampingConstantInitial=0.99)
                      , PoleAgent004(**params, dampingConstantInitial=0.99)):
            
            for batchDataIn in environment.generateBatchDataIterator():
                batchDataOut = agent(batchDataIn)
                
                assert isinstance(batchDataOut, PoleBatchDataAgent)
        

    def test002(self):
        
        Ny = 2
        Nu = 3
        
        params = dict(Ny=Ny, Nu=Nu, Nhidden=2**3)
        
        agent = PoleAgent001(**params)
        agentAnother = PoleAgent001(**params)
        assert isinstance(agent, PoleAgent)
        
        agentMemento = agent.createMemento()
        
        agentAnother.loadMemento(agentMemento)
        
        for _p1, _p2 in zip(agent.parameters(), agentAnother.parameters()):
            p1 = _p1.data.numpy()
            p2 = _p2.data.numpy()
            assert np.all(p1 == p2)        

        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test001']
    unittest.main()