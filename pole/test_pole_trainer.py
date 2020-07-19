'''
Created on 2020/07/16

@author: ukai
'''
import unittest

from pole_agent001 import PoleAgent001
from pole_agent004 import PoleAgent004
from pole_environment import PoleEnvironment
from pole_trainer import PoleTrainer
from builtins import isinstance


class Test(unittest.TestCase):


    def test001(self):
        
        Ny = 2
        Nu = 3
        
        environment = PoleEnvironment(Nhidden=2**2, Ntrain=2**10, T0=2**3, T1=2**4, Ny=Ny, Nu=Nu, Nbatch=2**5, Nhrz=2**3, seed = 1)
        
        params = dict(Ny=Ny, Nu=Nu, Nhidden=2**3)
        
        for agent in (PoleAgent001(**params)
                      , PoleAgent004(**params)):
            
            trainer = PoleTrainer(agent, environment)
            assert isinstance(trainer, PoleTrainer)
            
            trainer.train()
        
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test001']
    unittest.main()