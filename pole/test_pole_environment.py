'''
Created on 2020/07/16

@author: ukai
'''
import unittest

from data_generator_singleton import DataGeneratorSingleton
import numpy as np
from pole_batch_data_environment import PoleBatchDataEnvironment
from pole_environment import PoleEnvironment


class Test(unittest.TestCase):


    def test001(self):
        
        for _ in range(10):
            dg = DataGeneratorSingleton.getInstance(Nhidden=2**2, Ntrain=2**10, T0=2**3, T1=2**4, Ny=1, Nu=1, seed = 0)
            assert np.abs(dg.Y[0,0] - 0.076161645) < 1e-8
            
            dg = DataGeneratorSingleton.getInstance(Nhidden=2**2, Ntrain=2**10, T0=2**3, T1=2**4, Ny=1, Nu=1, seed = 1)
            assert np.abs(dg.Y[0,0] - 0.076161645) > 1e-8
            
        
    def test002(self):
        
        environment = PoleEnvironment(Nhidden=2**2, Ntrain=2**10, T0=2**3, T1=2**4, Ny=1, Nu=1, Nbatch=2**5, Nhrz=2**3, seed = 1)
        for batchDataEnvironment in environment.generateBatchDataIterator():
            assert isinstance(batchDataEnvironment, PoleBatchDataEnvironment)
        

    def test003(self):
        
        for _ in range(10):
            Nhidden = int(np.random.randint(5)) * 2 
            
            environment = PoleEnvironment(Nhidden=Nhidden, Ntrain=2**10, T0=2**3, T1=2**4, Ny=1, Nu=1, Nbatch=2**5, Nhrz=2**3, seed = 1)
            
            eig = environment.get_eig()
            
            assert eig.shape == (Nhidden,)
            assert np.all(np.abs(eig) <= 1. + 1e-8)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test001']
    unittest.main()