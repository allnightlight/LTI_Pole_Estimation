'''
Created on 2020/07/16

@author: ukai
'''
import torch
import numpy as np
from sl_environment import SlEnvironment
from data_generator_singleton import DataGeneratorSingleton
from pole_batch_data_environment import PoleBatchDataEnvironment

class PoleEnvironment(SlEnvironment):
    '''
    classdocs
    '''


    def __init__(self, Nhidden, Ntrain, T0, T1, Ny, Nu, Nbatch, Nhrz, seed):
        '''
        Constructor
        '''
        
        super(PoleEnvironment, self).__init__()
        
        self.dataGeneratorSingleton = DataGeneratorSingleton.getInstance(
                                            Nhidden=Nhidden
                                           , Ntrain=Ntrain
                                           , T0=T0
                                           , T1=T1
                                           , Ny=Ny
                                           , Nu=Nu
                                           , seed=seed)
        
        self.Ntrain = Ntrain
        self.Nbatch = Nbatch
        self.Nhrz = Nhrz
        
        
    def generateBatchDataIterator(self):


        Nbatch = self.Nbatch 
        Nhrz = self.Nhrz 
        
        for _ in range((self.Ntrain-Nhrz)//Nbatch):
            idx = np.random.randint(low=0, high=self.Ntrain-Nhrz, size=(Nbatch,))
            idx = idx.reshape((1,-1)) + np.arange(Nhrz+1).reshape(-1,1) # (Nhrz+1, Nbatch)
            Ubatch = self.dataGeneratorSingleton.U[idx[:-1,:],:] # (Nhrz, *, Nu)
            Ybatch = self.dataGeneratorSingleton.Y[idx,:] # (Nhrz+1, *, Ny)
            y0 = Ybatch[0,...] # (*, Ny)
            
            _y0 = torch.tensor(y0)
            _U = torch.tensor(Ubatch)
            _Y = torch.tensor(Ybatch)
            
            batchDataEnvironment = PoleBatchDataEnvironment(_y0, _U, _Y)
            
            yield batchDataEnvironment
    
    def get_eig(self):
        
        A = self.dataGeneratorSingleton.A        
        eig, _ = np.linalg.eig(A)
        
        return eig
