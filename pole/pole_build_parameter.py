'''
Created on 2020/07/16

@author: ukai
'''
from sl_build_parameter import SlBuildParameter

class PoleBuildParameter(SlBuildParameter):
    '''
    classdocs
    '''


    def __init__(self, nIntervalSave=2 ** 4, nEpoch=2 ** 7, label="None", Ny = 1, Nu = 1, NhiddenAgent=2**2, agentClass = "agent001", Nbatch = 2**5, Nhrz=1, seed=0, NhiddenEnv = 2**2, Ntrain = 2**12, T0 = 2**1, T1=2**7, dampingConstantInitial=0.99):
        SlBuildParameter.__init__(self, nIntervalSave=nIntervalSave, nEpoch=nEpoch, label=label)
        
        self.Ny = Ny
        self.Nu = Nu
        self.NhiddenAgent = NhiddenAgent
        self.agentClass = agentClass
        self.Nbatch = Nbatch
        self.Nhrz = Nhrz
        self.seed = seed
        self.NhiddenEnv = NhiddenEnv
        self.Ntrain = Ntrain
        self.T0 = T0
        self.T1 = T1
        self.dampingConstantInitial = dampingConstantInitial