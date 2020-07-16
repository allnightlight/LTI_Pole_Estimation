'''
Created on 2020/07/16

@author: ukai
'''
from sl_environment_factory import SlEnvironmentFactory
from pole_environment import PoleEnvironment
from pole_build_parameter import PoleBuildParameter

class PoleEnvironmentFactory(SlEnvironmentFactory):
    '''
    classdocs
    '''


    def create(self, buildParameter):
        
        assert isinstance(buildParameter, PoleBuildParameter)
        
        environment = PoleEnvironment(Nhidden = buildParameter.NhiddenEnv
                        , Ntrain = buildParameter.Ntrain
                        , T0 = buildParameter.T0
                        , T1 = buildParameter.T1
                        , Ny = buildParameter.Ny
                        , Nu = buildParameter.Nu
                        , Nbatch = buildParameter.Nbatch
                        , Nhrz = buildParameter.Nhrz
                        , seed = buildParameter.seed)
        
        return environment