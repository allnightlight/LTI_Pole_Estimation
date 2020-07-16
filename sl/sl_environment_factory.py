'''
Created on 2020/07/11

@author: ukai
'''
from environment_factory import EnvironmentFactory
from sl_build_parameter import SlBuildParameter
from sl_environment import SlEnvironment


class SlEnvironmentFactory(EnvironmentFactory):
    '''
    classdocs
    '''


    def create(self, buildParameter):
        assert isinstance(buildParameter, SlBuildParameter)

        return SlEnvironment()