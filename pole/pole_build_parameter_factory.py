'''
Created on 2020/07/16

@author: ukai
'''
from sl_build_parameter_factory import SlBuildParameterFactory
from pole_build_parameter import PoleBuildParameter

class PoleBuildParameterFactory(SlBuildParameterFactory):
    '''
    classdocs
    '''


    def create(self):
        return PoleBuildParameter()    