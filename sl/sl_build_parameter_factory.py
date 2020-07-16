'''
Created on 2020/07/11

@author: ukai
'''
from build_parameter_factory import BuildParameterFactory
from sl_build_parameter import SlBuildParameter

class SlBuildParameterFactory(BuildParameterFactory):
    '''
    classdocs
    '''

    def __init__(self):
        super(SlBuildParameterFactory, self).__init__()
        

    def create(self):
        return SlBuildParameter()