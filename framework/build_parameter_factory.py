'''
Created on 2020/07/10

@author: ukai
'''
from build_parameter import BuildParameter

class BuildParameterFactory(object):
    '''
    classdocs
    '''

    def create(self):
        return BuildParameter()