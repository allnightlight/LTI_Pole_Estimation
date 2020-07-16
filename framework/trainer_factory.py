'''
Created on 2020/07/09

@author: ukai
'''
from build_parameter import BuildParameter
from trainer import Trainer

class TrainerFactory(object):
    '''
    classdocs
    '''


    def create(self, buildParameter, agent, environment):
        isinstance(buildParameter, BuildParameter)
        
        trainer = Trainer(agent, environment)
        
        return trainer