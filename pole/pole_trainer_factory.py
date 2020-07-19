'''
Created on 2020/07/16

@author: ukai
'''
from sl_trainer_factory import SlTrainerFactory
from pole_trainer import PoleTrainer

class PoleTrainerFactory(SlTrainerFactory):
    '''
    classdocs
    '''


    def create(self, buildParameter, agent, environment):
        trainer = PoleTrainer(agent, environment)
        return trainer