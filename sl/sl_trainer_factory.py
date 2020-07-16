'''
Created on 2020/07/11

@author: ukai
'''
from trainer_factory import TrainerFactory
from sl_trainer import SlTrainer
from builtins import isinstance
from sl_build_parameter import SlBuildParameter
from sl_agent import SlAgent
from sl_environment import SlEnvironment

class SlTrainerFactory(TrainerFactory):
    '''
    classdocs
    '''


    def create(self, buildParameter, agent, environment):
        assert isinstance(buildParameter, SlBuildParameter)
        assert isinstance(agent, SlAgent)
        assert isinstance(environment, SlEnvironment)
        return  SlTrainer(agent, environment)