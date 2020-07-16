'''
Created on 2020/07/09

@author: ukai
'''
from agent import Agent
from environment import Environment


class Trainer(object):
    '''
    classdocs
    '''
    
    def __init__(self, agent, environment):
        assert isinstance(agent, Agent)
        assert isinstance(environment, Environment)
        self.agent = agent
        self.environment = environment
        
    # <<public>>
    def train(self):
        return
        