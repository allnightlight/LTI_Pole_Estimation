'''
Created on 2020/07/10

@author: ukai
'''
from agent_factory import AgentFactory
from environment_factory import EnvironmentFactory
from store import Store


class Loader(object):
    '''
    classdocs
    '''


    def __init__(self, agentFactory, buildParameterFactory, environmentFactory, store):
        '''
        Constructor
        '''
        
        assert isinstance(agentFactory, AgentFactory)
        assert isinstance(environmentFactory, EnvironmentFactory)        
        assert isinstance(store, Store)
        
        self.agentFactory = agentFactory
        self.environmentFactory = environmentFactory
        self.buildParameterFactory = buildParameterFactory
        self.store = store
        
    def load(self, buildParameterLabel, epoch = None, buildParameterKey = None):
        
        for storeField in self.store.restore(buildParameterLabel, epoch, buildParameterKey):
            buildParameter = self.buildParameterFactory.create()            
            buildParameter.loadMemento(storeField.buildParameterMemento)
            
            environment = self.environmentFactory.create(buildParameter)
            agent = self.agentFactory.create(buildParameter, environment)
            agent.loadMemento(storeField.agentMemento)
            
            epoch = storeField.epoch
            
            yield agent, buildParameter, epoch