'''
Created on 2020/07/10

@author: ukai
'''
from builtins import isinstance

from batch_data_agent import BatchDataAgent
from batch_data_environment import BatchDataEnvironment
from sl_agent import SlAgent
from sl_environment import SlEnvironment
from trainer import Trainer


class SlTrainer(Trainer):
    '''
    classdocs
    '''

    def __init__(self, agent, environment):
        '''
        Constructor
        '''
        super(SlTrainer, self).__init__(agent, environment)
        
        assert isinstance(environment, SlEnvironment)
        assert isinstance(agent, SlAgent)
         
    def train(self):
        
        for batchDataIn in self.environment.generateBatchDataIterator():            
            batchDataOut = self.agent.forward(batchDataIn)
            self.update(batchDataIn, batchDataOut)
        
    # <<abstract>>    
    def update(self, batchDataIn, batchDataOut):
        assert isinstance(batchDataIn, BatchDataEnvironment)
        assert isinstance(batchDataOut, BatchDataAgent)
        return