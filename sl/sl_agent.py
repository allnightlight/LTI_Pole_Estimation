'''
Created on 2020/07/10

@author: ukai
'''

from agent import Agent
from batch_data_agent import BatchDataAgent
from batch_data_environment import BatchDataEnvironment


class SlAgent(Agent):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        super(SlAgent, self).__init__()

    # <<abstract>>
    def forward(self, batchDataIn):
        assert isinstance(batchDataIn, BatchDataEnvironment)
        batchDataOut = BatchDataAgent()
        return batchDataOut
    