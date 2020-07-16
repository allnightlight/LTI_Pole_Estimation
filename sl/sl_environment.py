'''
Created on 2020/07/10

@author: ukai
'''

from batch_data_environment import BatchDataEnvironment
from environment import Environment


class SlEnvironment(Environment):
    '''
    classdocs
    '''

    def __init__(self):
        super(SlEnvironment, self).__init__()

    # <<abstract>>
    def generateBatchDataIterator(self):
        
        yield BatchDataEnvironment()