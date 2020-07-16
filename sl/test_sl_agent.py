'''
Created on 2020/07/10

@author: ukai
'''
from builtins import isinstance
import unittest

from batch_data_agent import BatchDataAgent
from batch_data_environment import BatchDataEnvironment
from sl_agent import SlAgent


class Test(unittest.TestCase):


    def test001(self):
        
        agent = SlAgent()
        
        assert isinstance(agent, SlAgent)
        
        batchDataIn = BatchDataEnvironment()
        batchDataOut = agent.forward(batchDataIn)
        
        assert isinstance(batchDataOut, BatchDataAgent)
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test001']
    unittest.main()