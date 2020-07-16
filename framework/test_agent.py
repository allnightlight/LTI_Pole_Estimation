'''
Created on 2020/07/09

@author: ukai
'''
import unittest

from agent import Agent
from builtins import isinstance


class Test(unittest.TestCase):
    
    def setUp(self):
        unittest.TestCase.setUp(self)
        
        agent = Agent()
        
        self.agents = [agent,]


    def test001(self):
        
        for agent in self.agents:
            
            assert isinstance(agent, Agent)
            agentMemento = agent.createMemento()
            agent.loadMemento(agentMemento)
        
        
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test001']
    unittest.main()