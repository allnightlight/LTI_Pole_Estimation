'''
Created on 2020/07/09

@author: ukai
'''
import unittest

from trainer import Trainer
from agent import Agent
from environment import Environment


class Test(unittest.TestCase):


    def setUp(self):
        unittest.TestCase.setUp(self)
        
        agent = Agent()
        environment = Environment()
        
        self.trainers = []
        self.trainers.append(Trainer(agent, environment))

    def test001(self):
        
        for trainer in self.trainers:
            assert isinstance(trainer, Trainer)
            
            trainer.train()
            

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test001']
    unittest.main()