'''
Created on 2020/07/10

@author: ukai
'''
import unittest
from sl_trainer import SlTrainer
from sl_agent import SlAgent
from sl_environment import SlEnvironment


class Test(unittest.TestCase):


    def test001(self):
        
        agent = SlAgent()
        environment = SlEnvironment()
        
        trainer = SlTrainer(agent, environment)
        
        trainer.train()
        
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test001']
    unittest.main()