'''
Created on 2020/07/09

@author: ukai
'''
from builtins import isinstance
import os
import unittest

from agent_factory import AgentFactory
from build_parameter import BuildParameter
from builder import Builder
from environment_factory import EnvironmentFactory
from mylogger import MyLogger
from store import Store
from trainer_factory import TrainerFactory
from loader import Loader
from build_parameter_factory import BuildParameterFactory
from agent import Agent


class Test(unittest.TestCase):
    
    
    @classmethod
    def setUpClass(cls):
        super(Test, cls).setUpClass()
        
        cls.dbPath = "testDb.sqlite"
        if os.path.exists(cls.dbPath):
            os.remove(cls.dbPath)
            
    @classmethod
    def tearDownClass(cls):
        super(Test, cls).tearDownClass()
        if os.path.exists(cls.dbPath):
            os.remove(cls.dbPath)
    
    def setUp(self):
        unittest.TestCase.setUp(self)
        
        agentFactory = AgentFactory()
        environmentFactory = EnvironmentFactory()
        trainerFactory = TrainerFactory()        
        buildParameterFactory = BuildParameterFactory()
        store = Store(self.dbPath)
        logger = MyLogger(console_print=False)
        
        self.builder = Builder(trainerFactory, agentFactory, environmentFactory, store, logger)
        
        self.buildParameters = []
        for k1 in range(3):
            nIntervalSave = 10
            nEpoch = 100
            self.buildParameters.append(BuildParameter(int(nIntervalSave), int(nEpoch), label="test" + str(k1)))
        
        self.loader = Loader(agentFactory, buildParameterFactory, environmentFactory, store)
        

    def test001(self):
        for buildParameter in self.buildParameters:
            assert isinstance(buildParameter, BuildParameter)
            self.builder.build(buildParameter)
            
        assert isinstance(self.loader, Loader)        
        for agent, buildParameter, epoch in self.loader.load("test%", None):
            assert isinstance(agent, Agent)
            assert isinstance(buildParameter, BuildParameter)
        
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test001']
    unittest.main()