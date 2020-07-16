'''
Created on 2020/07/11

@author: ukai
'''
import os
import unittest

from builder import Builder
from loader import Loader
from mylogger import MyLogger
from sl_agent import SlAgent
from sl_agent_factory import SlAgentFactory
from sl_build_parameter import SlBuildParameter
from sl_build_parameter_factory import SlBuildParameterFactory
from sl_environment_factory import SlEnvironmentFactory
from sl_trainer_factory import SlTrainerFactory
from store import Store


class Test(unittest.TestCase):
    
    
    @classmethod
    def setUpClass(cls):
        super(Test, cls).setUpClass()
        
        cls.dbPath = "testDb.sqlite"
        if os.path.exists(cls.dbPath):
            os.remove(cls.dbPath)
    
    def setUp(self):
        unittest.TestCase.setUp(self)
        
        agentFactory = SlAgentFactory()
        environmentFactory = SlEnvironmentFactory()
        trainerFactory = SlTrainerFactory()        
        buildParameterFactory = SlBuildParameterFactory()
        store = Store(self.dbPath)
        logger = MyLogger(console_print=False)
        
        self.builder = Builder(trainerFactory, agentFactory, environmentFactory, store, logger)
        
        self.buildParameters = []
        for k1 in range(3):
            nIntervalSave = 10
            nEpoch = 100
            self.buildParameters.append(SlBuildParameter(int(nIntervalSave), int(nEpoch), label="test" + str(k1)))
        
        self.loader = Loader(agentFactory, buildParameterFactory, environmentFactory, store)
        
    @classmethod
    def tearDownClass(cls):
        super(Test, cls).tearDownClass()
        cls.dbPath = "testDb.sqlite"
        if os.path.exists(cls.dbPath):
            os.remove(cls.dbPath)

    def test001(self):
        for buildParameter in self.buildParameters:
            assert isinstance(buildParameter, SlBuildParameter)
            self.builder.build(buildParameter)
            
        assert isinstance(self.loader, Loader)        
        for agent, buildParameter, epoch in self.loader.load("test%", None):
            assert isinstance(agent, SlAgent)
            assert isinstance(buildParameter, SlBuildParameter)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()