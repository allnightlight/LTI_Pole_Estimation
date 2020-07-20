'''
Created on 2020/07/11

@author: ukai
'''
import os
import unittest

from builder import Builder
from loader import Loader
from mylogger import MyLogger
from pole_agent import PoleAgent
from pole_agent_factory import PoleAgentFactory
from pole_build_parameter import PoleBuildParameter
from pole_build_parameter_factory import PoleBuildParameterFactory
from pole_environment_factory import PoleEnvironmentFactory
from pole_trainer_factory import PoleTrainerFactory
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
        
        agentFactory = PoleAgentFactory()
        environmentFactory = PoleEnvironmentFactory()
        trainerFactory = PoleTrainerFactory()        
        buildParameterFactory = PoleBuildParameterFactory()
        store = Store(self.dbPath)
        logger = MyLogger(console_print=True)
        
        self.builder = Builder(trainerFactory, agentFactory, environmentFactory, store, logger)
        
        self.buildParameters = []
        for k1 in range(2):
            nIntervalSave = 3
            nEpoch = 5
            self.buildParameters.append(PoleBuildParameter(int(nIntervalSave), int(nEpoch), label="test" + str(k1)))

        for agentClass in ("agent002", "agent003", "agent004"): 
            self.buildParameters.append(PoleBuildParameter(int(nIntervalSave), int(nEpoch), agentClass = agentClass, label="test " + agentClass))        
        
        self.loader = Loader(agentFactory, buildParameterFactory, environmentFactory, store)
        
    @classmethod
    def tearDownClass(cls):
        super(Test, cls).tearDownClass()
        cls.dbPath = "testDb.sqlite"
        if os.path.exists(cls.dbPath):
            os.remove(cls.dbPath)

    def test001(self):
        for buildParameter in self.buildParameters:
            assert isinstance(buildParameter, PoleBuildParameter)
            self.builder.build(buildParameter)
            
        assert isinstance(self.loader, Loader)        
        for agent, buildParameter, epoch in self.loader.load("test%", None):
            assert isinstance(agent, PoleAgent)  
            assert isinstance(buildParameter, PoleBuildParameter)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()