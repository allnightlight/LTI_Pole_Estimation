'''
Created on 2020/07/11

@author: ukai
'''
from agent_factory import AgentFactory
from sl_agent import SlAgent
from sl_build_parameter import SlBuildParameter


class SlAgentFactory(AgentFactory):
    '''
    classdocs
    '''


    def create(self, buildParameter, environment):
        assert isinstance(buildParameter, SlBuildParameter)

        return SlAgent()