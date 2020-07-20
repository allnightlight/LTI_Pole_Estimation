'''
Created on 2020/07/16

@author: ukai
'''
from pole_agent001 import PoleAgent001
from pole_agent002 import PoleAgent002
from pole_agent003 import PoleAgent003
from pole_agent004 import PoleAgent004
from pole_build_parameter import PoleBuildParameter
from sl_agent_factory import SlAgentFactory


class PoleAgentFactory(SlAgentFactory):
    '''
    classdocs
    '''


    def create(self, buildParameter, environment):
        
        assert isinstance(buildParameter, PoleBuildParameter)
        
        agent = None
        if buildParameter.agentClass == "agent001":
            agent = PoleAgent001(Ny = buildParameter.Ny, Nu = buildParameter.Nu, Nhidden = buildParameter.NhiddenAgent)
        if buildParameter.agentClass == "agent002":
            agent = PoleAgent002(Ny = buildParameter.Ny, Nu = buildParameter.Nu, Nhidden = buildParameter.NhiddenAgent, dampingConstantInitial = buildParameter.dampingConstantInitial)
        if buildParameter.agentClass == "agent003":
            agent = PoleAgent003(Ny = buildParameter.Ny, Nu = buildParameter.Nu, Nhidden = buildParameter.NhiddenAgent, dampingConstantInitial = buildParameter.dampingConstantInitial)
        if buildParameter.agentClass == "agent004":        
            agent = PoleAgent004(Ny = buildParameter.Ny, Nu = buildParameter.Nu, Nhidden = buildParameter.NhiddenAgent, dampingConstantInitial = buildParameter.dampingConstantInitial)
        assert agent is not None
        
        return agent