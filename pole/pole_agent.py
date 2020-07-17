'''
Created on 2020/07/17

@author: ukai
'''
from sl_agent import SlAgent

# <<abstract>>
class PoleAgent(SlAgent):
    '''
    classdocs
    '''


    def get_eig(self):
        raise NotImplementedError()