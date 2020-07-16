'''
Created on 2020/07/09

@author: ukai
'''
from util import Utils


class Agent(object):
    '''
    classdocs
    '''

    
    def createMemento(self):
        agentMemento = Utils.generateRandomString(16)        
        return agentMemento
    
    def loadMemento(self, agentMemento):
        pass
        