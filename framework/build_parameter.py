'''
Created on 2020/07/09

@author: ukai
'''
import json

from util import Utils


class BuildParameter(object):
    '''
    classdocs
    '''

    def __init__(self, nIntervalSave=2**4, nEpoch = 2**8, label = "None"):
        '''
        Constructor
        '''
        
        self.key = Utils.generateRandomString(16)
        self.nIntervalSave = nIntervalSave
        self.nEpoch = nEpoch
        self.label = label
        
    def createMemento(self):
        return json.dumps(self.__dict__)
    
    def loadMemento(self, buildParameterMemento):
        d = json.loads(buildParameterMemento)
        for key in self.__dict__:
            self.__dict__[key] = d[key]