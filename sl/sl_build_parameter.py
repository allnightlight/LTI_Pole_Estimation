'''
Created on 2020/07/11

@author: ukai
'''
from build_parameter import BuildParameter

class SlBuildParameter(BuildParameter):
    '''
    classdocs
    '''

    def __init__(self, nIntervalSave = 2**4, nEpoch = 2**7, label = "None"):
        super(SlBuildParameter, self).__init__(nIntervalSave, nEpoch, label)
