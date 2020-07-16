'''
Created on 2020/07/16

@author: ukai
'''
from batch_data_agent import BatchDataAgent



class PoleBatchDataAgent(BatchDataAgent):
    '''
    classdocs
    '''


    def __init__(self, _Yhat):
        # _Y: (Nhrz, *, Ny)
        assert len(_Yhat.shape) == 3
        
        self._Yhat = _Yhat