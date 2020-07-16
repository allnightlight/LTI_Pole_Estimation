'''
Created on 2020/07/16

@author: ukai
'''
from batch_data_environment import BatchDataEnvironment

class PoleBatchDataEnvironment(BatchDataEnvironment):
    '''
    classdocs
    '''


    def __init__(self, _y0, _U, _Y):
        # _y0: (*, Ny), _U: (Nhrz, *, Nu), _Y: (Nhrz+1, *, Ny)
        
        assert len(_y0.shape) == 2
        assert len(_U.shape) == 3
        assert len(_Y.shape) == 3
        assert _U.shape[0] == _Y.shape[0]-1
        assert _y0.shape == _Y.shape[1:]

        self._y0 = _y0
        self._U = _U
        self._Y = _Y