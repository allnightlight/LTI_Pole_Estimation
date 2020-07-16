'''
Created on 2020/07/16

@author: ukai
'''

import json

import numpy as np


class DataGeneratorSingleton(object):
    '''
    classdocs
    '''
    
    _instances = {}
    
    @classmethod
    def getInstance(cls, **params):
        key = json.dumps(params)
        
        if key in cls._instances:
            _inst = cls._instances[key]
        else:
            _inst = cls(**params)
            cls._instances[key] = _inst
        return _inst
        
    def __init__(self, Nhidden, Ntrain = 2**12, T0 = 2**1, T1 = 2**7, 
        Ny = None, Nu = None, seed = 0):
        
        print("""\
        
        ========================
        DataGeneratorSingleton's constructor was called with the following parameters:
        Nhidden: {0}
        Ntrain: {1}
        T0: {2}
        T1: {3}
        Ny: {4}
        Nu: {5}
        seed: {6}
        ========================        
        
        """.format(Nhidden, Ntrain, T0, T1, Ny, Nu, seed))
        
        rstate = np.random.RandomState(seed)
        
        Ny = Nhidden if Ny is None else Ny
        Nu = Nhidden if Nu is None else Nu
        Nhalf = Nhidden//2

# pole/continuous time system = -alpha + 1j * beta * 2 * pi
        alpha = 1/np.exp(np.log(T0) + rstate.rand(Nhalf) * (np.log(T1) - np.log(T0)))
        beta  = 1/np.exp(np.log(T0) + rstate.rand(Nhalf) * (np.log(T1) - np.log(T0)))

        Diag = np.diag(np.concatenate([np.exp(-alpha + 1j * beta * np.pi), np.exp(-alpha - 1j * beta * np.pi)], axis=0))
        Vr = rstate.randn(Nhidden, Nhalf) 
        Vi = rstate.randn(Nhidden, Nhalf) 
        V = np.concatenate([Vr + 1j * Vi, Vr - 1j * Vi], axis=1)

        A = np.real(np.dot( np.dot(V, Diag), np.linalg.inv(V)))
        multiplier = np.concatenate((np.sqrt(1-np.exp(-2*alpha)), np.sqrt(1-np.exp(-2*alpha))))
        B = multiplier.reshape((-1,1)) *  rstate.randn(Nhidden, Nu)
        C = rstate.randn(Ny, Nhidden)

        x = rstate.randn(Nhidden)
        X = [x,]
        U = rstate.randn(Ntrain, Nu)
        for k1 in range(Ntrain):
            u = U[k1,:]
            x = np.dot(A, x) + np.dot(B, u)
            X.append(x)
        X = np.stack(X, axis=0)
        Y = np.dot(X, C.T)
        Y = (Y - np.mean(Y, axis=0))/np.std(Y, axis=0)

        self.A = A
        self.B = B
        self.C = C
        self.U = U.astype(np.float32)
        self.Y = Y.astype(np.float32)
        self.Ntrain = Ntrain
        self.Nhidden = Nhidden
        self.Nu = Nu
        self.Ny = Ny
