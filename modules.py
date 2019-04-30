
import sys
import numpy as np
from scipy.optimize import linprog
import torch 
import torch.nn as nn

class model001(nn.Module):
    def __init__(self, Ny, Nu, Nhidden):
        super(model001, self).__init__()
        self.Ny, self.Nu, self.Nhidden = Ny, Nu, Nhidden 
        self.y2x = nn.Linear(Ny, Nhidden)
        self.xu2x = nn.Linear(Nhidden+Nu, Nhidden)
        self.x2y = nn.Linear(Nhidden, Ny)

    def forward(self, _y0, _U):
        # _y0: (*, Ny), _U: (Nhrz, *, Nu)
        Nhrz = _U.shape[0]

        X = []
        _x = self.y2x(_y0) # (*, Nhidden)
        for k1 in range(Nhrz):
            _u = _U[k1,:] # (*, Nu)
            _x = self.xu2x(torch.cat((_x, _u), dim=1)) # (*, Nhidden)
            X.append(_x)
        _X = torch.stack(X, dim=0) # (Nhrz, *, Nhidden)
        _Y = self.x2y(_X) # (Nhrz, *, Ny)
        return _Y, dict()

class DataGenerator():
    def __init__(self, Nhidden, Ntrain = 2**12, T0 = 2**1, T1 = 2**7, 
        Ny = None, Nu = None):

        Ny = Nhidden if Ny is None else Ny
        Nu = Nhidden if Nu is None else Nu
        Nhalf = Nhidden//2

# pole/continuous time system = -alpha + 1j * beta * 2 * pi
        alpha = 1/np.exp(np.log(T0) + np.random.rand(Nhalf) * (np.log(T1) - np.log(T0)))
        beta  = 1/np.exp(np.log(T0) + np.random.rand(Nhalf) * (np.log(T1) - np.log(T0)))

        Diag = np.diag(np.concatenate([np.exp(-alpha + 1j * beta * np.pi), np.exp(-alpha - 1j * beta * np.pi)], axis=0))
        Vr = np.random.randn(Nhidden, Nhalf) 
        Vi = np.random.randn(Nhidden, Nhalf) 
        V = np.concatenate([Vr + 1j * Vi, Vr - 1j * Vi], axis=1)

        A = np.real(np.dot( np.dot(V, Diag), np.linalg.inv(V)))
        B = np.random.randn(Nhidden, Nu)
        C = np.random.randn(Ny, Nhidden)

        x = np.random.randn(Nhidden)
        X = [x,]
        U = np.random.randn(Ntrain, Nu)
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

    def next(self, Nbatch, Nhrz):
        idx = np.random.randint(low=0, high=self.Ntrain-Nhrz, size=(Nbatch,))
        idx = idx.reshape((1,-1)) + np.arange(Nhrz+1).reshape(-1,1) # (Nhrz+1, Nbatch)
        Ubatch = self.U[idx[:-1,:],:] # (Nhrz, *, Nu)
        Ybatch = self.Y[idx,:] # (Nhrz+1, *, Nu)
        return Ybatch, Ubatch

def run_training(mdl_constructor, data_generator, Nhidden, Nbatch, Nepoch, Nhrz, print_log = True):
    mdl = mdl_constructor(Nhidden)
    optimizer = torch.optim.Adam(mdl.parameters())
    Ntrain = data_generator.Ntrain
    Nitr = Ntrain//Nbatch

    criterion = nn.MSELoss()

    training_hist = []
    for epoch in range(Nepoch):
        for k1 in range(Nitr):

            Y, U = data_generator.next(Nbatch, Nhrz)
            _y0 = torch.tensor(Y[0,:]) # (*, Ny)
            _U  = torch.tensor(U) # (Nhrz, *, Nu)
            _Y = torch.tensor(Y[1:,:]) # (Nhrz, *, Ny)

            _Yhat, _ = mdl(_y0, _U) # (Nhrz, *, Ny)
            _loss = criterion(_Y, _Yhat)

            loss = float(_loss)
            training_hist.append((epoch + k1/Nitr, loss))  

            mdl.zero_grad()
            _loss.backward()
            optimizer.step()
        if print_log:
            sys.stdout.write('epoch %04d/%04d itr %03d loss %8.2e\r' % (epoch, Nepoch, k1, loss))
    return mdl, training_hist

def calc_wdist(c, p = None, q = None):
# c: (n,m)
    n, m = c.shape
    if p is None:
        p = np.ones((m,))/m
    if q is None:
        q = np.ones((n,))/n

    A_eq = np.zeros((m+n, m*n))
    for k1 in range(n):
        A_eq[m+k1, k1*m:(k1+1)*m] = 1
    for k1 in range(m):
        A_eq[k1, k1::m] = 1
    b_eq = np.concatenate((p,q), axis=0)

    bounds = ((0,1),) * (m*n)

    res = linprog(c.reshape(-1), A_eq=A_eq, b_eq=b_eq, bounds=bounds, 
        method = "interior-point", options = {'disp': False})

    wdist = res.fun
    P = res.x.reshape(n,m) # (n,m)
    return wdist, P

def check_spectrum(A, A_hat):
# check_spectrum given A and A_hat
    Nhidden = A.shape[0]
    assert A.shape == (Nhidden, Nhidden)
    assert A_hat.shape == (Nhidden, Nhidden)

    lmbd, _ = np.linalg.eig(A)
    lmbd_hat, _ = np.linalg.eig(A_hat)

    alpha = np.log(np.abs(lmbd)) + 1e-8
    beta = np.angle(lmbd) + 1e-8

    alpha_hat = np.log(np.abs(lmbd_hat))
    beta_hat = np.angle(lmbd_hat)

    c = 1/2 * np.abs(1-alpha_hat.reshape(-1,1)/alpha.reshape(1,-1)) \
        + 1/2 * np.abs(1-beta_hat.reshape(-1,1)/beta.reshape(1,-1))

    #c = 1/2 * np.abs(alpha_hat.reshape(-1,1)-alpha.reshape(1,-1)) \
    #    + 1/2 * np.abs(beta_hat.reshape(-1,1)-beta.reshape(1,-1))

    wdist, _ = calc_wdist(c)

    return wdist

def test001():

    for k1 in range(2**3):
        sys.stdout.write("%04d\r" % k1)

        Ny, Nu, Nhidden = np.random.randint(1, 10, size=(3,))
        mdl = model001(Ny, Nu, Nhidden)

        Nbatch = np.random.randint(1, 10)
        Nhrz = np.random.randint(1, 10)

        y0 =  np.random.randn(Nbatch, Ny).astype(np.float32)
        U =  np.random.randn(Nhrz, Nbatch, Nu).astype(np.float32)
        
        _y0 = torch.tensor(y0) # (*, Ny)
        _U = torch.tensor(U) # (Nhrz, *, Nu)

        _Yhat, _ = mdl(_y0, _U) # Nhrz, *, Ny

        Yhat = _Yhat.data.numpy()
        assert Yhat.shape == (Nhrz, Nbatch, Ny)
        assert np.all(~np.isnan(Yhat))

def test002():

    data_generator = DataGenerator(2**2, Ntrain = 2**7)
    Ny, Nu = data_generator.Ny, data_generator.Nu
    mdl_constructor = lambda Nhidden: model001(Ny, Nu, Nhidden)
    Nhidden, Nbatch, Nepoch, Nhrz =  np.random.randint(1, 100, size=(4,))

    run_training(mdl_constructor, data_generator, Nhidden, Nbatch, Nepoch, Nhrz)
    pass

def test003():
    m = 5
    n = 2
    c = np.exp(np.random.randn(n, m))

    _, P = calc_wdist(c)
    
    assert P.shape == (n, m)
    assert np.all(P >= 0)
    assert np.all(P <= 1)

    Nhidden = 2**3
    A = np.random.randn(Nhidden, Nhidden)
    A_hat = np.random.randn(Nhidden, Nhidden)
    wdist = check_spectrum(A, A_hat)

if __name__ == "__main__":
    test001()
    test002()
    test003()

        

