
import sys
import numpy as np
from scipy.optimize import linprog
import torch 
import torch.nn as nn
import itertools
import time

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

    def get_eig(self):
        _weight = self.xu2x.weight
        weight = _weight.data.numpy()
        A_hat = weight[:, :self.Nhidden]
        eig_hat, _ = np.linalg.eig(A_hat)
        return eig_hat


class model002(nn.Module):
    def __init__(self, Ny, Nu, Nhidden):
        super(model002, self).__init__()
        self.Ny, self.Nu, self.Nhidden = Ny, Nu, Nhidden 

        r = np.random.rand(Nhidden//2)
        theta = np.random.rand(Nhidden//2) * np.pi/2
        #theta = np.pi/4

        lmbd_real = r * np.cos(theta) # (*, Nhidden//2)
        lmbd_imag = r * np.sin(theta) # (*, Nhidden//2)
        B = np.random.randn(Nhidden, Nu)/np.sqrt(Nu)

        self._lmbd_real = nn.Parameter(torch.from_numpy(lmbd_real.astype(np.float32)))
        self._lmbd_imag = nn.Parameter(torch.from_numpy(lmbd_imag.astype(np.float32)))
        self._B = nn.Parameter(torch.from_numpy(B.astype(np.float32)))

        self.y2x = nn.Linear(Ny, Nhidden)
        self.x2y = nn.Linear(Nhidden, Ny)

    def forward(self, _y0, _U):
        # _y0: (*, Ny), _U: (Nhrz, *, Nu)
        Nhrz = _U.shape[0]

        X = []
        _Bu = torch.matmul(_U, self._B.t()) # (Nhrz, *, Nx)

        _A11 = torch.diag(self._lmbd_real) # = A22, (Nx//2, Nx//2)
        _A21 = torch.diag(self._lmbd_imag) # = A21, (Nx//2, Nx//2)
        _A = torch.cat((torch.cat((_A11, -_A21), dim=1), 
            torch.cat((_A21, _A11), dim=1)), dim=0) #(Nx, Nx)

        _x = self.y2x(_y0) # (*, Nhidden)
        for k1 in range(Nhrz):
            _x = torch.matmul(_x, _A.t()) + _Bu[k1,:]
            X.append(_x)
        _X = torch.stack(X, dim=0) # (Nhrz, *, Nx)
        _Y = self.x2y(_X) # (Nhrz, *, Ny)
        return _Y, dict()

    def get_eig(self):
        lmbd_real = self._lmbd_real.data.numpy()
        lmbd_imag = self._lmbd_imag.data.numpy()
        eig_hat = np.concatenate((lmbd_real + 1j * lmbd_imag, lmbd_real - 1j * lmbd_imag))
        return eig_hat

class model003(nn.Module):
    def __init__(self, Ny, Nu, Nhidden):
        super(model003, self).__init__()
        self.Ny, self.Nu, self.Nhidden = Ny, Nu, Nhidden 

        lmbd_cont_real = np.random.rand(Nhidden//2) # (*, Nhidden//2)
        lmbd_cont_imag = np.random.rand(Nhidden//2) # (*, Nhidden//2)
        B = np.random.randn(Nhidden, Nu)/np.sqrt(Nu)

        self._lmbd_cont_real = nn.Parameter(torch.from_numpy(lmbd_cont_real.astype(np.float32)))
        self._lmbd_cont_imag = nn.Parameter(torch.from_numpy(lmbd_cont_imag.astype(np.float32)))
        self._B = nn.Parameter(torch.from_numpy(B.astype(np.float32)))

        self.y2x = nn.Linear(Ny, Nhidden)
        self.x2y = nn.Linear(Nhidden, Ny)

    def forward(self, _y0, _U):
        # _y0: (*, Ny), _U: (Nhrz, *, Nu)
        Nhrz = _U.shape[0]

        X = []

        _r = torch.exp(-torch.abs(self._lmbd_cont_real)) # (Nx//2,)
        _theta = np.pi/2 * self._lmbd_cont_imag # (Nx//2,)
        _lmbd_real = _r * torch.cos(_theta) # (Nx//2,)
        _lmbd_imag = _r * torch.sin(_theta) # (Nx//2,)

        _normalized_factor_tmp = torch.sqrt(1 - torch.exp(-2*torch.abs(self._lmbd_cont_real))) # (Nx//2,)
        _normalized_factor = torch.cat((_normalized_factor_tmp, _normalized_factor_tmp)) # (Nx,)
        _Bu = torch.matmul(_U, self._B.t())  * _normalized_factor # (Nhrz, *, Nx)

        _A11 = torch.diag(_lmbd_real) # = A22, (Nx//2, Nx//2)
        _A21 = torch.diag(_lmbd_imag) # = A21, (Nx//2, Nx//2)
        _A = torch.cat((torch.cat((_A11, -_A21), dim=1), 
            torch.cat((_A21, _A11), dim=1)), dim=0) #(Nx, Nx)

        _x = self.y2x(_y0) # (*, Nhidden)
        for k1 in range(Nhrz):
            _x = torch.matmul(_x, _A.t()) + _Bu[k1,:]
            X.append(_x)
        _X = torch.stack(X, dim=0) # (Nhrz, *, Nx)
        _Y = self.x2y(_X) # (Nhrz, *, Ny)
        return _Y, dict()

    def get_eig(self):
        lmbd_cont_real = self._lmbd_cont_real.data.numpy()
        lmbd_cont_imag = self._lmbd_cont_imag.data.numpy()
        
        r = np.exp(-np.abs(lmbd_cont_real))
        theta = np.pi/2 * lmbd_cont_imag
        
        lmbd_real = r * np.cos(theta)
        lmbd_imag = r * np.sin(theta)
        
        eig_hat = np.concatenate((lmbd_real + 1j * lmbd_imag, lmbd_real - 1j * lmbd_imag))
        return eig_hat


class model004(nn.Module):
    def __init__(self, Ny, Nu, Nhidden):
        super(model004, self).__init__()
        self.Ny, self.Nu, self.Nhidden = Ny, Nu, Nhidden 

        log_lmbd_cont_real = np.random.rand(Nhidden//2) # (*, Nhidden//2)
        log_lmbd_cont_imag = np.random.rand(Nhidden//2) # (*, Nhidden//2)
        B = np.random.randn(Nhidden, Nu)/np.sqrt(Nu)

        self._log_lmbd_cont_real = nn.Parameter(torch.from_numpy(log_lmbd_cont_real.astype(np.float32)))
        self._log_lmbd_cont_imag = nn.Parameter(torch.from_numpy(log_lmbd_cont_imag.astype(np.float32)))
        self._B = nn.Parameter(torch.from_numpy(B.astype(np.float32)))

        self.y2x = nn.Linear(Ny, Nhidden)
        self.x2y = nn.Linear(Nhidden, Ny)

    def forward(self, _y0, _U):
        # _y0: (*, Ny), _U: (Nhrz, *, Nu)
        Nhrz = _U.shape[0]

        X = []

        _r = torch.exp(-torch.exp(-torch.abs(self._log_lmbd_cont_real))) # (Nx//2,)
        _theta = np.pi/2 * torch.exp(-torch.abs(self._log_lmbd_cont_imag)) # (Nx//2,)
        _lmbd_real = _r * torch.cos(_theta) # (Nx//2,)
        _lmbd_imag = _r * torch.sin(_theta) # (Nx//2,)

        _A11 = torch.diag(_lmbd_real) # = A22, (Nx//2, Nx//2)
        _A21 = torch.diag(_lmbd_imag) # = A21, (Nx//2, Nx//2)
        _A = torch.cat((torch.cat((_A11, -_A21), dim=1), 
            torch.cat((_A21, _A11), dim=1)), dim=0) #(Nx, Nx)

        _normalized_factor_tmp = torch.sqrt(1 - torch.exp(-2*torch.exp(-torch.abs(self._log_lmbd_cont_real)))) # (Nx//2,)
        _normalized_factor = torch.cat((_normalized_factor_tmp, _normalized_factor_tmp)) # (Nx,)
        _Bu = torch.matmul(_U, self._B.t())  * _normalized_factor # (Nhrz, *, Nx)

        _x = self.y2x(_y0) # (*, Nhidden)
        for k1 in range(Nhrz):
            _x = torch.matmul(_x, _A.t()) + _Bu[k1,:]
            X.append(_x)
        _X = torch.stack(X, dim=0) # (Nhrz, *, Nx)
        _Y = self.x2y(_X) # (Nhrz, *, Ny)
        return _Y, dict()

    def get_eig(self):

        log_lmbd_cont_real = self._log_lmbd_cont_real.data.numpy()
        log_lmbd_cont_imag = self._log_lmbd_cont_imag.data.numpy()

        r = np.exp(-np.exp(-np.abs(log_lmbd_cont_real))) # (Nx//2,)
        theta = np.pi/2 * np.exp(-np.abs(log_lmbd_cont_imag)) # (Nx//2,)
        lmbd_real = r * np.cos(theta) # (Nx//2,)
        lmbd_imag = r * np.sin(theta) # (Nx//2,)
        
        eig_hat = np.concatenate((lmbd_real + 1j * lmbd_imag, lmbd_real - 1j * lmbd_imag))
        return eig_hat

class model003a(nn.Module):
    def __init__(self, Ny, Nu, Nhidden):
        super(model003a, self).__init__()
        self.Ny, self.Nu, self.Nhidden = Ny, Nu, Nhidden 

        lmbd_cont_real = np.random.rand(Nhidden//2) # (*, Nhidden//2)
        lmbd_cont_imag = np.random.rand(Nhidden//2) # (*, Nhidden//2)
        B = np.random.randn(Nhidden, Nu)/np.sqrt(Nu)

        self._lmbd_cont_real = nn.Parameter(torch.from_numpy(lmbd_cont_real.astype(np.float32)))
        self._lmbd_cont_imag = nn.Parameter(torch.from_numpy(lmbd_cont_imag.astype(np.float32)))
        self._B = nn.Parameter(torch.from_numpy(B.astype(np.float32)))

        self.y2x = nn.Linear(Ny, Nhidden)
        self.x2y = nn.Linear(Nhidden, Ny)

    def forward(self, _y0, _U):
        # _y0: (*, Ny), _U: (Nhrz, *, Nu)
        Nhrz = _U.shape[0]

        X = []
        _Bu = torch.matmul(_U, self._B.t()) # (Nhrz, *, Nx)
        _Bu.unsqueeze_(2) # (Nhrz, *, 1, Nx)

        _lmbd_cont_real_series = torch.arange(float(Nhrz),0,-1).reshape(-1,1) \
            * self._lmbd_cont_real.reshape(1,-1) # (Nhrz, Nx//2,)
        _lmbd_cont_imag_series = torch.arange(float(Nhrz),0,-1).reshape(-1,1) \
            * self._lmbd_cont_imag.reshape(1,-1) # (Nhrz, Nx//2,)

        _r_series = torch.exp(-torch.abs(_lmbd_cont_real_series)) # (Nhrz,Nx//2,)
        _theta_series = np.pi/2 * _lmbd_cont_imag_series # (Nhrz,Nx//2,)
        _lmbd_real_series = _r_series * torch.cos(_theta_series) # (Nhrz,Nx//2,)
        _lmbd_imag_series = _r_series * torch.sin(_theta_series) # (Nhrz,Nx//2,)

        _A11_series = torch.diag_embed(_lmbd_real_series) # = A22, (Nhrz, Nx//2, Nx//2)
        _A21_series = torch.diag_embed(_lmbd_imag_series) # = A21, (Nhrz, Nx//2, Nx//2)
        _A_series = torch.cat((torch.cat((_A11_series, -_A21_series), dim=2), 
            torch.cat((_A21_series, _A11_series), dim=2)), dim=1) #(Nhrz, Nx, Nx)
        _A_series.unsqueeze_(1) # (Nhrz, 1, Nx, Nx)

        _x0 = self.y2x(_y0) # (*, Nhidden)
        _x0.unsqueeze_(1) # (*, 1, Nhidden)
        for k1 in range(Nhrz):
            _x = _Bu[k1,:,0,:] \
                + torch.sum(_Bu[:k1,:] * _A_series[Nhrz-k1:,:], dim=(0,3)) \
                + torch.sum(_A_series[Nhrz-1-k1,:] * _x0, dim=2) # (*, Nx)
            X.append(_x)
        _X = torch.stack(X, dim=0) # (Nhrz, *, Nx)
        _Y = self.x2y(_X) # (Nhrz, *, Ny)
        return _Y, dict()

    def get_eig(self):
        lmbd_cont_real = self._lmbd_cont_real.data.numpy()
        lmbd_cont_imag = self._lmbd_cont_imag.data.numpy()
        
        r = np.exp(-np.abs(lmbd_cont_real))
        theta = np.pi/2 * lmbd_cont_imag
        
        lmbd_real = r * np.cos(theta)
        lmbd_imag = r * np.sin(theta)
        
        eig_hat = np.concatenate((lmbd_real + 1j * lmbd_imag, lmbd_real - 1j * lmbd_imag))
        return eig_hat

class model003b(nn.Module):
    def __init__(self, Ny, Nu, Nhidden):
        super(model003b, self).__init__()
        self.Ny, self.Nu, self.Nhidden = Ny, Nu, Nhidden 

        lmbd_cont_real = np.random.rand(Nhidden//2) # (*, Nhidden//2)
        lmbd_cont_imag = np.random.rand(Nhidden//2) # (*, Nhidden//2)
        B = np.random.randn(Nhidden, Nu)/np.sqrt(Nu)
        multiplier_on_B = np.zeros(Nhidden)

        self._lmbd_cont_real = nn.Parameter(torch.from_numpy(lmbd_cont_real.astype(np.float32)))
        self._lmbd_cont_imag = nn.Parameter(torch.from_numpy(lmbd_cont_imag.astype(np.float32)))
        self._B = nn.Parameter(torch.from_numpy(B.astype(np.float32)))
        self._multiplier_on_B = nn.Parameter(torch.from_numpy(multiplier_on_B.astype(np.float32)))

        self.y2x = nn.Linear(Ny, Nhidden)
        self.x2y = nn.Linear(Nhidden, Ny)

    def forward(self, _y0, _U):
        # _y0: (*, Ny), _U: (Nhrz, *, Nu)
        Nhrz = _U.shape[0]

        X = []
        _Bu = torch.matmul(_U, self._B.t() ) \
            * torch.exp(self._multiplier_on_B)# (Nhrz, *, Nx)

        _r = torch.exp(-torch.abs(self._lmbd_cont_real)) # (Nx//2,)
        _theta = np.pi/2 * self._lmbd_cont_imag # (Nx//2,)
        _lmbd_real = _r * torch.cos(_theta) # (Nx//2,)
        _lmbd_imag = _r * torch.sin(_theta) # (Nx//2,)

        _A11 = torch.diag(_lmbd_real) # = A22, (Nx//2, Nx//2)
        _A21 = torch.diag(_lmbd_imag) # = A21, (Nx//2, Nx//2)
        _A = torch.cat((torch.cat((_A11, -_A21), dim=1), 
            torch.cat((_A21, _A11), dim=1)), dim=0) #(Nx, Nx)

        _x = self.y2x(_y0) # (*, Nhidden)
        for k1 in range(Nhrz):
            _x = torch.matmul(_x, _A.t()) + _Bu[k1,:]
            X.append(_x)
        _X = torch.stack(X, dim=0) # (Nhrz, *, Nx)
        _Y = self.x2y(_X) # (Nhrz, *, Ny)
        return _Y, dict()

    def get_eig(self):
        lmbd_cont_real = self._lmbd_cont_real.data.numpy()
        lmbd_cont_imag = self._lmbd_cont_imag.data.numpy()
        
        r = np.exp(-np.abs(lmbd_cont_real))
        theta = np.pi/2 * lmbd_cont_imag
        
        lmbd_real = r * np.cos(theta)
        lmbd_imag = r * np.sin(theta)
        
        eig_hat = np.concatenate((lmbd_real + 1j * lmbd_imag, lmbd_real - 1j * lmbd_imag))
        return eig_hat



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
        multiplier = np.concatenate((np.sqrt(1-np.exp(-2*alpha)), np.sqrt(1-np.exp(-2*alpha))))
        B = multiplier.reshape((-1,1)) *  np.random.randn(Nhidden, Nu)
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


def run_training(mdl_constructor, data_generator, Nhidden, Nbatch, Nepoch, Nhrz, 
        weight_on_diff_loss = 0., print_log = True):
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
            _loss_plain = criterion(_Y, _Yhat)
            #_loss_diff = criterion( _Y[1:,:] - _Y[:-1,:],  _Yhat[1:,:] - _Yhat[:-1,:])

            _loss = _loss_plain 

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

        Ny, Nu, Nhidden = np.random.randint(1, 16, size=(3,))

        Nbatch = 2**6
        Nhrz = 2**7

        y0 =  np.random.randn(Nbatch, Ny).astype(np.float32)
        U =  np.random.randn(Nhrz, Nbatch, Nu).astype(np.float32)

        _y0 = torch.tensor(y0) # (*, Ny)
        _U = torch.tensor(U) # (Nhrz, *, Nu)

        print("Ny, Nu, Nhidden = (%d, %d, %d)" % (Ny, Nu, Nhidden))
        for model_constructor in [model001, model002, model003, 
            model004, model003b]:

            mdl = model_constructor(Ny, Nu, Nhidden*2)

            t_bgn = time.time()
            _Yhat, _ = mdl(_y0, _U) # Nhrz, *, Ny
            elapsed_time = time.time() - t_bgn
            print("%40s \t %8.4f" % (model_constructor, elapsed_time))

            Yhat = _Yhat.data.numpy()
            assert Yhat.shape == (Nhrz, Nbatch, Ny)
            assert np.all(~np.isnan(Yhat))

            eig_hat = mdl.get_eig()
            assert eig_hat.shape == (2*Nhidden,), str(eig_hat.shape) + ", " + str(Nhidden*2)

def test002():

    for model_constructor in [model001, model002, model003, model004]*3:

        data_generator = DataGenerator(2**2, Ntrain = 2**7)
        Ny, Nu = data_generator.Ny, data_generator.Nu
        Nhidden, Nbatch, Nepoch, Nhrz =  np.random.randint(1, 20, size=(4,))

        run_training(lambda Nhidden: model_constructor(Ny, Nu, Nhidden*2), data_generator, Nhidden, Nbatch, Nepoch, Nhrz)

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

        

