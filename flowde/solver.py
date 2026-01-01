# ------------------------------------------------------------------------
# GSOC PINNDE Project
# Sijil Jose, Pushpalatha C. Bhat, Sergei Gleyzer, Harrison B. Prosper
# Created: Tue May 27 2025
# Updated: Thu May 29 2025 HBP: generalize sigma(t)
# Updated: Sat Jul 19 2025 HBP: inspired by Sijil's logsumexp suggestion, 
#                              useg torch.softmax to compute weights.
# ------------------------------------------------------------------------
# standard system modules
import os, sys, re

# standard research-level machine learning toolkit from Meta
try:
    import torch
    import torch.nn as nn
except:
    raise ImportError('''
    Please import PyTorch, e.g.:

    conda install pytorch
    ''')

# standard module for array manipulation
try:
    import numpy as np
except:
    raise ImportError('''
    Please install numpy!
    ''')

try:
    from tqdm import tqdm
except:
    raise ImportError('''
    Please install tqdm, e.g.:

    conda install tqdm
    ''')
    
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
class qVectorField(nn.Module):    
    '''
    Compute a Monte Carlo approximation of the q vector field of 
    the reverse-time diffusion equation, 
    
     [sigma_0 + (1 - sigma_0) * t] * dx/dt = (1-sigma0) * x - q(t, x),

    where xt = x(t) is a d-dimensional vector and q(t, xt) is a 
    d-dimensional time-dependent vector field and s0 is that value of 
    sigma(t) at t=0.
     
    The vector field q(t, x) is defined by a d-dimensional integral 
    which is approximated with a Monte Carlo (MC)-generated sample, x0, 
    of shape (M, d), where M is the sample size. 

    Example
    -------

    q = qVectorField(x0)
        :  :
    qt = q(t, x)
    '''
    def __init__(self, x0, sigma0=1e-3, debug=False):

        super().__init__()

        assert(x0.ndim==2)
        # x0.shape: (M, d)
        
        # Change shape of x0 from (M, d) to (1, M, d)
        # so that broadcasting works correctly later.
        self.x0 = x0.unsqueeze(0)

        self.sigma0 = sigma0
        self.debug  = debug

        if debug:
            print('qVectorField.__init__: x0.shape', x0.shape)

    def set_debug(self, debug=True):
        self.debug = debug

    def sigma(self, t):
        return self.sigma0 + (1 - self.sigma0) * t

    def forward(self, t, x):
        assert(x.ndim==2)

        if type(t) == type(x):
            assert(t.ndim==2)

            # Change shape of t so that broadcasting works correctly
            t = t.unsqueeze(1)
            # t.shape: (N, 1) => (N, 1, 1)
            
        debug = self.debug
        x0 = self.x0
        sigma0 = self.sigma0
        
        # Change shape xt so that broadcasting works correctly
        x = x.unsqueeze(1)
        # x.shape: (N, d) => (N, 1, d)

        if debug:
            print('qVectorField(BEGIN)')
            print('  x.shape: ', x.shape)
            print('  x0.shape:', x0.shape)

        alphat = 1 - t
        sigmat = self.sigma(t)
        y  = x - alphat * x0
        vt = y / sigmat
        # vt.shape: (N, M, d)
        
        if debug:
            print('  qVectorField: vt.shape', vt.shape)
            print('                vt', vt)

        if torch.isnan(vt).any():
            if torch.isnan(y):
                print('NAN in y')
            if torch.isnan(sigmat):
                print('NAN in sigmat')
            raise ValueError("vt contains at least one NAN")

        if torch.isinf(vt).any():
            raise ValueError("vt contains at least one INF")

        # Sum over arguments of exponential, that is,
        # over the d-dimensions of each element in x0,
        # so that we get the product of d normal densities
        # for each Monte Carlo-sampled point and for each 
        # batch instance. 
        u = (vt*vt).sum(dim=-1)/2
        # u.shape: (N, M)
        
        # For each row, find minimum value, umin, of u
        # and change shape of umin from (N,) => (N, 1) so
        # that broadcasting with u (of shape (N, M)) will
        # work correctly. Note: min(...) returns (min(u), pos(u))
        # where "min(u)" is the minimum value of u and pos(u) 
        # its ordinal value (position).
        umin = u.min(dim=-1)[0].unsqueeze(-1)
        
        if debug:
            print('  qVectorField: u.shape, umin.shape', 
		u.shape, umin.shape)

        # Compute weights
        # ---------------
        # Note: by adding umin to -u, we guarantee that at least
        # one term in the sum exp will be equal to unity, while all 
        # the rest will be < 1.  The softmax is performed along the
        # Monte Carlo sample direction.
        wt = torch.softmax(-u + umin, dim=-1)
        # wt.shape: (N, M)

        # Compute effective count
        #neff = 1.0 / (wt*wt).sum(dim=-1)
        
        if debug:
            print('  qVectorField: wt.shape', wt.shape)
            print('              : wt.sum', wt.sum(dim=-1))
  
        if torch.isnan(wt).any():
            raise ValueError("wt contains at least one NAN")

        if torch.isinf(wt).any():
            raise ValueError("wt contains at least one INF")

        # Now sum over the Monte Carlo sample of M weighted 
        # elements of x0
        # x0.shape: (1, M, d)
        # wt.shape: (N, M) => (N, M, 1)
        x0_wt = x0 * wt.unsqueeze(-1)
        # x0_wt.shape: (N, M, d)

        if debug:
            print('  qVectorField: x0_wt.shape', x0_wt.shape)

        if torch.isnan(x0_wt).any():
            raise ValueError("x0_wt contains at least one NAN")

        # Sum over the MC sample dimension of x0_wt
        qt = x0_wt.sum(dim=1)
        # qt.shape: (N, d)

        if debug:
            print('  qVectorField: qt.shape', qt.shape)
            print('qVectorField(END)')
 
        if torch.isnan(qt).any():
            raise ValueError("qt contains at least one NAN")

        return qt

    def Gprime(self, t, x):
        qt = self(t, x)
        return (1-self.sigma0)*x - qt
        
    def G(self, t, x):
        return self.Gprime(t, x) / self.sigma(t)
# ------------------------------------------------------------------------
def get_normal_sample(x):
    try:
        x = x.cpu()
    except:
        pass
    means = np.zeros_like(x)
    scales = np.ones_like(x)
    return torch.Tensor(np.random.normal(loc=means, scale=scales))
        
def get_target_sample(x, size=5000):
    ii = np.random.randint(0, len(x)-1, size)
    return torch.Tensor(x[ii])

class FlowDE(nn.Module):    
    '''
    Given standard normal vectors z = x(t=1), compute target vectors 
    x0 = x(t=0) by mapping z to x0 deterministically. x0, which is 
    of shape (M, d), where M is the Monte Carlo (MC) sample size and d 
    the dimension of the vector x0 = x(0), is used to compute a MC 
    approximation of the q vector field. The tensor z is of shape (N, d), 
    where N is the number of points sampled from a d-dimensionbal 
    standard normal. 

    Utility functions
    =================
    1. get_normal_sample(X0) returns a tensor z = x(1), with same shape 
    as X0, whose elements are sampled from a d-dimensional Gaussian. 

    2. get_target_sample(X0, M) returns a sample of points, x0, of size M 
    from X0, which will be used to approximate the q vector field.

    Example
    -------
    N = 4000
    M = 4000
    
    z  = get_normal_sample(X0[:N]).to(DEVICE)
    x0 = get_target_sample(X0, M).to(DEVICE)
    
    flow = FlowDE(x0)

    y = flow(z)
    
    '''
    def __init__(self, x0,
                     sigma0=1e-3, T=500,
                     savepath=False, step=5, debug=False,
                     device=torch.device('cuda'
                                             if torch.cuda.is_available()
                                             else 'cpu')):
        
        # x0: MC sample of shape (M, d)
        # T:  number of time steps in [1, 0]
        
        super().__init__()

        assert(x0.ndim==2)
        
        self.q = qVectorField(x0, sigma0, debug).to(device)
        self.sigma0 = sigma0
        
        if T < 4: T = 4

        self.T = T
        self.h = 1/T # step size
        self.savepath = savepath
        self.step = step
        self.debug = debug

    def set_debug(self, debug=True):
        self.debug = debug

    def G(self, t, xt):
        # t is either a float or a 2D tensor of shape (N, 1)
        # xt.shape: (N, d)
        return  self.q.G(t, xt)

    def forward(self, z):
        assert(z.ndim==2)
        
        debug = self.debug

        savepath = self.savepath
        T = self.T
        h = self.h
        t = 1      # initial "time"
        xt= z
        dim = xt.shape[-1]
        
        if debug:
            print('FlowDE.forward: xt.shape', xt.shape)
            print('FlowDE.forward: t', t)

        if savepath:
            y = [xt]
            t_y = [t]
            
        G1 = self.G(t, xt)
        
        for i in tqdm(range(T-1)):
            t = (T-i-1) * h
            #t -= h
            if t < 0: 
                break
                
            if debug:
                print('FlowDE.forward: t', t)
                print('FlowDE.forward: xt.shape', xt.shape) 
                print('FlowDE.forward: G1.shape', G1.shape)
                
            G2 = self.G(t, xt - G1 * h)

            xt = xt - (G1 + G2) * h / 2
            
            G1 = G2.detach().clone()

            if savepath:
                if (i+1) % self.step == 0:
                    y.append(xt)
                    t_y.append(t)

        if savepath:
            return torch.Tensor(t_y), torch.concat(y, dim=0).view(len(y),
                                                                      *xt.shape)
        else:
            return xt
            
