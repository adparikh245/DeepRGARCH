import numpy as np
import pandas as pd
import scipy as sp
from rerech import resampling as rs, smc
from scipy import stats
import utils as ut  # Your custom utilities

class MultiRVRealRECH_LSTM(smc.SMC):
    def __init__(self, data=None, n_rv_measures=8, **kwargs):
        super().__init__(**kwargs)
        self.Y, self.RV_measures = data
        self.n_rv_measures = n_rv_measures
        self.T = self.Y.shape[0]
        self.wgts_ = rs.Weights()
        
    def safe_sqrt(self, x):
        return np.sqrt(np.maximum(x, 1e-10))
    
    def _create_rv_parameter(self, base_name):
        return [f'{base_name}_{i}' for i in range(self.n_rv_measures)]
    
    def calculate_variance(self, omega, beta, gamma, rv, var_prev):
        var_next = omega + beta * var_prev
        for i in range(self.n_rv_measures):
            var_next += gamma[i] * rv[:, i]
        return np.maximum(1e-10, var_next)
    
    def loglik(self, theta, get_v=False):
        N = self.Y.shape[0]
        var = np.zeros((N, theta.shape[0]))
        var[0] = np.var(self.Y)
        omega = np.zeros((N, theta.shape[0]))
        omega[0] = theta['beta0']
        
        # Initialize all RV-related parameters
        gamma = [theta[f'gamma_{i}'] for i in range(self.n_rv_measures)]
        xi = np.zeros((N, self.n_rv_measures, theta.shape[0]))
        U = np.zeros((N, self.n_rv_measures, theta.shape[0]))
        
        # LSTM states
        h = np.zeros((N, theta.shape[0]))
        c = np.zeros((N, theta.shape[0]))
        h_rv = np.zeros((N, theta.shape[0]))
        c_rv = np.zeros((N, theta.shape[0]))
        
        for n in range(N-1):
            # Volatility LSTM Gates
            rv_input = sum(theta[f'v3f_{i}']*self.RV_measures[n,i] for i in range(self.n_rv_measures))
            gf = ut.sigmoid(theta['v0f']*omega[n] + theta['v1f']*self.Y[n] + 
                          theta['v2f']*var[n] + rv_input + theta['wf']*h[n] + theta['bf'])
            
            # Similar for gi, go, chat using list comprehensions
            # ... (full gate implementations from previous code)
            
            # Variance update
            var[n+1] = self.calculate_variance(omega[n+1], theta['beta'], gamma, 
                                             self.RV_measures[n], var[n])
            
            # RV prediction LSTMs
            for i in range(self.n_rv_measures):
                # RV-specific gate calculations
                gf_rv = ut.sigmoid(theta[f'v0f_rv_{i}']*xi[n,i] + theta['v1f_rv']*self.Y[n] + 
                                  theta['v2f_rv']*var[n] + theta[f'v3f_rv_{i}']*self.RV_measures[n,i] + 
                                  theta['wf_rv']*h_rv[n] + theta['bf_rv'])
                # ... other gates
                
                # Measurement equation
                eps = self.Y[n+1]/np.sqrt(var[n+1])
                U[n+1,i] = self.RV_measures[n+1,i] - xi[n+1,i] - theta[f'phi_{i}']*var[n+1] - \
                          theta[f'tau1_{i}']*eps - theta[f'tau2_{i}']*(eps**2 - 1)
        
        # Log-likelihood calculations
        llik_y = np.sum(stats.norm.logpdf(self.Y[1:], 0, self.safe_sqrt(var[1:])), axis=0)
        llik_rv = sum(np.sum(stats.norm.logpdf(U[1:,i], 0, self.safe_sqrt(theta[f'sigmau2_{i}']))) 
                           for i in range(self.n_rv_measures))
        
        if get_v:
            self.var_ls = np.average(var[1:], axis=1, weights=self.wgts.W)
            
        return llik_y + llik_rv

def configure_smc(n_rv):
    prior_spec = {
        'beta0': ut.TruncatedNormal(0.01, 0.1, 0, 0.1),
        'beta': ut.Beta(5, 1),
        **{f'gamma_{i}': ut.TruncatedNormal(0.1, 0.05, 0, 0.3) for i in range(n_rv)},
        **{f'phi_{i}': ut.TruncatedNormal(0.8, 0.1, 0, 1) for i in range(n_rv)},
        # ... add all other parameters
    }
    return prior_spec