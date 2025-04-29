import numpy as np
import scipy as sp
import sys, os
sys.path.append(os.path.abspath("/Users/ananyaparikh/Documents/Coding/DeepRGARCH/DeepRGARCH"))
from scipy import stats
from rerech import smc, utils as ut, resampling as rs
from rerech import model_patch


################################################################### 
# GARCH
class GARCH(smc.SMC):
    def __init__(self, data=None, **kwargs):
        super().__init__(**kwargs)
        self.Y = data
        self.T = self.Y.shape[0]
    
    def loglik(self, theta, get_v=False):  
        var    = np.zeros((self.Y.shape[0], theta.shape[0])) 
        var[0] = np.var(self.Y)
        for n in range(self.Y.shape[0]-1):  #yt is excluded as it calculate vart+1
            var[n+1] = theta['omega'] + theta['alpha'] * np.square(self.Y[n]) + theta['beta'] * var[n]
        llik = np.sum(stats.norm.logpdf(self.Y[1:], loc=0, scale = np.sqrt(var[1:])), axis=0) #y0, var0 is excluded, end with yt, vart
        
        if get_v:
            self.var_ls = np.average(var[1:], axis=1, weights=self.wgts.W)
        return llik #(N,) array
class GARCHD(smc.SMCD):
    def __init__(self, data=None, **kwargs):
        super().__init__(**kwargs)
        self.Y_train, self.Y_test = data
        self.T = self.Y_test.shape[0]
        
    def loglik(self, theta, t=None, lpyt=False):
        Y = np.concatenate((self.Y_train, self.Y_test[:t+1]))
        N = Y.shape[0]
        var = np.zeros((N, theta.shape[0])) 
        var[0] = np.var(Y)
        for n in range(N-1): # yt is excluded as it calculate vart+1
            var[n+1] = theta['omega'] + theta['alpha'] * np.square(Y[n]) + theta['beta'] * var[n]
        if lpyt:
            llik = stats.norm.logpdf(Y[-1], loc=0, scale = np.sqrt(var[-1]))
            v = np.average(var[-1]) if self.rs_flag else np.average(var[-1],weights=self.wgts.W) 
            self.var_ls.append(v)       
        else:       
            llik = np.sum(stats.norm.logpdf(Y, loc=0, scale = np.sqrt(var)), axis=0)
        return llik
    
class GARCHD_fitless(smc.SMCD):
    def __init__(self, data=None, len_fit=1000, **kwargs):
        super().__init__(**kwargs)
        self.Y_train, self.Y_test = data
        self.T = self.Y_test.shape[0]
        self.len_fit = len_fit
        
    def loglik(self, theta, t=None, lpyt=False):
        Y = np.concatenate((self.Y_train, self.Y_test[:t+1]))
        N = Y.shape[0]
        var = np.zeros((N, theta.shape[0])) 
        var[0] = np.var(Y)
        for n in range(N-1): # yt is excluded as it calculate vart+1
            var[n+1] = theta['omega'] + theta['alpha'] * np.square(Y[n]) + theta['beta'] * var[n]
        if lpyt:
            if self.tdist:
                llik = stats.t.logpdf(Y[-1],theta['nu'],loc=0,scale = np.sqrt(var[-1]))
                nu = np.average(theta['nu']) if self.rs_flag else np.average(theta['nu'],weights=self.wgts.W)
                self.nu_ls.append(nu) 
            else:
                llik = stats.norm.logpdf(Y[-1], loc=0, scale = np.sqrt(var[-1]))
            v = np.average(var[-1]) if self.rs_flag else np.average(var[-1],weights=self.wgts.W) 
            self.var_ls.append(v)       
        else:       
            llik = np.sum(stats.norm.logpdf(Y[-self.len_fit], loc=0, scale = np.sqrt(var[-self.len_fit])), axis=0)
        return llik
    
################################################################### 
# RealGARCH
class RealGARCH(smc.SMC):
    def __init__(self,data=None, **kwargs):
        super().__init__(**kwargs)
        self.Y, self.RV = data
        self.T = self.Y.shape[0]
        self.wgts_ = rs.Weights()
    
    def loglik(self, theta, get_v=False):  
        N = self.Y.shape[0]
        var    = np.zeros((self.Y.shape[0], theta.shape[0]))
        var[0] = np.var(self.Y)
        self.RV[0]  = np.mean(self.RV)
        for n in range(N-1):  #yt is excluded as it calculate vart+1
            var[n+1] = theta['omega'] + theta['beta']*var[n] + theta['gamma']*self.RV[n]
        eps = self.Y/np.sqrt(var)
        U   = self.RV - theta['xi'] - theta['phi']*var-theta['tau1']*eps-theta['tau2']*(np.square(eps)-1)
        
        llik_y = np.sum(stats.norm.logpdf(self.Y[1:],loc=0, scale=np.sqrt(var[1:])), axis=0)  
        llik_rv = np.sum(stats.norm.logpdf(U[1:], loc=0, scale=np.sqrt(theta['sigmau2'])), axis=0)
        if get_v:
            self.var_ls = np.average(var[1:], axis=1, weights=self.wgts.W)
        return llik_y+llik_rv #(N,) array
    
    
    def loglik_(self, theta, get_v=False):  
        N = self.Y.shape[0]
        var    = np.zeros((self.Y.shape[0], theta.shape[0]))
        var[0] = np.var(self.Y)
        self.RV[0]  = np.mean(self.RV)
        for n in range(N-1):  #yt is excluded as it calculate vart+1
            var[n+1] = theta['omega'] + theta['beta']*var[n] + theta['gamma']*self.RV[n]
        eps = self.Y/np.sqrt(var)
        U   = self.RV - theta['xi'] - theta['phi']*var-theta['tau1']*eps-theta['tau2']*(np.square(eps)-1)
        if self.tdist:
            llik_y = np.sum(stats.t.logpdf(self.Y[1:],theta['nu'],loc=0, scale=np.sqrt(var[1:])), axis=0)  
        else:
            llik_y = np.sum(stats.norm.logpdf(self.Y[1:],loc=0, scale=np.sqrt(var[1:])), axis=0)  
        return llik_y
    
    def reweight_particles(self):
        # calculate new epn
        ESSmin = self.ESSrmin_ * self.X.N
        f = lambda e: rs.essl(e * self.X.llik) - ESSmin
        epn = self.X.shared['exponents'][-1]
        if f(1. - epn) > 0:  # we're done (last iteration)
            delta = 1. - epn
            new_epn = 1. # set 1. manually so that we can safely test == 1.
        else:
            delta = sp.optimize.brentq(f, 1.e-12, 1. - epn)  # secant search
            # left endpoint is >0, since f(0.) = nan if any likelihood = -inf
            new_epn = epn + delta
        self.X.shared['exponents'].append(new_epn)
        # calculate delta llik
        dllik = delta * self.X.llik
        self.X.lpost += dllik
        #update weights
        self.wgts = self.wgts.add(dllik)
        self.wgts_ = rs.Weights(delta*self.loglik_(self.X.theta))
        
    def __next__(self):
        if self.done():
            self.loglik(self.X.theta, get_v=True)
            self.var_ls = np.array(self.var_ls)
            self.w_ls = np.array(self.w_ls)
            raise StopIteration
        if self.t == 0:
            self.generate_particles()
        else:
            self.resample_move()
        self.reweight_particles()
        self.wgts_ls.append(self.wgts_)
        self.X_ls.append(self.X)
        if self.verbose:
            print("t={}, accept_rate={:.2f}, accept_rate2={:.2f}, epn={:.3f}".format(self.t, np.average(self.X.shared['acc_rates'][-1]),self.X.shared['acc_rates2'][-1],self.X.shared['exponents'][-1]))
        self.t += 1   
class RealGARCHD(smc.SMCD):
    def __init__(self,data=None, **kwargs):
        super().__init__(**kwargs)
        self.Y_train, self.Y_test, self.RV_train, self.RV_test = data
        self.T = self.Y_test.shape[0]

    def loglik(self, theta, t=None, lpyt=False): 
        Y = np.concatenate((self.Y_train, self.Y_test[:t+1]))
        RV= np.concatenate((self.RV_train, self.RV_test[:t+1]))
        N = Y.shape[0]
        var    = np.zeros((N, theta.shape[0]))
        var[0] = np.var(Y)
        for n in range(N-1):  #yt is excluded as it calculate vart+1
            var[n+1] = theta['omega'] + theta['beta']*var[n] + theta['gamma']*RV[n]
        eps = Y/np.sqrt(var)
        U   = RV - theta['xi']-theta['phi']*var-theta['tau1']*eps-theta['tau2']*(np.square(eps)-1)
        if lpyt:
            if self.tdist:
                llik_y = stats.t.logpdf(Y[-1],theta['nu'],loc=0, scale=np.sqrt(var[-1])) 
                nu = np.average(theta['nu']) if self.rs_flag else np.average(theta['nu'],weights=self.wgts.W)
                self.nu_ls.append(nu)                 
            else:
                llik_y = stats.norm.logpdf(Y[-1],loc=0,scale=np.sqrt(var[-1]))
            llik_rv = stats.norm.logpdf(U[-1],loc=0, scale=np.sqrt(theta['sigmau2'])) 
            v = np.average(var[-1]) if self.rs_flag else np.average(var[-1],weights=self.wgts.W) 
            self.var_ls.append(v)           
        else:
            if self.tdist:
                llik_y = np.sum(stats.t.logpdf(Y,theta['nu'],loc=0,scale=np.sqrt(var)),axis=0)  
            else:
                llik_y = np.sum(stats.norm.logpdf(Y,loc=0,scale=np.sqrt(var)),axis=0)  
            llik_rv = np.sum(stats.norm.logpdf(U, loc=0, scale=np.sqrt(theta['sigmau2'])),axis=0)
        return llik_y+llik_rv
    
################################################################### 
#RECH
class RECH(smc.SMC):  
    def __init__(self, data=None, **kwargs):
        super().__init__(**kwargs)
        self.Y = data
        self.T = self.Y.shape[0]
        
    def loglik(self, theta, get_v=False):  
        N = self.Y.shape[0]
        var    = np.zeros((N, theta.shape[0]))
        var[0] = np.var(self.Y)
        omega = np.zeros((N, theta.shape[0]))
        omega[0] = theta['beta0'] # + theta['beta1'] * h0 which is 0 
        h = np.zeros((N, theta.shape[0])) # h0 = 0
        for n in range(N-1):  # yt generate std t+1, while we don't need t+1
            h[n+1] = ut.relu(theta['v0']*omega[n]+theta['v1']*self.Y[n]+theta['v2']*var[n]+theta['w']*h[n]+theta['b'])
            omega[n+1] = ut.relu(theta['beta0'] + theta['beta1'] * h[n+1])
            var[n+1] = omega[n+1]+theta['alpha']*np.square(self.Y[n])+theta['beta']*var[n]  
        llik = np.sum(stats.norm.logpdf(self.Y[1:],loc=0,scale=np.sqrt(var[1:])),axis=0) # y0, std0 is excluded
        if get_v:
            self.var_ls = list(np.average(var[1:], axis=1, weights=self.wgts.W))
            self.w_ls = list(np.average(omega[1:], axis=1, weights=self.wgts.W))
        return llik # (N,)   
class RECHD(smc.SMCD):
    def __init__(self, data=None, **kwargs):
        super().__init__(**kwargs)
        self.Y_train, self.Y_test = data
        self.T = self.Y_test.shape[0]
    
    def loglik(self, theta, t=None, lpyt=False):  
        Y = np.concatenate((self.Y_train, self.Y_test[:t+1]))
        N = Y.shape[0]
        var    = np.zeros((N, theta.shape[0]))
        var[0] = np.var(Y)
        omega = np.zeros((N, theta.shape[0]))
        omega[0] = theta['beta0'] # + theta['beta1'] * h0 which is 0 
        h = np.zeros((N, theta.shape[0])) # h0 = 0
        for n in range(N-1):  #yt generate std t+1, while we don't need t+1
            h[n+1] = ut.relu(theta['v0']*omega[n]+theta['v1']*Y[n]+theta['v2']*var[n]+theta['w']*h[n]+theta['b'])
            omega[n+1] = ut.relu(theta['beta0'] + theta['beta1'] * h[n+1])
            var[n+1] = omega[n+1]+theta['alpha']*np.square(Y[n])+theta['beta']*var[n]
        if lpyt:
            if self.tdist:
                llik = stats.t.logpdf(Y[-1], theta['nu'],loc=0, scale=np.sqrt(var[-1]))
                nu = np.average(theta['nu']) if self.rs_flag else np.average(theta['nu'],weights=self.wgts.W)
                self.nu_ls.append(nu) 
            else: 
                 llik = stats.norm.logpdf(Y[-1], loc=0, scale=np.sqrt(var[-1]))
            v = np.average(var[-1]) if self.rs_flag else np.average(var[-1],weights=self.wgts.W) 
            w = np.average(omega[-1]) if self.rs_flag else np.average(omega[-1],weights=self.wgts.W) 
            self.var_ls.append(v) 
            self.w_ls.append(w)             
        else:
            if self.tdist:
                llik = np.sum(stats.t.logpdf(Y[1:],theta['nu'],loc=0, scale=np.sqrt(var[1:])), axis=0)  
            else:
                llik = np.sum(stats.norm.logpdf(Y[1:], loc=0, scale=np.sqrt(var[1:])), axis=0)
        return llik 

##########s######################################################### 
# RealRECH 
class RealRECH(smc.SMC):  
    def __init__(self,data=None, **kwargs):
        super().__init__(**kwargs)
        self.Y, self.RV = data
        self.T = self.Y.shape[0]
        self.wgts_ = rs.Weights()
    
    def loglik(self, theta, get_v=False, scale=True):  
        N = self.Y.shape[0]
        var    = np.zeros((N, theta.shape[0]))
        var[0] = np.var(self.Y)
        omega = np.zeros((N, theta.shape[0]))
        omega[0] = theta['beta0'] # + theta['beta1'] * h0 which is 0 
        h = np.zeros((N, theta.shape[0]))
        c = np.zeros((N, theta.shape[0]))# h0 = 0
        
        for n in range(N-1):  #yt generate std t+1, while we don't need t+1
            gf = ut.sigmoid(theta['v0f']*omega[n]+theta['v1f']*self.Y[n]+theta['v2f']*var[n]+theta['v3f']*self.RV[n]+theta['wf']*h[n]+theta['bf'])
            gi = ut.sigmoid(theta['v0i']*omega[n]+theta['v1i']*self.Y[n]+theta['v2i']*var[n]+theta['v3i']*self.RV[n]+theta['wi']*h[n]+theta['bi'])
            go = ut.sigmoid(theta['v0o']*omega[n]+theta['v1o']*self.Y[n]+theta['v2o']*var[n]+theta['v3o']*self.RV[n]+theta['wo']*h[n]+theta['bo'])         
            chat =  np.tanh(theta['v0d']*omega[n]+theta['v1d']*self.Y[n]+theta['v2d']*var[n]+theta['v3d']*self.RV[n]+theta['wd']*h[n]+theta['bd'])         
            c[n+1] = gi*chat+gf*c[n]
            h[n+1] = go*np.tanh(c[n+1])
            omega[n+1] = theta['beta0']+theta['beta1']*h[n+1]
            var[n+1] = omega[n+1]+theta['beta']*var[n]+theta['gamma']*self.RV[n]
        eps = self.Y/np.sqrt(var)
        U   = self.RV - theta['xi'] - theta['phi']*var-theta['tau1']*eps-theta['tau2']*(np.square(eps)-1)
        llik_y = np.sum(stats.norm.logpdf(self.Y[1:], loc=0, scale=np.sqrt(var[1:])), axis=0)  
        llik_rv = np.sum(stats.norm.logpdf(U[1:], loc=0, scale=np.sqrt(theta['sigmau2'])), axis=0)
        if get_v:
            self.var_ls = np.average(var[1:], axis=1, weights=self.wgts.W)
            self.w_ls = list(np.average(omega[1:], axis=1, weights=self.wgts.W))
        return llik_y+llik_rv #(N,) array
    
    def loglik_(self, theta, get_v=False, scale=True):  
        N = self.Y.shape[0]
        var    = np.zeros((N, theta.shape[0]))
        var[0] = np.var(self.Y)
        omega = np.zeros((N, theta.shape[0]))
        omega[0] = theta['beta0'] # + theta['beta1'] * h0 which is 0 
        h = np.zeros((N, theta.shape[0]))
        c = np.zeros((N, theta.shape[0]))# h0 = 0
        
        for n in range(N-1):  #yt generate std t+1, while we don't need t+1
            gf = ut.sigmoid(theta['v0f']*omega[n]+theta['v1f']*self.Y[n]+theta['v2f']*var[n]+theta['v3f']*self.RV[n]+theta['wf']*h[n]+theta['bf'])
            gi = ut.sigmoid(theta['v0i']*omega[n]+theta['v1i']*self.Y[n]+theta['v2i']*var[n]+theta['v3i']*self.RV[n]+theta['wi']*h[n]+theta['bi'])
            go = ut.sigmoid(theta['v0o']*omega[n]+theta['v1o']*self.Y[n]+theta['v2o']*var[n]+theta['v3o']*self.RV[n]+theta['wo']*h[n]+theta['bo'])         
            chat = np.tanh(theta['v0d']*omega[n]+theta['v1d']*self.Y[n]+theta['v2d']*var[n]+theta['v3d']*self.RV[n]+theta['wd']*h[n]+theta['bd'])         
            c[n+1] = gi*chat+gf*c[n]
            h[n+1] = go*np.tanh(c[n+1])
            omega[n+1] = theta['beta0']+theta['beta1']*h[n+1]
            var[n+1] = omega[n+1]+theta['beta']*var[n]+theta['gamma']*self.RV[n]
        eps = self.Y/np.sqrt(var)
        U   = self.RV - theta['xi'] - theta['phi']*var-theta['tau1']*eps-theta['tau2']*(np.square(eps)-1)
        llik_y = np.sum(stats.norm.logpdf(self.Y[1:], loc=0, scale=np.sqrt(var[1:])), axis=0)  
        return llik_y
    
    def reweight_particles(self):
        # calculate new epn
        ESSmin = self.ESSrmin_ * self.X.N
        f = lambda e: rs.essl(e * self.X.llik) - ESSmin
        epn = self.X.shared['exponents'][-1]
        if f(1. - epn) > 0:  # we're done (last iteration)
            delta = 1. - epn
            new_epn = 1. # set 1. manually so that we can safely test == 1.
        else:
            delta = sp.optimize.brentq(f, 1.e-12, 1. - epn)  # secant search
            # left endpoint is >0, since f(0.) = nan if any likelihood = -inf
            new_epn = epn + delta
        self.X.shared['exponents'].append(new_epn)
        # calculate delta llik
        dllik = delta * self.X.llik
        self.X.lpost += dllik
        #update weights
        self.wgts = self.wgts.add(dllik)
        self.wgts_ = rs.Weights(delta*self.loglik_(self.X.theta))
        
    def __next__(self):
        if self.done():
            self.loglik(self.X.theta, get_v=True)
            self.var_ls = np.array(self.var_ls)
            self.w_ls = np.array(self.w_ls)
            raise StopIteration
        if self.t == 0:
            self.generate_particles()
        else:
            self.resample_move()
        self.reweight_particles()
        self.wgts_ls.append(self.wgts_)
        self.X_ls.append(self.X)
        if self.verbose:
            print("t={}, accept_rate={:.2f}, accept_rate2={:.2f}, epn={:.3f}".format(self.t, np.average(self.X.shared['acc_rates'][-1]),self.X.shared['acc_rates2'][-1],self.X.shared['exponents'][-1]))
        self.t += 1

class RealRECHD(smc.SMCD):  
    def __init__(self,data=None, **kwargs):
        super().__init__(**kwargs)
        self.Y_train, self.Y_test, self.RV_train, self.RV_test = data
        self.T = self.Y_test.shape[0]
        
    def loglik(self, theta, t=None, lpyt=False): 
        Y = np.concatenate((self.Y_train, self.Y_test[:t+1]))
        RV= np.concatenate((self.RV_train, self.RV_test[:t+1]))
        N = Y.shape[0]
        var    = np.zeros((N, theta.shape[0]))
        var[0] = np.var(Y)
        omega = np.zeros((N, theta.shape[0]))
        omega[0] = theta['beta0'] # + theta['beta1'] * h0 which is 0 
        h = np.zeros((N, theta.shape[0]))
        c = np.zeros((N, theta.shape[0]))# h0 = 0
        for n in range(N-1):  #yt generate std t+1, while we don't need t+1
            gf = ut.sigmoid(theta['v0f']*omega[n]+theta['v1f']*Y[n]+theta['v2f']*var[n]+theta['v3f']*RV[n]+theta['wf']*h[n]+theta['bf'])
            gi = ut.sigmoid(theta['v0i']*omega[n]+theta['v1i']*Y[n]+theta['v2i']*var[n]+theta['v3i']*RV[n]+theta['wi']*h[n]+theta['bi'])
            go = ut.sigmoid(theta['v0o']*omega[n]+theta['v1o']*Y[n]+theta['v2o']*var[n]+theta['v3o']*RV[n]+theta['wo']*h[n]+theta['bo'])         
            chat = np.tanh(theta['v0d']*omega[n]+theta['v1d']*Y[n]+theta['v2d']*var[n]+theta['v3d']*RV[n]+theta['wd']*h[n]+theta['bd'])         
            c[n+1] = gi*chat+gf*c[n]
            h[n+1] = go*np.tanh(c[n+1])
            omega[n+1] = theta['beta0']+theta['beta1']*h[n+1]
            var[n+1] = omega[n+1]+theta['beta']*var[n]+theta['gamma']*RV[n]  
        eps = Y/np.sqrt(var)
        U   = RV - theta['xi']-theta['phi']*var-theta['tau1']*eps-theta['tau2']*(np.square(eps)-1)
        if lpyt:
            llik_y = stats.norm.logpdf(Y[-1], loc=0, scale=np.sqrt(var[-1]))
            llik_rv = stats.norm.logpdf(U[-1], loc=0, scale=np.sqrt(theta['sigmau2']))
            v = np.average(var[-1]) if self.rs_flag else np.average(var[-1],weights=self.wgts.W) 
            w = np.average(omega[-1]) if self.rs_flag else np.average(omega[-1],weights=self.wgts.W) 
            self.var_ls.append(v) 
            self.w_ls.append(w)       
        else:
            llik_y = np.sum(stats.norm.logpdf(Y, loc=0, scale=np.sqrt(var)), axis=0)  
            llik_rv = np.sum(stats.norm.logpdf(U, loc=0, scale=np.sqrt(theta['sigmau2'])), axis=0)  
        return llik_y+llik_rv       

##########s######################################################### 
# RealRECH + weekly, monthly rv5
class RealRECH_wm(smc.SMC):  
    def __init__(self,data=None, **kwargs):
        super().__init__(**kwargs)
        self.Y, self.RV, self.RV_week, self.RV_month = data
        self.T = self.Y.shape[0]
        self.wgts_ = rs.Weights()
    
    def loglik(self, theta, get_v=False, scale=True):  
        N = self.Y.shape[0]
        var    = np.zeros((N, theta.shape[0]))
        var[0] = np.var(self.Y)
        omega = np.zeros((N, theta.shape[0]))
        omega[0] = theta['beta0'] # + theta['beta1'] * h0 which is 0 
        h = np.zeros((N, theta.shape[0]))
        c = np.zeros((N, theta.shape[0]))# h0 = 0
        
        for n in range(N-1):  #yt generate std t+1, while we don't need t+1
            gf = ut.sigmoid(theta['v0f']*omega[n]+theta['v1f']*self.Y[n]+theta['v2f']*var[n]+theta['v3f']*self.RV[n]+theta['v4f']*self.RV_week[n]+theta['v5f']*self.RV_month[n]+theta['wf']*h[n]+theta['bf'])
            gi = ut.sigmoid(theta['v0i']*omega[n]+theta['v1i']*self.Y[n]+theta['v2i']*var[n]+theta['v3i']*self.RV[n]+theta['v4i']*self.RV_week[n]+theta['v5i']*self.RV_month[n]+theta['wi']*h[n]+theta['bi'])
            go = ut.sigmoid(theta['v0o']*omega[n]+theta['v1o']*self.Y[n]+theta['v2o']*var[n]+theta['v3o']*self.RV[n]+theta['v4o']*self.RV_week[n]+theta['v5o']*self.RV_month[n]+theta['wo']*h[n]+theta['bo'])         
            chat =  np.tanh(theta['v0d']*omega[n]+theta['v1d']*self.Y[n]+theta['v2d']*var[n]+theta['v3d']*self.RV[n]+theta['v4d']*self.RV_week[n]+theta['v5d']*self.RV_month[n]+theta['wd']*h[n]+theta['bd'])         
            c[n+1] = gi*chat+gf*c[n]
            h[n+1] = go*np.tanh(c[n+1])
            omega[n+1] = theta['beta0']+theta['beta1']*h[n+1]
            var[n+1] = omega[n+1]+theta['beta']*var[n]+theta['gamma']*self.RV[n]
        eps = self.Y/np.sqrt(var)
        U   = self.RV - theta['xi'] - theta['phi']*var-theta['tau1']*eps-theta['tau2']*(np.square(eps)-1)
        llik_y = np.sum(stats.norm.logpdf(self.Y[1:], loc=0, scale=np.sqrt(var[1:])), axis=0)  
        llik_rv = np.sum(stats.norm.logpdf(U[1:], loc=0, scale=np.sqrt(theta['sigmau2'])), axis=0)
        if get_v:
            self.var_ls = np.average(var[1:], axis=1, weights=self.wgts.W)
            self.w_ls = list(np.average(omega[1:], axis=1, weights=self.wgts.W))
        return llik_y+llik_rv #(N,) array
    
    def loglik_(self, theta, get_v=False, scale=True):  
        N = self.Y.shape[0]
        var    = np.zeros((N, theta.shape[0]))
        var[0] = np.var(self.Y)
        omega = np.zeros((N, theta.shape[0]))
        omega[0] = theta['beta0'] # + theta['beta1'] * h0 which is 0 
        h = np.zeros((N, theta.shape[0]))
        c = np.zeros((N, theta.shape[0]))# h0 = 0
        
        for n in range(N-1):  #yt generate std t+1, while we don't need t+1
            gf = ut.sigmoid(theta['v0f']*omega[n]+theta['v1f']*self.Y[n]+theta['v2f']*var[n]+theta['v3f']+theta['v4f']*self.RV_week[n]+theta['v5f']*self.RV_month[n]*self.RV[n]+theta['wf']*h[n]+theta['bf'])
            gi = ut.sigmoid(theta['v0i']*omega[n]+theta['v1i']*self.Y[n]+theta['v2i']*var[n]+theta['v3i']+theta['v4i']*self.RV_week[n]+theta['v5i']*self.RV_month[n]*self.RV[n]+theta['wi']*h[n]+theta['bi'])
            go = ut.sigmoid(theta['v0o']*omega[n]+theta['v1o']*self.Y[n]+theta['v2o']*var[n]+theta['v3o']+theta['v4o']*self.RV_week[n]+theta['v5o']*self.RV_month[n]*self.RV[n]+theta['wo']*h[n]+theta['bo'])         
            chat = np.tanh(theta['v0d']*omega[n]+theta['v1d']*self.Y[n]+theta['v2d']*var[n]+theta['v3d']+theta['v4d']*self.RV_week[n]+theta['v5d']*self.RV_month[n]*self.RV[n]+theta['wd']*h[n]+theta['bd'])         
            c[n+1] = gi*chat+gf*c[n]
            h[n+1] = go*np.tanh(c[n+1])
            omega[n+1] = theta['beta0']+theta['beta1']*h[n+1]
            var[n+1] = omega[n+1]+theta['beta']*var[n]+theta['gamma']*self.RV[n]
        eps = self.Y/np.sqrt(var)
        U   = self.RV - theta['xi'] - theta['phi']*var-theta['tau1']*eps-theta['tau2']*(np.square(eps)-1)
        llik_y = np.sum(stats.norm.logpdf(self.Y[1:], loc=0, scale=np.sqrt(var[1:])), axis=0)  
        return llik_y
    
    def reweight_particles(self):
        # calculate new epn
        ESSmin = self.ESSrmin_ * self.X.N
        f = lambda e: rs.essl(e * self.X.llik) - ESSmin
        epn = self.X.shared['exponents'][-1]
        if f(1. - epn) > 0:  # we're done (last iteration)
            delta = 1. - epn
            new_epn = 1. # set 1. manually so that we can safely test == 1.
        else:
            delta = sp.optimize.brentq(f, 1.e-12, 1. - epn)  # secant search
            # left endpoint is >0, since f(0.) = nan if any likelihood = -inf
            new_epn = epn + delta
        self.X.shared['exponents'].append(new_epn)
        # calculate delta llik
        dllik = delta * self.X.llik
        self.X.lpost += dllik
        #update weights
        self.wgts = self.wgts.add(dllik)
        self.wgts_ = rs.Weights(delta*self.loglik_(self.X.theta))
        
    def __next__(self):
        if self.done():
            self.loglik(self.X.theta, get_v=True)
            self.var_ls = np.array(self.var_ls)
            self.w_ls = np.array(self.w_ls)
            raise StopIteration
        if self.t == 0:
            self.generate_particles()
        else:
            self.resample_move()
        self.reweight_particles()
        self.wgts_ls.append(self.wgts_)
        self.X_ls.append(self.X)
        if self.verbose:
            print("t={}, accept_rate={:.2f}, accept_rate2={:.2f}, epn={:.3f}".format(self.t, np.average(self.X.shared['acc_rates'][-1]),self.X.shared['acc_rates2'][-1],self.X.shared['exponents'][-1]))
        self.t += 1
        
class RealRECHD_wm(smc.SMCD):  
    def __init__(self,data=None, **kwargs):
        super().__init__(**kwargs)
        self.Y_train, self.Y_test, self.RV_train, self.RV_test, self.RV_week_train, self.RV_week_test, self.RV_month_train, self.RV_month_test = data
        self.T = self.Y_test.shape[0]
        
    def loglik(self, theta, t=None, lpyt=False): 
        Y = np.concatenate((self.Y_train, self.Y_test[:t+1]))
        RV = np.concatenate((self.RV_train, self.RV_test[:t+1]))
        RV_week = np.concatenate((self.RV_week_train, self.RV_week_test[:t+1]))
        RV_month = np.concatenate((self.RV_month_train, self.RV_month_test[:t+1]))
        N = Y.shape[0]
        var    = np.zeros((N, theta.shape[0]))
        var[0] = np.var(Y)
        omega = np.zeros((N, theta.shape[0]))
        omega[0] = theta['beta0'] # + theta['beta1'] * h0 which is 0 
        h = np.zeros((N, theta.shape[0]))
        c = np.zeros((N, theta.shape[0]))# h0 = 0
        for n in range(N-1):  #yt generate std t+1, while we don't need t+1
            gf = ut.sigmoid(theta['v0f']*omega[n]+theta['v1f']*Y[n]+theta['v2f']*var[n]+theta['v3f']*RV[n]+theta['v4f']*RV_week[n]+theta['v5f']*RV_month[n]+theta['wf']*h[n]+theta['bf'])
            gi = ut.sigmoid(theta['v0i']*omega[n]+theta['v1i']*Y[n]+theta['v2i']*var[n]+theta['v3i']*RV[n]+theta['v4i']*RV_week[n]+theta['v5i']*RV_month[n]+theta['wi']*h[n]+theta['bi'])
            go = ut.sigmoid(theta['v0o']*omega[n]+theta['v1o']*Y[n]+theta['v2o']*var[n]+theta['v3o']*RV[n]+theta['v4o']*RV_week[n]+theta['v5o']*RV_month[n]+theta['wo']*h[n]+theta['bo'])         
            chat = np.tanh(theta['v0d']*omega[n]+theta['v1d']*Y[n]+theta['v2d']*var[n]+theta['v3d']*RV[n]+theta['v4d']*RV_week[n]+theta['v5d']*RV_month[n]+theta['wd']*h[n]+theta['bd'])         
            c[n+1] = gi*chat+gf*c[n]
            h[n+1] = go*np.tanh(c[n+1])
            omega[n+1] = theta['beta0']+theta['beta1']*h[n+1]
            var[n+1] = omega[n+1]+theta['beta']*var[n]+theta['gamma']*RV[n]  
        eps = Y/np.sqrt(var)
        U   = RV - theta['xi']-theta['phi']*var-theta['tau1']*eps-theta['tau2']*(np.square(eps)-1)
        if lpyt:
            llik_y = stats.norm.logpdf(Y[-1], loc=0, scale=np.sqrt(var[-1]))
            llik_rv = stats.norm.logpdf(U[-1], loc=0, scale=np.sqrt(theta['sigmau2']))
            v = np.average(var[-1]) if self.rs_flag else np.average(var[-1],weights=self.wgts.W) 
            w = np.average(omega[-1]) if self.rs_flag else np.average(omega[-1],weights=self.wgts.W) 
            self.var_ls.append(v) 
            self.w_ls.append(w)       
        else:
            llik_y = np.sum(stats.norm.logpdf(Y, loc=0, scale=np.sqrt(var)), axis=0)  
            llik_rv = np.sum(stats.norm.logpdf(U, loc=0, scale=np.sqrt(theta['sigmau2'])), axis=0)  
        return llik_y+llik_rv    

##########s######################################################### 
# RealRECH 
class RealRECHsim(smc.SMC):  
    def __init__(self,data=None, **kwargs):
        super().__init__(**kwargs)
        self.Y, self.RV = data
        self.T = self.Y.shape[0]
        self.wgts_ = rs.Weights()
    
    def loglik(self, theta, get_v=False, scale=True):  
        N = self.Y.shape[0]
        var    = np.zeros((N, theta.shape[0]))
        var[0] = np.var(self.Y)
        omega = np.zeros((N, theta.shape[0]))
        omega[0] = theta['beta0'] # + theta['beta1'] * h0 which is 0 
        h = np.zeros((N, theta.shape[0]))
        c = np.zeros((N, theta.shape[0]))# h0 = 0
        
        for n in range(N-1):  #yt generate std t+1, while we don't need t+1
            gf = ut.sigmoid(theta['v0f']*omega[n]+theta['v1f']*self.Y[n]+theta['v2f']*var[n]+theta['v3f']*self.RV[n]+theta['wf']*h[n]+theta['bf'])
            gi = ut.sigmoid(theta['v0i']*omega[n]+theta['v1i']*self.Y[n]+theta['v2i']*var[n]+theta['v3i']*self.RV[n]+theta['wi']*h[n]+theta['bi'])
            go = ut.sigmoid(theta['v0o']*omega[n]+theta['v1o']*self.Y[n]+theta['v2o']*var[n]+theta['v3o']*self.RV[n]+theta['wo']*h[n]+theta['bo'])         
            chat =  np.tanh(theta['v0d']*omega[n]+theta['v1d']*self.Y[n]+theta['v2d']*var[n]+theta['v3d']*self.RV[n]+theta['wd']*h[n]+theta['bd'])         
            c[n+1] = gi*chat+gf*c[n]
            h[n+1] = go*np.tanh(c[n+1])
            omega[n+1] = theta['beta0']+theta['beta1']*h[n+1]
            var[n+1] = omega[n+1]+theta['beta']*var[n]+theta['gamma']*self.RV[n]
        U   = self.RV - theta['xi'] - theta['phi']*var
        llik_y = np.sum(stats.norm.logpdf(self.Y[1:], loc=0, scale=np.sqrt(var[1:])), axis=0)  
        llik_rv = np.sum(stats.norm.logpdf(U[1:], loc=0, scale=np.sqrt(theta['sigmau2'])), axis=0)
        if get_v:
            self.var_ls = np.average(var[1:], axis=1, weights=self.wgts.W)
            self.w_ls = list(np.average(omega[1:], axis=1, weights=self.wgts.W))
        return llik_y+llik_rv #(N,) array
    
    def loglik_(self, theta, get_v=False, scale=True):  
        N = self.Y.shape[0]
        var    = np.zeros((N, theta.shape[0]))
        var[0] = np.var(self.Y)
        omega = np.zeros((N, theta.shape[0]))
        omega[0] = theta['beta0'] # + theta['beta1'] * h0 which is 0 
        h = np.zeros((N, theta.shape[0]))
        c = np.zeros((N, theta.shape[0]))# h0 = 0
        
        for n in range(N-1):  #yt generate std t+1, while we don't need t+1
            gf = ut.sigmoid(theta['v0f']*omega[n]+theta['v1f']*self.Y[n]+theta['v2f']*var[n]+theta['v3f']*self.RV[n]+theta['wf']*h[n]+theta['bf'])
            gi = ut.sigmoid(theta['v0i']*omega[n]+theta['v1i']*self.Y[n]+theta['v2i']*var[n]+theta['v3i']*self.RV[n]+theta['wi']*h[n]+theta['bi'])
            go = ut.sigmoid(theta['v0o']*omega[n]+theta['v1o']*self.Y[n]+theta['v2o']*var[n]+theta['v3o']*self.RV[n]+theta['wo']*h[n]+theta['bo'])         
            chat = np.tanh(theta['v0d']*omega[n]+theta['v1d']*self.Y[n]+theta['v2d']*var[n]+theta['v3d']*self.RV[n]+theta['wd']*h[n]+theta['bd'])         
            c[n+1] = gi*chat+gf*c[n]
            h[n+1] = go*np.tanh(c[n+1])
            omega[n+1] = theta['beta0']+theta['beta1']*h[n+1]
            var[n+1] = omega[n+1]+theta['beta']*var[n]+theta['gamma']*self.RV[n]
        llik_y = np.sum(stats.norm.logpdf(self.Y[1:], loc=0, scale=np.sqrt(var[1:])), axis=0)  
        return llik_y
    
    def reweight_particles(self):
        # calculate new epn
        ESSmin = self.ESSrmin_ * self.X.N
        f = lambda e: rs.essl(e * self.X.llik) - ESSmin
        epn = self.X.shared['exponents'][-1]
        if f(1. - epn) > 0:  # we're done (last iteration)
            delta = 1. - epn
            new_epn = 1. # set 1. manually so that we can safely test == 1.
        else:
            delta = sp.optimize.brentq(f, 1.e-12, 1. - epn)  # secant search
            # left endpoint is >0, since f(0.) = nan if any likelihood = -inf
            new_epn = epn + delta
        self.X.shared['exponents'].append(new_epn)
        # calculate delta llik
        dllik = delta * self.X.llik
        self.X.lpost += dllik
        #update weights
        self.wgts = self.wgts.add(dllik)
        self.wgts_ = rs.Weights(delta*self.loglik_(self.X.theta))
        
    def __next__(self):
        if self.done():
            self.loglik(self.X.theta, get_v=True)
            self.var_ls = np.array(self.var_ls)
            self.w_ls = np.array(self.w_ls)
            raise StopIteration
        if self.t == 0:
            self.generate_particles()
        else:
            self.resample_move()
        self.reweight_particles()
        self.wgts_ls.append(self.wgts_)
        self.X_ls.append(self.X)
        if self.verbose:
            print("t={}, accept_rate={:.2f}, accept_rate2={:.2f}, epn={:.3f}".format(self.t, np.average(self.X.shared['acc_rates'][-1]),self.X.shared['acc_rates2'][-1],self.X.shared['exponents'][-1]))
        self.t += 1
class RealRECHDsim(smc.SMCD):  
    def __init__(self,data=None, **kwargs):
        super().__init__(**kwargs)
        self.Y_train, self.Y_test, self.RV_train, self.RV_test = data
        self.T = self.Y_test.shape[0]
        
    def loglik(self, theta, t=None, lpyt=False): 
        Y = np.concatenate((self.Y_train, self.Y_test[:t+1]))
        RV= np.concatenate((self.RV_train, self.RV_test[:t+1]))
        N = Y.shape[0]
        var    = np.zeros((N, theta.shape[0]))
        var[0] = np.var(Y)
        omega = np.zeros((N, theta.shape[0]))
        omega[0] = theta['beta0'] # + theta['beta1'] * h0 which is 0 
        h = np.zeros((N, theta.shape[0]))
        c = np.zeros((N, theta.shape[0]))# h0 = 0
        for n in range(N-1):  #yt generate std t+1, while we don't need t+1
            gf = ut.sigmoid(theta['v0f']*omega[n]+theta['v1f']*Y[n]+theta['v2f']*var[n]+theta['v3f']*RV[n]+theta['wf']*h[n]+theta['bf'])
            gi = ut.sigmoid(theta['v0i']*omega[n]+theta['v1i']*Y[n]+theta['v2i']*var[n]+theta['v3i']*RV[n]+theta['wi']*h[n]+theta['bi'])
            go = ut.sigmoid(theta['v0o']*omega[n]+theta['v1o']*Y[n]+theta['v2o']*var[n]+theta['v3o']*RV[n]+theta['wo']*h[n]+theta['bo'])         
            chat = np.tanh(theta['v0d']*omega[n]+theta['v1d']*Y[n]+theta['v2d']*var[n]+theta['v3d']*RV[n]+theta['wd']*h[n]+theta['bd'])         
            c[n+1] = gi*chat+gf*c[n]
            h[n+1] = go*np.tanh(c[n+1])
            omega[n+1] = theta['beta0']+theta['beta1']*h[n+1]
            var[n+1] = omega[n+1]+theta['beta']*var[n]+theta['gamma']*RV[n]  
        U   = RV - theta['xi']-theta['phi']*var
        if lpyt:
            llik_y = stats.norm.logpdf(Y[-1], loc=0, scale=np.sqrt(var[-1]))
            llik_rv = stats.norm.logpdf(U[-1], loc=0, scale=np.sqrt(theta['sigmau2']))
            v = np.average(var[-1]) if self.rs_flag else np.average(var[-1],weights=self.wgts.W) 
            w = np.average(omega[-1]) if self.rs_flag else np.average(omega[-1],weights=self.wgts.W) 
            self.var_ls.append(v) 
            self.w_ls.append(w)       
        else:
            llik_y = np.sum(stats.norm.logpdf(Y, loc=0, scale=np.sqrt(var)), axis=0)  
            llik_rv = np.sum(stats.norm.logpdf(U, loc=0, scale=np.sqrt(theta['sigmau2'])), axis=0)  
        return llik_y+llik_rv   


##########s######################################################### 
# RealRECH 
class RealRECH_norv(smc.SMC):  
    def __init__(self,data=None, **kwargs):
        super().__init__(**kwargs)
        self.Y, self.RV = data
        self.T = self.Y.shape[0]
        self.wgts_ = rs.Weights()
    
    def loglik(self, theta, get_v=False):  
        N = self.Y.shape[0]
        var    = np.zeros((N, theta.shape[0]))
        var[0] = np.var(self.Y)
        omega = np.zeros((N, theta.shape[0]))
        omega[0] = theta['beta0'] # + theta['beta1'] * h0 which is 0 
        h = np.zeros((N, theta.shape[0]))
        c = np.zeros((N, theta.shape[0]))# h0 = 0
        
        for n in range(N-1):  #yt generate std t+1, while we don't need t+1
            gf = ut.sigmoid(theta['v0f']*omega[n]+theta['v1f']*self.Y[n]+theta['v2f']*var[n]+theta['v3f']*self.RV[n]+theta['wf']*h[n]+theta['bf'])
            gi = ut.sigmoid(theta['v0i']*omega[n]+theta['v1i']*self.Y[n]+theta['v2i']*var[n]+theta['v3i']*self.RV[n]+theta['wi']*h[n]+theta['bi'])
            go = ut.sigmoid(theta['v0o']*omega[n]+theta['v1o']*self.Y[n]+theta['v2o']*var[n]+theta['v3o']*self.RV[n]+theta['wo']*h[n]+theta['bo'])         
            chat =  np.tanh(theta['v0d']*omega[n]+theta['v1d']*self.Y[n]+theta['v2d']*var[n]+theta['v3d']*self.RV[n]+theta['wd']*h[n]+theta['bd'])         
            c[n+1] = gi*chat+gf*c[n]
            h[n+1] = go*np.tanh(c[n+1])
            omega[n+1] = theta['beta0']+theta['beta1']*h[n+1]
            var[n+1] = omega[n+1]+theta['beta']*var[n]+theta['gamma']*self.RV[n]
        eps = self.Y/np.sqrt(var)
        U   = self.RV - theta['xi'] - theta['phi']*var-theta['tau1']*eps-theta['tau2']*(np.square(eps)-1)
        llik_y = np.sum(stats.norm.logpdf(self.Y[1:], loc=0, scale=np.sqrt(var[1:])), axis=0)  
        llik_rv = np.sum(stats.norm.logpdf(U[1:], loc=0, scale=np.sqrt(theta['sigmau2'])), axis=0)
        if get_v:
            self.var_ls = np.average(var[1:], axis=1, weights=self.wgts.W)
            self.w_ls = list(np.average(omega[1:], axis=1, weights=self.wgts.W))
        return llik_y #(N,) array
    
    def loglik_(self, theta, get_v=False):  
        N = self.Y.shape[0]
        var    = np.zeros((N, theta.shape[0]))
        var[0] = np.var(self.Y)
        omega = np.zeros((N, theta.shape[0]))
        omega[0] = theta['beta0'] # + theta['beta1'] * h0 which is 0 
        h = np.zeros((N, theta.shape[0]))
        c = np.zeros((N, theta.shape[0]))# h0 = 0
        
        for n in range(N-1):  #yt generate std t+1, while we don't need t+1
            gf = ut.sigmoid(theta['v0f']*omega[n]+theta['v1f']*self.Y[n]+theta['v2f']*var[n]+theta['v3f']*self.RV[n]+theta['wf']*h[n]+theta['bf'])
            gi = ut.sigmoid(theta['v0i']*omega[n]+theta['v1i']*self.Y[n]+theta['v2i']*var[n]+theta['v3i']*self.RV[n]+theta['wi']*h[n]+theta['bi'])
            go = ut.sigmoid(theta['v0o']*omega[n]+theta['v1o']*self.Y[n]+theta['v2o']*var[n]+theta['v3o']*self.RV[n]+theta['wo']*h[n]+theta['bo'])         
            chat = np.tanh(theta['v0d']*omega[n]+theta['v1d']*self.Y[n]+theta['v2d']*var[n]+theta['v3d']*self.RV[n]+theta['wd']*h[n]+theta['bd'])         
            c[n+1] = gi*chat+gf*c[n]
            h[n+1] = go*np.tanh(c[n+1])
            omega[n+1] = theta['beta0']+theta['beta1']*h[n+1]
            var[n+1] = omega[n+1]+theta['beta']*var[n]+theta['gamma']*self.RV[n]
        eps = self.Y/np.sqrt(var)
        U   = self.RV - theta['xi'] - theta['phi']*var-theta['tau1']*eps-theta['tau2']*(np.square(eps)-1)
        llik_y = np.sum(stats.norm.logpdf(self.Y[1:], loc=0, scale=np.sqrt(var[1:])), axis=0)  
        return llik_y
    
    def reweight_particles(self):
        # calculate new epn
        ESSmin = self.ESSrmin_ * self.X.N
        f = lambda e: rs.essl(e * self.X.llik) - ESSmin
        epn = self.X.shared['exponents'][-1]
        if f(1. - epn) > 0:  # we're done (last iteration)
            delta = 1. - epn
            new_epn = 1. # set 1. manually so that we can safely test == 1.
        else:
            delta = sp.optimize.brentq(f, 1.e-12, 1. - epn)  # secant search
            # left endpoint is >0, since f(0.) = nan if any likelihood = -inf
            new_epn = epn + delta
        self.X.shared['exponents'].append(new_epn)
        # calculate delta llik
        dllik = delta * self.X.llik
        self.X.lpost += dllik
        #update weights
        self.wgts = self.wgts.add(dllik)
        self.wgts_ = rs.Weights(delta*self.loglik_(self.X.theta))
        
    def __next__(self):
        if self.done():
            self.loglik(self.X.theta, get_v=True)
            self.var_ls = np.array(self.var_ls)
            self.w_ls = np.array(self.w_ls)
            raise StopIteration
        if self.t == 0:
            self.generate_particles()
        else:
            self.resample_move()
        self.reweight_particles()
        self.wgts_ls.append(self.wgts_)
        self.X_ls.append(self.X)
        if self.verbose:
            print("t={}, accept_rate={:.2f}, accept_rate2={:.2f}, epn={:.3f}".format(self.t, np.average(self.X.shared['acc_rates'][-1]),self.X.shared['acc_rates2'][-1],self.X.shared['exponents'][-1]))
        self.t += 1
class RealRECHD_norv(smc.SMCD):  
    def __init__(self,data=None, **kwargs):
        super().__init__(**kwargs)
        self.Y_train, self.Y_test, self.RV_train, self.RV_test = data
        self.T = self.Y_test.shape[0]
        
    def loglik(self, theta, t=None, lpyt=False): 
        Y = np.concatenate((self.Y_train, self.Y_test[:t+1]))
        RV= np.concatenate((self.RV_train, self.RV_test[:t+1]))
        N = Y.shape[0]
        var    = np.zeros((N, theta.shape[0]))
        var[0] = np.var(Y)
        omega = np.zeros((N, theta.shape[0]))
        omega[0] = theta['beta0'] # + theta['beta1'] * h0 which is 0 
        h = np.zeros((N, theta.shape[0]))
        c = np.zeros((N, theta.shape[0]))# h0 = 0
        for n in range(N-1):  #yt generate std t+1, while we don't need t+1
            gf = ut.sigmoid(theta['v0f']*omega[n]+theta['v1f']*Y[n]+theta['v2f']*var[n]+theta['v3f']*RV[n]+theta['wf']*h[n]+theta['bf'])
            gi = ut.sigmoid(theta['v0i']*omega[n]+theta['v1i']*Y[n]+theta['v2i']*var[n]+theta['v3i']*RV[n]+theta['wi']*h[n]+theta['bi'])
            go = ut.sigmoid(theta['v0o']*omega[n]+theta['v1o']*Y[n]+theta['v2o']*var[n]+theta['v3o']*RV[n]+theta['wo']*h[n]+theta['bo'])         
            chat = np.tanh(theta['v0d']*omega[n]+theta['v1d']*Y[n]+theta['v2d']*var[n]+theta['v3d']*RV[n]+theta['wd']*h[n]+theta['bd'])         
            c[n+1] = gi*chat+gf*c[n]
            h[n+1] = go*np.tanh(c[n+1])
            omega[n+1] = theta['beta0']+theta['beta1']*h[n+1]
            var[n+1] = omega[n+1]+theta['beta']*var[n]+theta['gamma']*RV[n]  
        eps = Y/np.sqrt(var)
        U   = RV - theta['xi']-theta['phi']*var-theta['tau1']*eps-theta['tau2']*(np.square(eps)-1)
        if lpyt:
            llik_y = stats.norm.logpdf(Y[-1], loc=0, scale=np.sqrt(var[-1]))
            llik_rv = stats.norm.logpdf(U[-1], loc=0, scale=np.sqrt(theta['sigmau2']))
            v = np.average(var[-1]) if self.rs_flag else np.average(var[-1],weights=self.wgts.W) 
            w = np.average(omega[-1]) if self.rs_flag else np.average(omega[-1],weights=self.wgts.W) 
            self.var_ls.append(v) 
            self.w_ls.append(w)       
        else:
            llik_y = np.sum(stats.norm.logpdf(Y, loc=0, scale=np.sqrt(var)), axis=0)  
            llik_rv = np.sum(stats.norm.logpdf(U, loc=0, scale=np.sqrt(theta['sigmau2'])), axis=0)  
        return llik_y     
    

class Res:
    def __init__(self):
        self.var = []
class RealRECHD_MultiPeriod(smc.SMCD):  
    def __init__(self,data=None, **kwargs):
        super().__init__(**kwargs)
        self.Y_train, self.Y_test, self.RV_train, self.RV_test = data
        self.T = self.Y_test.shape[0]
        self.Res = Res()
    # Add this to both classes in the model.py file
    def calculate_variance(self, omega, beta, gamma, rv, var_prev):
        """Calculate variance with safeguards against negative values"""
        var_next = omega + beta * var_prev + gamma * rv
        # Ensure variance is always positive
        return np.maximum(1e-10, var_next)    
    def loglik(self, theta, t=None, lpyt=False): 
        Y = np.concatenate((self.Y_train, self.Y_test[:t+1]))
        RV= np.concatenate((self.RV_train, self.RV_test[:t+1]))
        N = Y.shape[0]
        var    = np.zeros((N, theta.shape[0]))
        var[0] = np.var(Y)
        omega = np.zeros((N, theta.shape[0]))
        omega[0] = theta['beta0'] # + theta['beta1'] * h0 which is 0 
        h = np.zeros((N, theta.shape[0]))
        c = np.zeros((N, theta.shape[0]))# h0 = 0
        for n in range(N-1):  #yt generate std t+1, while we don't need t+1
            gf = ut.sigmoid(theta['v0f']*omega[n]+theta['v1f']*Y[n]+theta['v2f']*var[n]+theta['v3f']*RV[n]+theta['wf']*h[n]+theta['bf'])
            gi = ut.sigmoid(theta['v0i']*omega[n]+theta['v1i']*Y[n]+theta['v2i']*var[n]+theta['v3i']*RV[n]+theta['wi']*h[n]+theta['bi'])
            go = ut.sigmoid(theta['v0o']*omega[n]+theta['v1o']*Y[n]+theta['v2o']*var[n]+theta['v3o']*RV[n]+theta['wo']*h[n]+theta['bo'])         
            chat = np.tanh(theta['v0d']*omega[n]+theta['v1d']*Y[n]+theta['v2d']*var[n]+theta['v3d']*RV[n]+theta['wd']*h[n]+theta['bd'])         
            c[n+1] = gi*chat+gf*c[n]
            h[n+1] = go*np.tanh(c[n+1])
            omega[n+1] = theta['beta0']+theta['beta1']*h[n+1]
            #var[n+1] = omega[n+1]+theta['beta']*var[n]+theta['gamma']*RV[n]
            var[n+1] = self.calculate_variance(omega[n+1], theta['beta'], theta['gamma'], self.RV[n], var[n])  
        eps = Y/np.sqrt(var)
        U   = RV - theta['xi']-theta['phi']*var-theta['tau1']*eps-theta['tau2']*(np.square(eps)-1)
        if lpyt:
            llik_y = stats.norm.logpdf(Y[-1], loc=0, scale=np.sqrt(var[-1]))
            llik_rv = stats.norm.logpdf(U[-1], loc=0, scale=np.sqrt(theta['sigmau2']))
            weights = None if self.rs_flag else self.wgts.W
            v = np.average(var[-1], weights=weights) 
            w = np.average(omega[-1], weights=weights) 
            self.var_ls.append(v) 
            self.w_ls.append(w)   


            # simulate n steps forecast
            # init at t 
            mc_size = 10000
            n_steps = 10
            theta = ut.as_dict(theta)


            # weights = weights = None if self.rs_flag else self.wgts.W.reshape((-1,1))
            # var = var[-1].reshape((-1,1)) # var = (n,1)
            # omega = omega[-1].reshape((-1,1))
            # h = h[-1].reshape((-1,1))
            # c = c[-1].reshape((-1,1))
            # var_ls = [np.average(var, weights=weights)]

            # for i in range(n_steps):
            #     var[var<=0] = 0.0001
            #     y = stats.norm.rvs(scale=var, size=(self.N,mc_size)) # y = (n,mc_size)
            #     u = stats.norm.rvs(size=mc_size).reshape((1,-1))
            #     eps = y/var
            #     rv = theta['xi'] + theta['phi']*var + theta['tau1']*eps + theta['tau2']*(np.square(eps)-1) + u # rv = (n,mc_size)

            #     # cal
            #     gf = ut.sigmoid(theta['v0f']*omega+theta['v1f']*y+theta['v2f']*var+theta['v3f']*rv+theta['wf']*h+theta['bf']) 
            #     gi = ut.sigmoid(theta['v0i']*omega+theta['v1i']*y+theta['v2i']*var+theta['v3i']*rv+theta['wi']*h+theta['bi'])
            #     go = ut.sigmoid(theta['v0o']*omega+theta['v1o']*y+theta['v2o']*var+theta['v3o']*rv+theta['wo']*h+theta['bo'])         
            #     chat = np.tanh(theta['v0d']*omega+theta['v1d']*y+theta['v2d']*var+theta['v3d']*rv+theta['wd']*h+theta['bd'])         
            #     c = gi*chat+gf*c
            #     h = go*np.tanh(c)
            #     omega = theta['beta0']+theta['beta1']*h
            #     var = omega+theta['beta']*var+theta['gamma']*rv # var = (n,mc_size)

            #     # save
            #     var_ls.append(np.average(var.mean(axis=1), weights=weights)) # float

            # self.Res.var.append(var_ls)

            try:
                weights = weights = None if self.rs_flag else self.wgts.W.reshape((-1,1))
                var = var[-1].reshape((-1,1)) # var = (n,1)
                omega = omega[-1].reshape((-1,1))
                h = h[-1].reshape((-1,1))
                c = c[-1].reshape((-1,1))
                var_ls = [np.average(var, weights=weights)]

                for i in range(n_steps):
                    var[var<=0] = 0.0001
                    y = stats.norm.rvs(scale=var, size=(self.N,mc_size)) # y = (n,mc_size)
                    u = stats.norm.rvs(size=mc_size).reshape((1,-1))
                    eps = y/var
                    rv = theta['xi'] + theta['phi']*var + theta['tau1']*eps + theta['tau2']*(np.square(eps)-1) + u # rv = (n,mc_size)

                    # cal
                    gf = ut.sigmoid(theta['v0f']*omega+theta['v1f']*y+theta['v2f']*var+theta['v3f']*rv+theta['wf']*h+theta['bf']) 
                    gi = ut.sigmoid(theta['v0i']*omega+theta['v1i']*y+theta['v2i']*var+theta['v3i']*rv+theta['wi']*h+theta['bi'])
                    go = ut.sigmoid(theta['v0o']*omega+theta['v1o']*y+theta['v2o']*var+theta['v3o']*rv+theta['wo']*h+theta['bo'])         
                    chat = np.tanh(theta['v0d']*omega+theta['v1d']*y+theta['v2d']*var+theta['v3d']*rv+theta['wd']*h+theta['bd'])         
                    c = gi*chat+gf*c
                    h = go*np.tanh(c)
                    omega = theta['beta0']+theta['beta1']*h
                    var = omega+theta['beta']*var+theta['gamma']*rv # var = (n,mc_size)

                    # save
                    var_ls.append(np.average(var.mean(axis=1, keepdims=True), weights=weights)) # float

                self.Res.var.append(var_ls)
            except:
                import pdb; pdb.set_trace()



        else:
            llik_y = np.sum(stats.norm.logpdf(Y, loc=0, scale=np.sqrt(var)), axis=0)  
            llik_rv = np.sum(stats.norm.logpdf(U, loc=0, scale=np.sqrt(theta['sigmau2'])), axis=0)  
        return llik_y+llik_rv      

################################################################### 
# RealRECH + 1x
class RealRECH_1x(smc.SMC):  
    def __init__(self,data=None, **kwargs):
        super().__init__(**kwargs)
        self.Y, self.RV, self.X1 = data
        self.T = self.Y.shape[0]
        self.wgts_ = rs.Weights()
    
    def loglik(self, theta, get_v=False):  
        N = self.Y.shape[0]
        var    = np.zeros((N, theta.shape[0]))
        var[0] = np.var(self.Y)
        omega = np.zeros((N, theta.shape[0]))
        omega[0] = theta['beta0'] # + theta['beta1'] * h0 which is 0 
        h = np.zeros((N, theta.shape[0]))
        c = np.zeros((N, theta.shape[0]))# h0 = 0
        
        for n in range(N-1):  #yt generate std t+1, while we don't need t+1
            gf = ut.sigmoid(theta['v0f']*omega[n]+theta['v1f']*self.Y[n]+theta['v2f']*var[n]+theta['v3f']*self.RV[n]+theta['wf']*h[n]+theta['bf']+theta['v4f']*self.X1[n]) 
            gi = ut.sigmoid(theta['v0i']*omega[n]+theta['v1i']*self.Y[n]+theta['v2i']*var[n]+theta['v3i']*self.RV[n]+theta['wi']*h[n]+theta['bi']+theta['v4i']*self.X1[n]) 
            go = ut.sigmoid(theta['v0o']*omega[n]+theta['v1o']*self.Y[n]+theta['v2o']*var[n]+theta['v3o']*self.RV[n]+theta['wo']*h[n]+theta['bo']+theta['v4o']*self.X1[n])      
            chat = np.tanh(theta['v0d']*omega[n]+theta['v1d']*self.Y[n]+theta['v2d']*var[n]+theta['v3d']*self.RV[n]+theta['wd']*h[n]+theta['bd']+theta['v4d']*self.X1[n])    
            
            c[n+1] = gi*chat+gf*c[n]
            h[n+1] = go*np.tanh(c[n+1])
            omega[n+1] = theta['beta0']+theta['beta1']*h[n+1]
            # var[n+1] = omega[n+1]+theta['beta']*var[n]+theta['gamma']*self.RV[n]
            var[n+1] = self.calculate_variance(omega[n+1], theta['beta'], theta['gamma'], self.RV[n], var[n])
        eps = self.Y/np.sqrt(var)
        U   = self.RV - theta['xi'] - theta['phi']*var-theta['tau1']*eps-theta['tau2']*(np.square(eps)-1)
        llik_y = np.sum(stats.norm.logpdf(self.Y[1:], loc=0, scale=np.sqrt(var[1:])), axis=0)  
        llik_rv = np.sum(stats.norm.logpdf(U[1:], loc=0, scale=np.sqrt(theta['sigmau2'])), axis=0)
        if get_v:
            self.var_ls = np.average(var[1:], axis=1, weights=self.wgts.W)
            self.w_ls = list(np.average(omega[1:], axis=1, weights=self.wgts.W))
        return llik_y+llik_rv #(N,) array
    
    def loglik_(self, theta, get_v=False):  
        N = self.Y.shape[0]
        var    = np.zeros((N, theta.shape[0]))
        var[0] = np.var(self.Y)
        omega = np.zeros((N, theta.shape[0]))
        omega[0] = theta['beta0'] # + theta['beta1'] * h0 which is 0 
        h = np.zeros((N, theta.shape[0]))
        c = np.zeros((N, theta.shape[0]))# h0 = 0
        
        for n in range(N-1):  #yt generate std t+1, while we don't need t+1
            gf = ut.sigmoid(theta['v0f']*omega[n]+theta['v1f']*self.Y[n]+theta['v2f']*var[n]+theta['v3f']*self.RV[n]+theta['wf']*h[n]+theta['bf']+theta['v4f']*self.X1[n]) 
            gi = ut.sigmoid(theta['v0i']*omega[n]+theta['v1i']*self.Y[n]+theta['v2i']*var[n]+theta['v3i']*self.RV[n]+theta['wi']*h[n]+theta['bi']+theta['v4i']*self.X1[n]) 
            go = ut.sigmoid(theta['v0o']*omega[n]+theta['v1o']*self.Y[n]+theta['v2o']*var[n]+theta['v3o']*self.RV[n]+theta['wo']*h[n]+theta['bo']+theta['v4o']*self.X1[n])  
            chat = np.tanh(theta['v0d']*omega[n]+theta['v1d']*self.Y[n]+theta['v2d']*var[n]+theta['v3d']*self.RV[n]+theta['wd']*h[n]+theta['bd']+theta['v4d']*self.X1[n])     
            c[n+1] = gi*chat+gf*c[n]
            h[n+1] = go*np.tanh(c[n+1])
            omega[n+1] = theta['beta0']+theta['beta1']*h[n+1]
            var[n+1] = omega[n+1]+theta['beta']*var[n]+theta['gamma']*self.RV[n]
        eps = self.Y/np.sqrt(var)
        U   = self.RV - theta['xi'] - theta['phi']*var-theta['tau1']*eps-theta['tau2']*(np.square(eps)-1)
        llik_y = np.sum(stats.norm.logpdf(self.Y[1:], loc=0, scale=np.sqrt(var[1:])), axis=0)  
        return llik_y
    
    def reweight_particles(self):
        # calculate new epn
        ESSmin = self.ESSrmin_ * self.X.N
        f = lambda e: rs.essl(e * self.X.llik) - ESSmin
        epn = self.X.shared['exponents'][-1]
        if f(1. - epn) > 0:  # we're done (last iteration)
            delta = 1. - epn
            new_epn = 1. # set 1. manually so that we can safely test == 1.
        else:
            delta = sp.optimize.brentq(f, 1.e-12, 1. - epn)  # secant search
            # left endpoint is >0, since f(0.) = nan if any likelihood = -inf
            new_epn = epn + delta
        self.X.shared['exponents'].append(new_epn)
        # calculate delta llik
        dllik = delta * self.X.llik
        self.X.lpost += dllik
        #update weights
        self.wgts = self.wgts.add(dllik)
        self.wgts_ = rs.Weights(delta*self.loglik_(self.X.theta))
        
    def __next__(self):
        if self.done():
            self.loglik(self.X.theta, get_v=True)
            self.var_ls = np.array(self.var_ls)
            self.w_ls = np.array(self.w_ls)
            raise StopIteration
        if self.t == 0:
            self.generate_particles()
        else:
            self.resample_move()
        self.reweight_particles()
        self.wgts_ls.append(self.wgts_)
        self.X_ls.append(self.X)
        if self.verbose:
            print("t={}, accept_rate={:.2f}, accept_rate2={:.2f}, epn={:.3f}".format(self.t, np.average(self.X.shared['acc_rates'][-1]),self.X.shared['acc_rates2'][-1],self.X.shared['exponents'][-1]))
        self.t += 1
class RealRECHD_1x(smc.SMCD):  
    def __init__(self,data=None, **kwargs):
        super().__init__(**kwargs)
        self.Y_train, self.Y_test, self.RV_train, self.RV_test, self.X1_train, self.X1_test = data
        self.T = self.Y_test.shape[0]
        
    def loglik(self, theta, t=None, lpyt=False): 
        Y  = np.concatenate((self.Y_train, self.Y_test[:t+1]))
        RV = np.concatenate((self.RV_train, self.RV_test[:t+1]))
        X1 = np.concatenate((self.X1_train, self.X1_test[:t+1]))
        N = Y.shape[0]
        var    = np.zeros((N, theta.shape[0]))
        var[0] = np.var(Y)
        omega = np.zeros((N, theta.shape[0]))
        omega[0] = theta['beta0'] # + theta['beta1'] * h0 which is 0 
        h = np.zeros((N, theta.shape[0]))
        c = np.zeros((N, theta.shape[0]))# h0 = 0
        for n in range(N-1):  #yt generate std t+1, while we don't need t+1
            gf = ut.sigmoid(theta['v0f']*omega[n]+theta['v1f']*Y[n]+theta['v2f']*var[n]+theta['v3f']*RV[n]+theta['wf']*h[n]+theta['bf']+theta['v4f']*X1[n]) 
            gi = ut.sigmoid(theta['v0i']*omega[n]+theta['v1i']*Y[n]+theta['v2i']*var[n]+theta['v3i']*RV[n]+theta['wi']*h[n]+theta['bi']+theta['v4i']*X1[n]) 
            go = ut.sigmoid(theta['v0o']*omega[n]+theta['v1o']*Y[n]+theta['v2o']*var[n]+theta['v3o']*RV[n]+theta['wo']*h[n]+theta['bo']+theta['v4o']*X1[n])     
            chat = np.tanh(theta['v0d']*omega[n]+theta['v1d']*Y[n]+theta['v2d']*var[n]+theta['v3d']*RV[n]+theta['wd']*h[n]+theta['bd']+theta['v4d']*X1[n])   
            c[n+1] = gi*chat+gf*c[n]
            h[n+1] = go*np.tanh(c[n+1])
            omega[n+1] = theta['beta0']+theta['beta1']*h[n+1]
            var[n+1] = omega[n+1]+theta['beta']*var[n]+theta['gamma']*RV[n]  
        eps = Y/np.sqrt(var)
        U   = RV - theta['xi']-theta['phi']*var-theta['tau1']*eps-theta['tau2']*(np.square(eps)-1)
        if lpyt:
            llik_y = stats.norm.logpdf(Y[-1], loc=0, scale=np.sqrt(var[-1]))
            llik_rv = stats.norm.logpdf(U[-1], loc=0, scale=np.sqrt(theta['sigmau2']))
            v = np.average(var[-1]) if self.rs_flag else np.average(var[-1],weights=self.wgts.W) 
            w = np.average(omega[-1]) if self.rs_flag else np.average(omega[-1],weights=self.wgts.W) 
            self.var_ls.append(v) 
            self.w_ls.append(w)       
        else:
            llik_y = np.sum(stats.norm.logpdf(Y, loc=0, scale=np.sqrt(var)), axis=0)  
            llik_rv = np.sum(stats.norm.logpdf(U, loc=0, scale=np.sqrt(theta['sigmau2'])), axis=0)  
        return llik_y+llik_rv       

################################################################### 
# RealRECH + 2x
class RealRECH_2morex(smc.SMC):  
    def __init__(self,data=None, **kwargs):
        super().__init__(**kwargs)
        self.Y, self.RV, self.X1, self.X2 = data
        self.T = self.Y.shape[0]
        self.wgts_ = rs.Weights()
    
    def loglik(self, theta, get_v=False):  
        N = self.Y.shape[0]
        var    = np.zeros((N, theta.shape[0]))
        var[0] = np.var(self.Y)
        omega = np.zeros((N, theta.shape[0]))
        omega[0] = theta['beta0'] # + theta['beta1'] * h0 which is 0 
        h = np.zeros((N, theta.shape[0]))
        c = np.zeros((N, theta.shape[0]))# h0 = 0
        
        for n in range(N-1):  #yt generate std t+1, while we don't need t+1
            gf = ut.sigmoid(theta['v0f']*omega[n]+theta['v1f']*self.Y[n]+theta['v2f']*var[n]+theta['v3f']*self.RV[n]+theta['wf']*h[n]+theta['bf']+theta['v4f']*self.X1[n]+theta['v5f']*self.X2[n])
            gi = ut.sigmoid(theta['v0i']*omega[n]+theta['v1i']*self.Y[n]+theta['v2i']*var[n]+theta['v3i']*self.RV[n]+theta['wi']*h[n]+theta['bi']+theta['v4i']*self.X1[n]+theta['v5i']*self.X2[n])
            go = ut.sigmoid(theta['v0o']*omega[n]+theta['v1o']*self.Y[n]+theta['v2o']*var[n]+theta['v3o']*self.RV[n]+theta['wo']*h[n]+theta['bo']+theta['v4o']*self.X1[n]+theta['v5o']*self.X2[n])         
            chat = np.tanh(theta['v0d']*omega[n]+theta['v1d']*self.Y[n]+theta['v2d']*var[n]+theta['v3d']*self.RV[n]+theta['wd']*h[n]+theta['bd']+theta['v4d']*self.X1[n]+theta['v5d']*self.X2[n])         
            
            c[n+1] = gi*chat+gf*c[n]
            h[n+1] = go*np.tanh(c[n+1])
            omega[n+1] = theta['beta0']+theta['beta1']*h[n+1]
            var[n+1] = omega[n+1]+theta['beta']*var[n]+theta['gamma']*self.RV[n]
        eps = self.Y/np.sqrt(var)
        U   = self.RV - theta['xi'] - theta['phi']*var-theta['tau1']*eps-theta['tau2']*(np.square(eps)-1)
        llik_y = np.sum(stats.norm.logpdf(self.Y[1:], loc=0, scale=np.sqrt(var[1:])), axis=0)  
        llik_rv = np.sum(stats.norm.logpdf(U[1:], loc=0, scale=np.sqrt(theta['sigmau2'])), axis=0)
        if get_v:
            self.var_ls = np.average(var[1:], axis=1, weights=self.wgts.W)
            self.w_ls = list(np.average(omega[1:], axis=1, weights=self.wgts.W))
        return llik_y+llik_rv #(N,) array
    
    def loglik_(self, theta, get_v=False):  
        N = self.Y.shape[0]
        var    = np.zeros((N, theta.shape[0]))
        var[0] = np.var(self.Y)
        omega = np.zeros((N, theta.shape[0]))
        omega[0] = theta['beta0'] # + theta['beta1'] * h0 which is 0 
        h = np.zeros((N, theta.shape[0]))
        c = np.zeros((N, theta.shape[0]))# h0 = 0
        
        for n in range(N-1):  #yt generate std t+1, while we don't need t+1
            gf = ut.sigmoid(theta['v0f']*omega[n]+theta['v1f']*self.Y[n]+theta['v2f']*var[n]+theta['v3f']*self.RV[n]+theta['wf']*h[n]+theta['bf']+theta['v4f']*self.X1[n]+theta['v5f']*self.X2[n])
            gi = ut.sigmoid(theta['v0i']*omega[n]+theta['v1i']*self.Y[n]+theta['v2i']*var[n]+theta['v3i']*self.RV[n]+theta['wi']*h[n]+theta['bi']+theta['v4i']*self.X1[n]+theta['v5i']*self.X2[n])
            go = ut.sigmoid(theta['v0o']*omega[n]+theta['v1o']*self.Y[n]+theta['v2o']*var[n]+theta['v3o']*self.RV[n]+theta['wo']*h[n]+theta['bo']+theta['v4o']*self.X1[n]+theta['v5o']*self.X2[n])         
            chat = np.tanh(theta['v0d']*omega[n]+theta['v1d']*self.Y[n]+theta['v2d']*var[n]+theta['v3d']*self.RV[n]+theta['wd']*h[n]+theta['bd']+theta['v4d']*self.X1[n]+theta['v5d']*self.X2[n])          
            c[n+1] = gi*chat+gf*c[n]
            h[n+1] = go*np.tanh(c[n+1])
            omega[n+1] = theta['beta0']+theta['beta1']*h[n+1]
            var[n+1] = omega[n+1]+theta['beta']*var[n]+theta['gamma']*self.RV[n]
        eps = self.Y/np.sqrt(var)
        U   = self.RV - theta['xi'] - theta['phi']*var-theta['tau1']*eps-theta['tau2']*(np.square(eps)-1)
        llik_y = np.sum(stats.norm.logpdf(self.Y[1:], loc=0, scale=np.sqrt(var[1:])), axis=0)  
        return llik_y
    
    def reweight_particles(self):
        # calculate new epn
        ESSmin = self.ESSrmin_ * self.X.N
        f = lambda e: rs.essl(e * self.X.llik) - ESSmin
        epn = self.X.shared['exponents'][-1]
        if f(1. - epn) > 0:  # we're done (last iteration)
            delta = 1. - epn
            new_epn = 1. # set 1. manually so that we can safely test == 1.
        else:
            delta = sp.optimize.brentq(f, 1.e-12, 1. - epn)  # secant search
            # left endpoint is >0, since f(0.) = nan if any likelihood = -inf
            new_epn = epn + delta
        self.X.shared['exponents'].append(new_epn)
        # calculate delta llik
        dllik = delta * self.X.llik
        self.X.lpost += dllik
        #update weights
        self.wgts = self.wgts.add(dllik)
        self.wgts_ = rs.Weights(delta*self.loglik_(self.X.theta))
        
    def __next__(self):
        if self.done():
            self.loglik(self.X.theta, get_v=True)
            self.var_ls = np.array(self.var_ls)
            self.w_ls = np.array(self.w_ls)
            raise StopIteration
        if self.t == 0:
            self.generate_particles()
        else:
            self.resample_move()
        self.reweight_particles()
        self.wgts_ls.append(self.wgts_)
        self.X_ls.append(self.X)
        if self.verbose:
            print("t={}, accept_rate={:.2f}, accept_rate2={:.2f}, epn={:.3f}".format(self.t, np.average(self.X.shared['acc_rates'][-1]),self.X.shared['acc_rates2'][-1],self.X.shared['exponents'][-1]))
        self.t += 1
class RealRECHD_2morex(smc.SMCD):  
    def __init__(self,data=None, **kwargs):
        super().__init__(**kwargs)
        self.Y_train, self.Y_test, self.RV_train, self.RV_test, self.X1_train, self.X1_test, self.X2_train, self.X2_test = data
        self.T = self.Y_test.shape[0]
        
    def loglik(self, theta, t=None, lpyt=False): 
        Y  = np.concatenate((self.Y_train, self.Y_test[:t+1]))
        RV = np.concatenate((self.RV_train, self.RV_test[:t+1]))
        X1 = np.concatenate((self.X1_train, self.X1_test[:t+1]))
        X2 = np.concatenate((self.X2_train, self.X2_test[:t+1]))
        N = Y.shape[0]
        var    = np.zeros((N, theta.shape[0]))
        var[0] = np.var(Y)
        omega = np.zeros((N, theta.shape[0]))
        omega[0] = theta['beta0'] # + theta['beta1'] * h0 which is 0 
        h = np.zeros((N, theta.shape[0]))
        c = np.zeros((N, theta.shape[0]))# h0 = 0
        for n in range(N-1):  #yt generate std t+1, while we don't need t+1
            gf = ut.sigmoid(theta['v0f']*omega[n]+theta['v1f']*Y[n]+theta['v2f']*var[n]+theta['v3f']*RV[n]+theta['wf']*h[n]+theta['bf']+theta['v4f']*X1[n]+theta['v5f']*X2[n])
            gi = ut.sigmoid(theta['v0i']*omega[n]+theta['v1i']*Y[n]+theta['v2i']*var[n]+theta['v3i']*RV[n]+theta['wi']*h[n]+theta['bi']+theta['v4i']*X1[n]+theta['v5i']*X2[n])
            go = ut.sigmoid(theta['v0o']*omega[n]+theta['v1o']*Y[n]+theta['v2o']*var[n]+theta['v3o']*RV[n]+theta['wo']*h[n]+theta['bo']+theta['v4o']*X1[n]+theta['v5o']*X2[n])         
            chat = np.tanh(theta['v0d']*omega[n]+theta['v1d']*Y[n]+theta['v2d']*var[n]+theta['v3d']*RV[n]+theta['wd']*h[n]+theta['bd']+theta['v4d']*X1[n]+theta['v5d']*X2[n])          
            c[n+1] = gi*chat+gf*c[n]
            h[n+1] = go*np.tanh(c[n+1])
            omega[n+1] = theta['beta0']+theta['beta1']*h[n+1]
            var[n+1] = omega[n+1]+theta['beta']*var[n]+theta['gamma']*RV[n]  
        eps = Y/np.sqrt(var)
        U   = RV - theta['xi']-theta['phi']*var-theta['tau1']*eps-theta['tau2']*(np.square(eps)-1)
        if lpyt:
            llik_y = stats.norm.logpdf(Y[-1], loc=0, scale=np.sqrt(var[-1]))
            llik_rv = stats.norm.logpdf(U[-1], loc=0, scale=np.sqrt(theta['sigmau2']))
            v = np.average(var[-1]) if self.rs_flag else np.average(var[-1],weights=self.wgts.W) 
            w = np.average(omega[-1]) if self.rs_flag else np.average(omega[-1],weights=self.wgts.W) 
            self.var_ls.append(v) 
            self.w_ls.append(w)       
        else:
            llik_y = np.sum(stats.norm.logpdf(Y, loc=0, scale=np.sqrt(var)), axis=0)  
            llik_rv = np.sum(stats.norm.logpdf(U, loc=0, scale=np.sqrt(theta['sigmau2'])), axis=0)  
        return llik_y+llik_rv       

################################################################### 
# EGARCH
class EGARCH(smc.SMC):
    def __init__(self, data=None, **kwargs):
        super().__init__(**kwargs)
        self.Y = data
        self.T = self.Y.shape[0]
    
    def loglik(self, theta, get_v=False):  
        var    = np.zeros((self.Y.shape[0], theta.shape[0])) #var0 is only used to calculate var1, not include in llik calculation 
        var[0] = np.var(self.Y)
        for n in range(self.Y.shape[0]-1):  #yt is excluded as it calculate vart+1
            var[n+1] = np.exp(theta['omega']+theta['beta']*np.log(var[n])+theta['alpha']*(np.abs(self.Y[n])/np.sqrt(var[n])-np.sqrt(2/np.pi))+theta['gammaa']*self.Y[n]/np.sqrt(var[n]))
        llik = np.sum(stats.norm.logpdf(self.Y[1:], loc=0, scale = np.sqrt(var[1:])), axis=0) #y0, var0 is excluded, end with yt, vart
        if get_v:
            self.var_ls = np.average(var[1:], axis=1, weights=self.wgts.W)
        return llik #(N,) array
class EGARCHD(smc.SMCD):
    def __init__(self, data=None, **kwargs):
        super().__init__(**kwargs)
        self.Y_train, self.Y_test = data
        self.T = self.Y_test.shape[0]
    def loglik(self, theta, t=None, lpyt=False):
        Y = np.concatenate((self.Y_train, self.Y_test[:t+1]))
        N = Y.shape[0]
        var = np.zeros((N, theta.shape[0])) 
        var[0] = np.var(Y)
        for n in range(N-1): # yt is excluded as it calculate vart+1
            var[n+1] = np.exp(theta['omega']+theta['beta']*np.log(var[n])+theta['alpha']*(np.abs(Y[n])/np.sqrt(var[n])-np.sqrt(2/np.pi))+theta['gammaa']*Y[n]/np.sqrt(var[n]))
        if lpyt:
            if self.tdist:
                llik = stats.t.logpdf(Y[-1],theta['nu'],loc=0,scale = np.sqrt(var[-1]))
                nu = np.average(theta['nu']) if self.rs_flag else np.average(theta['nu'],weights=self.wgts.W)
                self.nu_ls.append(nu) 
            else:
                llik = stats.norm.logpdf(Y[-1], loc=0, scale = np.sqrt(var[-1]))
            v = np.average(var[-1]) if self.rs_flag else np.average(var[-1],weights=self.wgts.W) 
            self.var_ls.append(v)       
        else:       
            if self.tdist:
                llik = np.sum(stats.t.logpdf(Y,theta['nu'],loc=0,scale=np.sqrt(var)),axis=0)
            else:

                llik = np.sum(stats.norm.logpdf(Y, loc=0, scale = np.sqrt(var)), axis=0)
        return llik

################################################################### 
# RealEGARCH
class RealEGARCH(smc.SMC):
    def __init__(self,data=None, **kwargs):
        super().__init__(**kwargs)
        self.Y, self.RV = data
        self.T = self.Y.shape[0]
    
    def loglik(self, theta, get_v=False):  
        N = self.Y.shape[0]
        var    = np.zeros((self.Y.shape[0], theta.shape[0]))
        var[0] = np.var(self.Y)
        eps = self.Y[0]/np.sqrt(var[0])
        U    = np.zeros((self.Y.shape[0], theta.shape[0]))
        U[0] = np.log(self.RV[0])-theta['xi']-theta['phi']*np.log(var[0]-theta['delta1']*eps-theta['delta1']*(np.square(eps)-1))
        for n in range(N-1):  #yt is excluded as it calculate vart+1
            var[n+1] = np.exp(theta['omega']+theta['beta']*np.log(var[n])+theta['tau1']*eps-theta['tau2']*(np.square(eps)-1)+theta['gamma']*U[n])
            eps = self.Y[n+1]/np.sqrt(var[n+1])
            U[n+1]   = np.log(self.RV[n+1])-theta['xi']-theta['phi']*np.log(var[n+1]-theta['delta1']*eps-theta['delta1']*(np.square(eps)-1))
        llik_y = np.sum(stats.norm.logpdf(self.Y[1:],loc=0, scale=np.sqrt(var[1:])), axis=0)  
        llik_rv = np.sum(stats.norm.logpdf(U[1:], loc=0, scale=np.sqrt(theta['sigmau2'])), axis=0)
        if get_v:
            self.var_ls = np.average(var[1:], axis=1, weights=self.wgts.W)
        return llik_y+llik_rv #(N,) array

    def loglik_(self, theta, get_v=False):  
        N = self.Y.shape[0]
        var    = np.zeros((self.Y.shape[0], theta.shape[0]))
        var[0] = np.var(self.Y)
        eps = self.Y[0]/np.sqrt(var[0])
        U    = np.zeros((self.Y.shape[0], theta.shape[0]))
        U[0] = np.log(self.RV[0])-theta['xi']-theta['phi']*np.log(var[0]-theta['delta1']*eps-theta['delta1']*(np.square(eps)-1))
        for n in range(N-1):  #yt is excluded as it calculate vart+1
            var[n+1] = np.exp(theta['omega']+theta['beta']*np.log(var[n])+theta['tau1']*eps-theta['tau2']*(np.square(eps)-1)+theta['gamma']*U[n])
            eps = self.Y[n+1]/np.sqrt(var[n+1])
            U[n+1]   = np.log(self.RV[n+1])-theta['xi']-theta['phi']*np.log(var[n+1]-theta['delta1']*eps-theta['delta1']*(np.square(eps)-1))
        llik_y = np.sum(stats.norm.logpdf(self.Y[1:],loc=0, scale=np.sqrt(var[1:])), axis=0)  
        return llik_y #(N,) array

    def reweight_particles(self):
        # calculate new epn
        ESSmin = self.ESSrmin_ * self.X.N
        f = lambda e: rs.essl(e * self.X.llik) - ESSmin
        epn = self.X.shared['exponents'][-1]
        if f(1. - epn) > 0:  # we're done (last iteration)
            delta = 1. - epn
            new_epn = 1. # set 1. manually so that we can safely test == 1.
        else:
            delta = sp.optimize.brentq(f, 1.e-12, 1. - epn)  # secant search
            # left endpoint is >0, since f(0.) = nan if any likelihood = -inf
            new_epn = epn + delta
        self.X.shared['exponents'].append(new_epn)
        # calculate delta llik
        dllik = delta * self.X.llik
        self.X.lpost += dllik
        #update weights
        self.wgts = self.wgts.add(dllik)
        self.wgts_ = rs.Weights(delta*self.loglik_(self.X.theta))
        
    def __next__(self):
        if self.done():
            self.loglik(self.X.theta, get_v=True)
            self.var_ls = np.array(self.var_ls)
            self.w_ls = np.array(self.w_ls)
            raise StopIteration
        if self.t == 0:
            self.generate_particles()
        else:
            self.resample_move()
        self.reweight_particles()
        self.wgts_ls.append(self.wgts_)
        self.X_ls.append(self.X)
        if self.verbose:
            print("t={}, accept_rate={:.2f}, accept_rate2={:.2f}, epn={:.3f}".format(self.t, np.average(self.X.shared['acc_rates'][-1]),self.X.shared['acc_rates2'][-1],self.X.shared['exponents'][-1]))
        self.t += 1
class RealEGARCHD(smc.SMCD):
    def __init__(self,data=None, **kwargs):
        super().__init__(**kwargs)
        self.Y_train, self.Y_test, self.RV_train, self.RV_test = data
        self.T = self.Y_test.shape[0]

    def loglik(self, theta, t=None, lpyt=False): 
        Y = np.concatenate((self.Y_train, self.Y_test[:t+1]))
        RV= np.concatenate((self.RV_train, self.RV_test[:t+1]))
        N = Y.shape[0]
        var    = np.zeros((N, theta.shape[0]))
        var[0] = np.var(Y)
        eps = Y[0]/np.sqrt(var[0])
        U    = np.zeros((Y.shape[0], theta.shape[0]))
        U[0] = np.log(RV[0])-theta['xi']-theta['phi']*np.log(var[0]-theta['delta1']*eps-theta['delta1']*(np.square(eps)-1))
        for n in range(N-1):  #yt is excluded as it calculate vart+1
            var[n+1] = np.exp(theta['omega']+theta['beta']*np.log(var[n])+theta['tau1']*eps-theta['tau2']*(np.square(eps)-1)+theta['gamma']*U[n])
            eps = Y[n+1]/np.sqrt(var[n+1])
            U[n+1]   = np.log(RV[n+1])-theta['xi']-theta['phi']*np.log(var[n+1]-theta['delta1']*eps-theta['delta1']*(np.square(eps)-1))
        if lpyt:
            if self.tdist:
                llik_y = stats.t.logpdf(Y[-1],theta['nu'],loc=0, scale=np.sqrt(var[-1])) 
                nu = np.average(theta['nu']) if self.rs_flag else np.average(theta['nu'],weights=self.wgts.W)
                self.nu_ls.append(nu)                 
            else:
                llik_y = stats.norm.logpdf(Y[-1],loc=0,scale=np.sqrt(var[-1]))
            llik_rv = stats.norm.logpdf(U[-1],loc=0, scale=np.sqrt(theta['sigmau2'])) 
            v = np.average(var[-1]) if self.rs_flag else np.average(var[-1],weights=self.wgts.W) 
            self.var_ls.append(v)           
        else:
            llik_y = np.sum(stats.norm.logpdf(Y,loc=0,scale=np.sqrt(var)),axis=0)  
            llik_rv = np.sum(stats.norm.logpdf(U, loc=0, scale=np.sqrt(theta['sigmau2'])),axis=0)
        return llik_y+llik_rv

################################################################### 
# ERECH
class ERECH(smc.SMC):  
    def __init__(self, data=None, **kwargs):
        super().__init__(**kwargs)
        self.Y = data
        self.T = self.Y.shape[0]

    def loglik(self, theta, get_v=False):  
        N = self.Y.shape[0]
        var    = np.zeros((N, theta.shape[0]))
        var[0] = np.var(self.Y)
        omega = np.zeros((N, theta.shape[0]))
        omega[0] = theta['beta0'] # + theta['beta1'] * h0 which is 0 
        h = np.zeros((N, theta.shape[0])) # h0 = 0
        for n in range(N-1):  # yt generate std t+1, while we don't need t+1
            h[n+1] = np.tanh(theta['v0']*omega[n]+theta['v1']*np.exp(self.Y[n])+theta['v2']*var[n]+theta['w']*h[n]+theta['b'])
            omega[n+1] = theta['beta0'] + theta['beta1'] * h[n+1]
            var[n+1] = np.exp(omega[n+1]+theta['beta']*np.log(var[n])+theta['alpha']*(np.abs(self.Y[n])/np.sqrt(var[n])-np.sqrt(2/np.pi))+theta['gammaa']*self.Y[n]/np.sqrt(var[n]))
        llik = np.sum(stats.norm.logpdf(self.Y[1:],loc=0,scale=np.sqrt(var[1:])),axis=0) # y0, std0 is excluded
        if get_v:
            self.var_ls = np.average(var[1:], axis=1, weights=self.wgts.W)
        return llik #(N,) array
class ERECHD(smc.SMCD):
    def __init__(self, data=None, **kwargs):
        super().__init__(**kwargs)
        self.Y_train, self.Y_test = data
        self.T = self.Y_test.shape[0]
    
    def loglik(self, theta, t=None, lpyt=False):  
        Y = np.concatenate((self.Y_train, self.Y_test[:t+1]))
        N = Y.shape[0]
        var    = np.zeros((N, theta.shape[0]))
        var[0] = np.var(Y)
        omega = np.zeros((N, theta.shape[0]))
        omega[0] = theta['beta0'] # + theta['beta1'] * h0 which is 0 
        h = np.zeros((N, theta.shape[0])) # h0 = 0
        for n in range(N-1):  #yt generate std t+1, while we don't need t+1
            h[n+1] = np.tanh(theta['v0']*omega[n]+theta['v1']*np.exp(Y[n])+theta['v2']*var[n]+theta['w']*h[n]+theta['b'])
            omega[n+1] = theta['beta0'] + theta['beta1'] * h[n+1]
            var[n+1] = np.exp(omega[n+1]+theta['beta']*np.log(var[n])+theta['alpha']*(np.abs(Y[n])/np.sqrt(var[n])-np.sqrt(2/np.pi))+theta['gammaa']*Y[n]/np.sqrt(var[n]))
        if lpyt:
            llik = stats.norm.logpdf(Y[-1], loc=0, scale=np.sqrt(var[-1]))
            v = np.average(var[-1]) if self.rs_flag else np.average(var[-1],weights=self.wgts.W) 
            self.var_ls.append(v)              
        else:
            llik = np.sum(stats.norm.logpdf(Y, loc=0, scale=np.sqrt(var)), axis=0)
        return llik 

################################################################### 
# RealERECH 
class RealERECH(smc.SMC):  
    def __init__(self,data=None, **kwargs):
        super().__init__(**kwargs)
        self.Y, self.RV = data
        self.T = self.Y.shape[0]
        
    def loglik(self, theta, get_v=False):  
        N = self.Y.shape[0]
        var    = np.zeros((N, theta.shape[0]))
        var[0] = np.var(self.Y)
        omega = np.zeros((N, theta.shape[0]))
        omega[0] = theta['beta0'] # + theta['beta1'] * h0 which is 0 
        h = np.zeros((N, theta.shape[0]))
        c = np.zeros((N, theta.shape[0]))# h0 = 0
        eps = self.Y[0]/np.sqrt(var[0])
        U    = np.zeros((self.Y.shape[0], theta.shape[0]))
        U[0] = np.log(self.RV[0])-theta['xi']-theta['phi']*np.log(var[0]-theta['delta1']*eps-theta['delta1']*(np.square(eps)-1))
        for n in range(N-1):  #yt generate std t+1, while we don't need t+1
            gf = ut.sigmoid(theta['v0f']*omega[n]+theta['v1f']*np.exp(self.Y[n])+theta['v2f']*var[n]+theta['v3f']*np.exp(self.RV[n])+theta['wf']*h[n]+theta['bf'])
            gi = ut.sigmoid(theta['v0i']*omega[n]+theta['v1i']*np.exp(self.Y[n])+theta['v2i']*var[n]+theta['v3i']*np.exp(self.RV[n])+theta['wi']*h[n]+theta['bi'])
            go = ut.sigmoid(theta['v0o']*omega[n]+theta['v1o']*np.exp(self.Y[n])+theta['v2o']*var[n]+theta['v3o']*np.exp(self.RV[n])+theta['wo']*h[n]+theta['bo'])         
            chat  = np.tanh(theta['v0d']*omega[n]+theta['v1d']*np.exp(self.Y[n])+theta['v2d']*var[n]+theta['v3d']*np.exp(self.RV[n])+theta['wd']*h[n]+theta['bd'])         
            c[n+1] = gi*chat+gf*c[n]
            h[n+1] = go*np.tanh(c[n+1])
            omega[n+1] = theta['beta0']+theta['beta1']*h[n+1]
            var[n+1] = np.exp(omega[n+1]+theta['beta']*np.log(var[n])+theta['tau1']*eps-theta['tau2']*(np.square(eps)-1)+theta['gamma']*U[n])
            eps = self.Y[n+1]/np.sqrt(var[n+1])
            U[n+1]   = np.log(self.RV[n+1])-theta['xi']-theta['phi']*np.log(var[n+1]-theta['delta1']*eps-theta['delta1']*(np.square(eps)-1))
        llik_y = np.sum(stats.norm.logpdf(self.Y[1:], loc=0, scale=np.sqrt(var[1:])), axis=0)  
        llik_rv = np.sum(stats.norm.logpdf(U[1:], loc=0, scale=np.sqrt(theta['sigmau2'])), axis=0)
        if get_v:
            self.var_ls = np.average(var[1:], axis=1, weights=self.wgts.W)
            self.w_ls = list(np.average(omega[1:], axis=1, weights=self.wgts.W))
        return llik_y+llik_rv #(N,) array

    def loglik_(self, theta, get_v=False):  
        N = self.Y.shape[0]
        var    = np.zeros((N, theta.shape[0]))
        var[0] = np.var(self.Y)
        omega = np.zeros((N, theta.shape[0]))
        omega[0] = theta['beta0'] # + theta['beta1'] * h0 which is 0 
        h = np.zeros((N, theta.shape[0]))
        c = np.zeros((N, theta.shape[0]))# h0 = 0
        eps = self.Y[0]/np.sqrt(var[0])
        U    = np.zeros((self.Y.shape[0], theta.shape[0]))
        U[0] = np.log(self.RV[0])-theta['xi']-theta['phi']*np.log(var[0]-theta['delta1']*eps-theta['delta1']*(np.square(eps)-1))
        for n in range(N-1):  #yt generate std t+1, while we don't need t+1
            gf = ut.sigmoid(theta['v0f']*omega[n]+theta['v1f']*np.exp(self.Y[n])+theta['v2f']*var[n]+theta['v3f']*np.exp(self.RV[n])+theta['wf']*h[n]+theta['bf'])
            gi = ut.sigmoid(theta['v0i']*omega[n]+theta['v1i']*np.exp(self.Y[n])+theta['v2i']*var[n]+theta['v3i']*np.exp(self.RV[n])+theta['wi']*h[n]+theta['bi'])
            go = ut.sigmoid(theta['v0o']*omega[n]+theta['v1o']*np.exp(self.Y[n])+theta['v2o']*var[n]+theta['v3o']*np.exp(self.RV[n])+theta['wo']*h[n]+theta['bo'])         
            chat  = np.tanh(theta['v0d']*omega[n]+theta['v1d']*np.exp(self.Y[n])+theta['v2d']*var[n]+theta['v3d']*np.exp(self.RV[n])+theta['wd']*h[n]+theta['bd'])         
            c[n+1] = gi*chat+gf*c[n]
            h[n+1] = go*np.tanh(c[n+1])
            omega[n+1] = theta['beta0']+theta['beta1']*h[n+1]
            var[n+1] = np.exp(omega[n+1]+theta['beta']*np.log(var[n])+theta['tau1']*eps-theta['tau2']*(np.square(eps)-1)+theta['gamma']*U[n])
            eps = self.Y[n+1]/np.sqrt(var[n+1])
            U[n+1] = np.log(self.RV[n+1])-theta['xi']-theta['phi']*np.log(var[n+1]-theta['delta1']*eps-theta['delta1']*(np.square(eps)-1))
        llik_y = np.sum(stats.norm.logpdf(self.Y[1:], loc=0, scale=np.sqrt(var[1:])), axis=0)  
        return llik_y


    def reweight_particles(self):
        # calculate new epn
        ESSmin = self.ESSrmin_ * self.X.N
        f = lambda e: rs.essl(e * self.X.llik) - ESSmin
        epn = self.X.shared['exponents'][-1]
        if f(1. - epn) > 0:  # we're done (last iteration)
            delta = 1. - epn
            new_epn = 1. # set 1. manually so that we can safely test == 1.
        else:
            delta = sp.optimize.brentq(f, 1.e-12, 1. - epn)  # secant search
            # left endpoint is >0, since f(0.) = nan if any likelihood = -inf
            new_epn = epn + delta
        self.X.shared['exponents'].append(new_epn)
        # calculate delta llik
        dllik = delta * self.X.llik
        self.X.lpost += dllik
        #update weights
        self.wgts = self.wgts.add(dllik)
        self.wgts_ = rs.Weights(delta*self.loglik_(self.X.theta))
        
    def __next__(self):
        if self.done():
            self.loglik(self.X.theta, get_v=True)
            self.var_ls = np.array(self.var_ls)
            self.w_ls = np.array(self.w_ls)
            raise StopIteration
        if self.t == 0:
            self.generate_particles()
        else:
            self.resample_move()
        self.reweight_particles()
        self.wgts_ls.append(self.wgts_)
        self.X_ls.append(self.X)
        if self.verbose:
            print("t={}, accept_rate={:.2f}, accept_rate2={:.2f}, epn={:.3f}".format(self.t, np.average(self.X.shared['acc_rates'][-1]),self.X.shared['acc_rates2'][-1],self.X.shared['exponents'][-1]))
        self.t += 1
class RealERECHD(smc.SMCD):  
    def __init__(self,data=None, **kwargs):
        super().__init__(**kwargs)
        self.Y_train, self.Y_test, self.RV_train, self.RV_test = data
        self.T = self.Y_test.shape[0]
        
    def loglik(self, theta, t=None, lpyt=False): 
        Y = np.concatenate((self.Y_train, self.Y_test[:t+1]))
        RV= np.concatenate((self.RV_train, self.RV_test[:t+1]))
        N = Y.shape[0]
        var    = np.zeros((N, theta.shape[0]))
        var[0] = np.var(Y)
        omega = np.zeros((N, theta.shape[0]))
        omega[0] = theta['beta0'] # + theta['beta1'] * h0 which is 0 
        h = np.zeros((N, theta.shape[0]))
        c = np.zeros((N, theta.shape[0]))# h0 = 0
        eps = Y[0]/np.sqrt(var[0])
        U    = np.zeros((Y.shape[0], theta.shape[0]))
        U[0] = np.log(RV[0])-theta['xi']-theta['phi']*np.log(var[0]-theta['delta1']*eps-theta['delta1']*(np.square(eps)-1))
        for n in range(N-1):  #yt generate std t+1, while we don't need t+1
            gf = ut.sigmoid(theta['v0f']*omega[n]+theta['v1f']*np.exp(Y[n])+theta['v2f']*var[n]+theta['v3f']*RV[n]+np.exp(theta['wf'])*h[n]+theta['bf'])
            gi = ut.sigmoid(theta['v0i']*omega[n]+theta['v1i']*np.exp(Y[n])+theta['v2i']*var[n]+theta['v3i']*RV[n]+np.exp(theta['wi'])*h[n]+theta['bi'])
            go = ut.sigmoid(theta['v0o']*omega[n]+theta['v1o']*np.exp(Y[n])+theta['v2o']*var[n]+theta['v3o']*RV[n]+np.exp(theta['wo'])*h[n]+theta['bo'])         
            chat =  np.tanh(theta['v0d']*omega[n]+theta['v1d']*np.exp(Y[n])+theta['v2d']*var[n]+theta['v3d']*RV[n]+np.exp(theta['wd'])*h[n]+theta['bd'])         
            c[n+1] = gi*chat+gf*c[n]
            h[n+1] = go*np.tanh(c[n+1])
            omega[n+1] = theta['beta0']+theta['beta1']*h[n+1]
            var[n+1] = np.exp(omega[n+1]+theta['beta']*np.log(var[n])+theta['tau1']*eps-theta['tau2']*(np.square(eps)-1)+theta['gamma']*U[n])
            eps = Y[n+1]/np.sqrt(var[n+1])
            U[n+1]   = np.log(RV[n+1])-theta['xi']-theta['phi']*np.log(var[n+1]-theta['delta1']*eps-theta['delta1']*(np.square(eps)-1))
        if lpyt:
            llik_y = stats.norm.logpdf(Y[-1], loc=0, scale=np.sqrt(var[-1]))
            llik_rv = stats.norm.logpdf(U[-1], loc=0, scale=np.sqrt(theta['sigmau2']))
            v = np.average(var[-1]) if self.rs_flag else np.average(var[-1],weights=self.wgts.W) 
            w = np.average(omega[-1]) if self.rs_flag else np.average(omega[-1],weights=self.wgts.W) 
            self.var_ls.append(v)    
            self.w_ls.append(w)      
        else:
            llik_y = np.sum(stats.norm.logpdf(Y, loc=0, scale=np.sqrt(var)), axis=0)  
            llik_rv = np.sum(stats.norm.logpdf(U, loc=0, scale=np.sqrt(theta['sigmau2'])), axis=0)  
        return llik_y+llik_rv     
    