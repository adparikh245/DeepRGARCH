import numpy as np
import scipy as sp
import sys, os
sys.path.append(os.path.abspath("/Users/ananyaparikh/Documents/Coding/DeepRGARCH/DeepRGARCH"))
from scipy import stats
from scipy.special import gammaln 
from rerech import smc, utils as ut, resampling as rs
from rerech import model_patch
from smc import ThetaParticles

##########s######################################################### 
# RealRECH 
class RealRECH_2LSTM(smc.SMC):  
    def __init__(self,data=None, **kwargs):
        super().__init__(**kwargs)
        self.Y_train, self.Y_test, self.RV_train, self.RV_test = data
        # build full series so calculate_variance can see it
        self.Y = np.concatenate((self.Y_train, self.Y_test))
        self.RV = np.concatenate((self.RV_train, self.RV_test))
        self.T = self.Y_test.shape[0]
        self.ESSrmin_ = getattr(self, 'ESSrmin', 0.5)
    def safe_sqrt(self, x):
        return np.sqrt(np.maximum(x, 1e-10))    
    def safe_logpdf_t(self, x, df, loc, scale):
        scale = np.maximum(scale, 1e-10)
        df = np.maximum(df, 2.1)
        try:
            return stats.t.logpdf(x, df, loc=loc, scale=scale)
        except:
            log_gamma_df_half = np.log(np.sqrt(np.pi)) + np.log(df / 2)
            log_gamma_half_df_half = np.log(np.sqrt(np.pi * df))
            log_pdf = log_gamma_half_df_half - log_gamma_df_half - 0.5 * np.log(df) - \
                    np.log(scale) - ((df + 1) / 2) * np.log(1 + x**2 / df)
            return log_pdf
    def safe_logpdf_norm(self, x, loc, scale):
        scale = np.maximum(scale, 1e-10)
        z = (x - loc) / scale
        z_safe = np.clip(z, -100, 100)
        log_pdf = -0.5 * np.log(2 * np.pi) - np.log(scale) - 0.5 * (z_safe ** 2)
        return log_pdf
    # Add this to both classes in the model.py file
    def calculate_variance(self, omega, beta, gamma, rv, var_prev):
        min_var = np.percentile(self.Y**2, 1)  # now self.Y exists
        return np.maximum(min_var, omega + beta*var_prev + gamma*rv)
    
    def loglik(self, theta, get_v=False):  
        N = self.Y.shape[0]
        var    = np.zeros((N, theta.shape[0]))
        var[0] = np.var(self.Y)
        omega = np.zeros((N, theta.shape[0]))
        omega[0] = theta['beta0'] # + theta['beta1'] * h0 which is 0 
        h = np.zeros((N, theta.shape[0]))
        c = np.zeros((N, theta.shape[0]))# h0 = 0
        
        U  = np.zeros((self.Y.shape[0], theta.shape[0]))
        xi = np.zeros((self.Y.shape[0], theta.shape[0]))
        xi[0] = theta['xi'] # + theta['beta1'] * h0 which is 0 
        h_rv = np.zeros((self.Y.shape[0], theta.shape[0]))
        c_rv = np.zeros((self.Y.shape[0], theta.shape[0]))
        
        for n in range(self.Y.shape[0]-1):  #yt generate std t+1, while we don't need t+1
            gf = ut.sigmoid(theta['v0f']*omega[n]+theta['v1f']*self.Y[n]+theta['v2f']*var[n]+theta['v3f']*self.RV[n]+theta['wf']*h[n]+theta['bf'])
            gi = ut.sigmoid(theta['v0i']*omega[n]+theta['v1i']*self.Y[n]+theta['v2i']*var[n]+theta['v3i']*self.RV[n]+theta['wi']*h[n]+theta['bi'])
            go = ut.sigmoid(theta['v0o']*omega[n]+theta['v1o']*self.Y[n]+theta['v2o']*var[n]+theta['v3o']*self.RV[n]+theta['wo']*h[n]+theta['bo'])         
            chat =  np.tanh(theta['v0d']*omega[n]+theta['v1d']*self.Y[n]+theta['v2d']*var[n]+theta['v3d']*self.RV[n]+theta['wd']*h[n]+theta['bd'])         
            c[n+1] = gi*chat+gf*c[n]
            h[n+1] = go*np.tanh(c[n+1])
            # omega[n+1] = ut.relu(theta['beta0']+theta['beta1']*h[n+1])
            raw_ω = theta['beta0'] + theta['beta1'] * h[n+1]
            # scale = theta.get('ω_scale', 1.0)
            scale = 1.0
            omega[n+1] = ut.relu(raw_ω) * scale
            # var[n+1] = omega[n+1]+theta['beta']*var[n]+theta['gamma']*self.RV[n]
            var[n+1] = self.calculate_variance(omega[n+1], theta['beta'], theta['gamma'], self.RV[n], var[n])
            
            eps = self.Y[n+1]/var[n+1]
            gf = ut.sigmoid(theta['v0f_rv']*xi[n]+theta['v1f_rv']*self.Y[n]+theta['v2f_rv']*var[n]+theta['v3f_rv']*self.RV[n]+theta['wf_rv']*h_rv[n]+theta['bf_rv'])
            gi = ut.sigmoid(theta['v0i_rv']*xi[n]+theta['v1i_rv']*self.Y[n]+theta['v2i_rv']*var[n]+theta['v3i_rv']*self.RV[n]+theta['wi_rv']*h_rv[n]+theta['bi_rv'])
            go = ut.sigmoid(theta['v0o_rv']*xi[n]+theta['v1o_rv']*self.Y[n]+theta['v2o_rv']*var[n]+theta['v3o_rv']*self.RV[n]+theta['wo_rv']*h_rv[n]+theta['bo_rv'])         
            chat =  np.tanh(theta['v0d_rv']*xi[n]+theta['v1d_rv']*self.Y[n]+theta['v2d_rv']*var[n]+theta['v3d_rv']*self.RV[n]+theta['wd_rv']*h_rv[n]+theta['bd_rv'])         
            c_rv[n+1] = gi*chat+gf*c_rv[n]
            h_rv[n+1] = go*np.tanh(c_rv[n+1])
            xi[n+1] = ut.relu(theta['beta0_rv']+theta['beta1_rv']*h_rv[n+1])
            U[n+1] = self.RV[n+1] - xi[n+1] - theta['phi']*var[n+1]-theta['tau1']*eps-theta['tau2']*(np.square(eps)-1)

        # llik_y = np.sum(stats.norm.logpdf(self.Y[1:], loc=0, scale=np.sqrt(var[1:])), axis=0)  
        # llik_rv = np.sum(stats.norm.logpdf(U[1:], loc=0, scale=np.sqrt(theta['sigmau2'])), axis=0)
        llik_y = np.sum(self.safe_logpdf_norm(self.Y[1:], 0, self.safe_sqrt(var[1:])), axis=0)
        llik_rv = np.sum(self.safe_logpdf_norm(U[1:], 0, self.safe_sqrt(np.maximum(theta['sigmau2'], 1e-10))), axis=0)
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
        
        for n in range(self.Y.shape[0]-1):  #yt generate std t+1, while we don't need t+1
            gf = ut.sigmoid(theta['v0f']*omega[n]+theta['v1f']*self.Y[n]+theta['v2f']*var[n]+theta['v3f']*self.RV[n]+theta['wf']*h[n]+theta['bf'])
            gi = ut.sigmoid(theta['v0i']*omega[n]+theta['v1i']*self.Y[n]+theta['v2i']*var[n]+theta['v3i']*self.RV[n]+theta['wi']*h[n]+theta['bi'])
            go = ut.sigmoid(theta['v0o']*omega[n]+theta['v1o']*self.Y[n]+theta['v2o']*var[n]+theta['v3o']*self.RV[n]+theta['wo']*h[n]+theta['bo'])         
            chat =  np.tanh(theta['v0d']*omega[n]+theta['v1d']*self.Y[n]+theta['v2d']*var[n]+theta['v3d']*self.RV[n]+theta['wd']*h[n]+theta['bd'])         
            c[n+1] = gi*chat+gf*c[n]
            h[n+1] = go*np.tanh(c[n+1])
            #omega[n+1] = ut.relu(theta['beta0']+theta['beta1']*h[n+1])
            raw_ω = theta['beta0'] + theta['beta1'] * h[n+1]
            # scale = theta.get('ω_scale', 1.0)
            scale = 1.0
            omega[n+1] = ut.relu(raw_ω) * scale
            # var[n+1] = omega[n+1]+theta['beta']*var[n]+theta['gamma']*self.RV[n]
            var[n+1] = self.calculate_variance(omega[n+1], theta['beta'], theta['gamma'], self.RV[n], var[n])

        #llik_y = np.sum(stats.norm.logpdf(self.Y[1:], loc=0, scale=np.sqrt(var[1:])), axis=0)
        llik_y = np.sum(self.safe_logpdf_norm(self.Y[1:], 0, self.safe_sqrt(var[1:])), axis=0)  
        return llik_y
    
    def reweight_particles(self):
        # calculate new epn
        N = self.X.theta.shape[0]  # Get N from the shape of theta array
        ESSmin = self.ESSrmin_ * N  # Using ESSrmin_ consistently
        f = lambda e: rs.essl(e * self.X.llik) - ESSmin
        epn = self.X.shared['exponents'][-1]
        if f(1. - epn) > 0:  # we're done (last iteration)
            delta = 1. - epn
            new_epn = 1. # set 1. manually so that we can safely test == 1.
        else:
            # Change rs.brentq to sp.optimize.brentq
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
    def __iter__(self):
        return self
    
class RealRECH_2LSTM_tdist(smc.SMC):  
    def __init__(self, data=None, **kwargs):
        super().__init__(**kwargs)
        # Changed to expect only 2 values in data - this is the crucial fix
        self.Y, self.RV = data
        self.T = self.Y.shape[0]
        self.wgts_ = rs.Weights()
        self.pre = None
        # Important to set this for calculate_variance method
        self.Y_train = self.Y  # Since we need this attribute
        self.ESSrmin_ = getattr(self, 'ESSrmin', 0.5)  # Default to 0.5 if ESSrmin doesn't exiss
    def generate_particles(self):
        if self.pre is None:
            # Initialize particles from the prior
            theta_samples = self.prior.rvs(size=self.N)
            self.X = ThetaParticles(theta=theta_samples)
            self.X.shared = {'acc_rates': [0.], 'exponents': [0.]}
            self.wgts = rs.Weights()
        else:
            super().generate_particles()
            
        self.X.shared['acc_rates'] = [0.]
        self.X.shared['exponents'] = [0.] 
    def safe_sqrt(self, x):
        return np.sqrt(np.maximum(x, 1e-10))

    def safe_logpdf_norm(self, x, loc, scale):
        scale = np.maximum(scale, 1e-10)
        z = (x - loc) / scale
        z_safe = np.clip(z, -100, 100)
        return -0.5 * np.log(2 * np.pi) - np.log(scale) - 0.5 * z_safe**2
    
    def safe_logpdf_t(self, x, df, loc, scale):
        x = np.asarray(x, dtype=np.float64)
        df = np.maximum(np.asarray(df, dtype=np.float64), 2.1)
        scale = np.maximum(np.asarray(scale, dtype=np.float64), 1e-10)
        z = (x - loc) / scale
        z_safe = np.clip(z, -50, 50)
        log_gamma_df_half = gammaln(df/2)
        log_gamma_df_plus_half = gammaln((df+1)/2)
        log_term = log_gamma_df_plus_half - log_gamma_df_half - 0.5*np.log(df*np.pi)
        kernel = -((df+1)/2)*np.log1p(z_safe**2/df)
        return log_term + kernel - np.log(scale)

    def calculate_variance(self, omega, beta, gamma, rv, var_prev):
        # enforce positivity
        base = omega + beta*var_prev + gamma*rv
        return np.maximum(1e-6, base)
    
    def loglik(self, theta, get_v=False):  
        df_y = theta['nu']
        df_u = theta['df_u']
        N = self.Y_train.shape[0]
        var    = np.zeros((N, theta.shape[0]))
        var[0] = np.var(self.Y)
        omega = np.zeros((N, theta.shape[0]))
        omega[0] = theta['beta0'] # + theta['beta1'] * h0 which is 0 
        h = np.zeros((N, theta.shape[0]))
        c = np.zeros((N, theta.shape[0]))# h0 = 0
        
        U  = np.zeros((self.Y.shape[0], theta.shape[0]))
        xi = np.zeros((self.Y.shape[0], theta.shape[0]))
        xi[0] = theta['xi'] # + theta['beta1'] * h0 which is 0 
        h_rv = np.zeros((self.Y.shape[0], theta.shape[0]))
        c_rv = np.zeros((self.Y.shape[0], theta.shape[0]))
        
        for n in range(self.Y.shape[0]-1):  #yt generate std t+1, while we don't need t+1
            gf = ut.sigmoid(theta['v0f']*omega[n]+theta['v1f']*self.Y[n]+theta['v2f']*var[n]+theta['v3f']*self.RV[n]+theta['wf']*h[n]+theta['bf'])
            gi = ut.sigmoid(theta['v0i']*omega[n]+theta['v1i']*self.Y[n]+theta['v2i']*var[n]+theta['v3i']*self.RV[n]+theta['wi']*h[n]+theta['bi'])
            go = ut.sigmoid(theta['v0o']*omega[n]+theta['v1o']*self.Y[n]+theta['v2o']*var[n]+theta['v3o']*self.RV[n]+theta['wo']*h[n]+theta['bo'])         
            chat =  np.tanh(theta['v0d']*omega[n]+theta['v1d']*self.Y[n]+theta['v2d']*var[n]+theta['v3d']*self.RV[n]+theta['wd']*h[n]+theta['bd'])         
            c[n+1] = gi*chat+gf*c[n]
            h[n+1] = go*np.tanh(c[n+1])
            # omega[n+1] = ut.relu(theta['beta0']+theta['beta1']*h[n+1])
            raw_ω = theta['beta0'] + theta['beta1'] * h[n+1]
            # scale = theta.get('ω_scale', 1.0)
            scale = 1.0
            omega[n+1] = ut.relu(raw_ω) * scale
            #var[n+1] = omega[n+1]+theta['beta']*var[n]+theta['gamma']*self.RV[n]
            var[n+1] = self.calculate_variance(omega[n+1], theta['beta'], theta['gamma'], self.RV[n], var[n])
            
            eps = self.Y[n+1]/var[n+1]
            gf = ut.sigmoid(theta['v0f_rv']*xi[n]+theta['v1f_rv']*self.Y[n]+theta['v2f_rv']*var[n]+theta['v3f_rv']*self.RV[n]+theta['wf_rv']*h_rv[n]+theta['bf_rv'])
            gi = ut.sigmoid(theta['v0i_rv']*xi[n]+theta['v1i_rv']*self.Y[n]+theta['v2i_rv']*var[n]+theta['v3i_rv']*self.RV[n]+theta['wi_rv']*h_rv[n]+theta['bi_rv'])
            go = ut.sigmoid(theta['v0o_rv']*xi[n]+theta['v1o_rv']*self.Y[n]+theta['v2o_rv']*var[n]+theta['v3o_rv']*self.RV[n]+theta['wo_rv']*h_rv[n]+theta['bo_rv'])         
            chat =  np.tanh(theta['v0d_rv']*xi[n]+theta['v1d_rv']*self.Y[n]+theta['v2d_rv']*var[n]+theta['v3d_rv']*self.RV[n]+theta['wd_rv']*h_rv[n]+theta['bd_rv'])         
            c_rv[n+1] = gi*chat+gf*c_rv[n]
            h_rv[n+1] = go*np.tanh(c_rv[n+1])
            xi[n+1] = ut.relu(theta['beta0_rv']+theta['beta1_rv']*h_rv[n+1])
            U[n+1] = self.RV[n+1] - xi[n+1] - theta['phi']*var[n+1]-theta['tau1']*eps-theta['tau2']*(np.square(eps)-1)
            
        # Use instance methods with self
        llik_y = np.sum(self.safe_logpdf_t(self.Y[1:], df_y, 0, self.safe_sqrt(var[1:])), axis=0)
        llik_rv = np.sum(self.safe_logpdf_t(U[1:], df_u, 0, self.safe_sqrt(np.maximum(theta['sigmau2'], 1e-10))), axis=0)
        
        if get_v:
            self.var_ls = np.average(var[1:], axis=1, weights=self.wgts.W)
            self.w_ls = list(np.average(omega[1:], axis=1, weights=self.wgts.W))
        return llik_y + llik_rv


class RealRECHD_2LSTM_tdist(smc.SMCD):
    """
    Data‐annealing SMC version of 2‐LSTM RECH with Student-t observation errors.
    """
    def __init__(self, data=None, **kwargs):
        super().__init__(**kwargs)
        # Fixed to correctly unpack 4 values
        self.Y_train, self.Y_test, self.RV_train, self.RV_test = data
        # Add these lines to create concatenated series
        self.Y = np.concatenate((self.Y_train, self.Y_test))
        self.RV = np.concatenate((self.RV_train, self.RV_test))
        self.T = self.Y_test.shape[0]
        self.ESSrmin_ = getattr(self, 'ESSrmin', 0.5)
    def calculate_variance(self, omega, beta, gamma, rv, var_prev):
        floor = np.percentile(self.Y_train**2, 1)  # Use Y_train instead of Y
        return np.maximum(floor, omega + beta * var_prev + gamma * rv)

    def safe_sqrt(self, x):
        return np.sqrt(np.maximum(x, 1e-10))

    def safe_logpdf_t(self, x, df, loc, scale):
        """
        Robust Student‐t log‐pdf:
        - clips df ≥ 2.1, scale ≥ 1e-10
        - vectorized over x
        """
        x = np.asarray(x, dtype=np.float64)
        df = np.maximum(df, 2.1)
        scale = np.maximum(scale, 1e-10)
        z = (x - loc) / scale
        z = np.clip(z, -50, 50)

        # log‐pdf = log Γ((ν+1)/2) − log Γ(ν/2) − ½ log(πν) − log σ − (ν+1)/2 · log(1 + z²/ν)
        return (
            gammaln((df + 1) / 2)
            - gammaln(df / 2)
            - 0.5 * np.log(df * np.pi)
            - np.log(scale)
            - ((df + 1) / 2) * np.log1p(z**2 / df)
        )

    def loglik(self, theta, t=None, lpyt=False):
        """
        If lpyt=True, only evaluate the last step (for one‐step‐ahead),
        else sum over the whole (train+test[:t+1]) path.
        """
        # 1) build the concatenated series up to time t
        if t is None:
            Y = np.concatenate((self.Y_train, self.Y_test))
            RV = np.concatenate((self.RV_train, self.RV_test))
        else:
            Y = np.concatenate((self.Y_train, self.Y_test[:t+1]))
            RV = np.concatenate((self.RV_train, self.RV_test[:t+1]))

        N_all = Y.shape[0]
        K = theta.shape[0]           # number of particles

        # allocate
        var   = np.zeros((N_all, K))
        omega = np.zeros((N_all, K))
        h     = np.zeros((N_all, K))
        c     = np.zeros((N_all, K))

        U     = np.zeros((N_all, K))
        xi    = np.zeros((N_all, K))
        h_rv  = np.zeros((N_all, K))
        c_rv  = np.zeros((N_all, K))

        # initial states
        var[0]   = np.var(self.Y_train)
        omega[0] = theta['beta0']
        xi[0]    = theta['xi']

        nu_y = theta['nu']
        nu_u = theta['df_u']

        # forward‐recursion
        for n in range(N_all - 1):
            # ———— volatility LSTM ————
            gf = ut.sigmoid(theta['v0f']   * omega[n] +
                            theta['v1f']   * Y[n]    +
                            theta['v2f']   * var[n]  +
                            theta['v3f']   * RV[n]   +
                            theta['wf']    * h[n]    +
                            theta['bf'])
            gi = ut.sigmoid(theta['v0i']   * omega[n] +
                            theta['v1i']   * Y[n]    +
                            theta['v2i']   * var[n]  +
                            theta['v3i']   * RV[n]   +
                            theta['wi']    * h[n]    +
                            theta['bi'])
            go = ut.sigmoid(theta['v0o']   * omega[n] +
                            theta['v1o']   * Y[n]    +
                            theta['v2o']   * var[n]  +
                            theta['v3o']   * RV[n]   +
                            theta['wo']    * h[n]    +
                            theta['bo'])
            chat = np.tanh(theta['v0d']    * omega[n] +
                           theta['v1d']    * Y[n]    +
                           theta['v2d']    * var[n]  +
                           theta['v3d']    * RV[n]   +
                           theta['wd']     * h[n]    +
                           theta['bd'])
            c[n+1] = gi * chat + gf * c[n]
            h[n+1] = go * np.tanh(c[n+1])

            raw_ω = theta['beta0'] + theta['beta1'] * h[n+1]
            omega[n+1] = ut.relu(raw_ω)

            var[n+1] = self.calculate_variance(
                omega[n+1],
                theta['beta'],
                theta['gamma'],
                RV[n],
                var[n]
            )

            # ———— measurement LSTM on RV ————
            eps = Y[n+1] / var[n+1]

            gf_rv = ut.sigmoid(theta['v0f_rv'] * xi[n] +
                               theta['v1f_rv'] * Y[n]    +
                               theta['v2f_rv'] * var[n]  +
                               theta['v3f_rv'] * RV[n]   +
                               theta['wf_rv']  * h_rv[n] +
                               theta['bf_rv'])
            gi_rv = ut.sigmoid(theta['v0i_rv'] * xi[n] +
                               theta['v1i_rv'] * Y[n]    +
                               theta['v2i_rv'] * var[n]  +
                               theta['v3i_rv'] * RV[n]   +
                               theta['wi_rv']  * h_rv[n] +
                               theta['bi_rv'])
            go_rv = ut.sigmoid(theta['v0o_rv'] * xi[n] +
                               theta['v1o_rv'] * Y[n]    +
                               theta['v2o_rv'] * var[n]  +
                               theta['v3o_rv'] * RV[n]   +
                               theta['wo_rv']  * h_rv[n] +
                               theta['bo_rv'])
            chat_rv = np.tanh(theta['v0d_rv'] * xi[n] +
                              theta['v1d_rv'] * Y[n]    +
                              theta['v2d_rv'] * var[n]  +
                              theta['v3d_rv'] * RV[n]   +
                              theta['wd_rv']  * h_rv[n] +
                              theta['bd_rv'])
            c_rv[n+1] = gi_rv * chat_rv + gf_rv * c_rv[n]
            h_rv[n+1] = go_rv * np.tanh(c_rv[n+1])

            xi[n+1] = ut.relu(theta['beta0_rv'] + theta['beta1_rv'] * h_rv[n+1])

            U[n+1] = (RV[n+1]
                      - xi[n+1]
                      - theta['phi'] * var[n+1]
                      - theta['tau1'] * eps
                      - theta['tau2'] * (eps**2 - 1))

        # compute log-lik
        if lpyt:
            # only last step
            l_y = self.safe_logpdf_t(
                Y[-1], nu_y, loc=0, scale=self.safe_sqrt(var[-1])
            )
            l_u = self.safe_logpdf_t(
                U[-1], nu_u, loc=0, scale=self.safe_sqrt(theta['sigmau2'])
            )
            # store forecasts
            v = np.average(var[-1], weights=self.wgts.W)
            w = np.average(omega[-1], weights=self.wgts.W)
            self.var_ls.append(v)
            self.w_ls.append(w)
            return l_y + l_u
        else:
            # sum over all
            l_y = np.sum(self.safe_logpdf_t(
                Y[1:], nu_y, loc=0, scale=self.safe_sqrt(var[1:])
            ), axis=0)
            l_u = np.sum(self.safe_logpdf_t(
                U[1:], nu_u, loc=0, scale=self.safe_sqrt(theta['sigmau2'])
            ), axis=0)
            return l_y + l_u

    def reweight_particles(self):
        """
        identical to base SMCD, but using our local loglik_ if needed
        """
        N = self.X.N
        ESSmin = self.ESSrmin_ * N
        f = lambda e: rs.essl(e * self.X.llik) - ESSmin

        epn = self.X.shared['exponents'][-1]
        if f(1. - epn) > 0:
            delta = 1. - epn
            new_epn = 1.
        else:
            # Fixed to use sp.optimize.brentq instead of rs.brentq
            delta = np.clip(
                sp.optimize.brentq(f, 1e-12, 1. - epn),
                1e-12, 1. - epn
            )
            new_epn = epn + delta

        self.X.shared['exponents'].append(new_epn)
        dllik = delta * self.X.llik
        self.X.lpost += dllik
        self.wgts_ = rs.Weights(delta * self.loglik(self.X.theta))

    def __next__(self):
        if self.done():
            self.loglik(self.X.theta, lpyt=True)
            self.var_ls = np.array(self.var_ls)
            self.w_ls   = np.array(self.w_ls)
            raise StopIteration

        if self.t == 0:
            self.generate_particles()
        else:
            self.resample_move()

        self.reweight_particles()
        self.wgts_ls.append(self.wgts_)
        self.X_ls.append(self.X)

        if self.verbose:
            r1 = np.average(self.X.shared['acc_rates'][-1])
            r2 = self.X.shared['acc_rates2'][-1]
            e  = self.X.shared['exponents'][-1]
            print(f"t={self.t}, α={r1:.2f}, α₂={r2:.2f}, epn={e:.3f}")

        self.t += 1
        return self

    def __iter__(self):
        return self
