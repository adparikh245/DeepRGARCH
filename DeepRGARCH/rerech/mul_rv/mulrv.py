from __future__ import annotations

import numpy as np
import scipy.stats as stats
import scipy as sp

import rerech.utils as ut                  # unchanged helpers
from rerech import distributions as dists  # prior building blocks
import rerech.resampling as rs             # SMC weights / resampling
from rerech.smc import SMC, SMCD          # original base classes

__all__ = [
    "make_multi_rv_prior",
    "RealRECH_2LSTM",
    "RealRECHD_2LSTM",
]

# =====================================================================
# 1.  PRIOR factory for K realised‑vol measures
# =====================================================================

def make_multi_rv_prior(K: int) -> dists.StructDist:
    """Return a StructDist with K‑vector realised‑vol parameters.

    The scalar entries are copied verbatim from the single‑RV prior used in
    the paper; the per‑RV parameters become *IID* vectors of length *K*.
    """
    return dists.StructDist({
        # ----- unchanged scalar parameters --------------------------------
        **{k: dists.Normal(0, .1) for k in [
            'v0f','v1f','v2f','v3f','wf','bf',
            'v0i','v1i','v2i','v3i','wi','bi',
            'v0o','v1o','v2o','v3o','wo','bo',
            'v0d','v1d','v2d','v3d','wd','bd']},
        'beta0'   : dists.TruncatedNormal(0.0, 2.0, -8.0, 8.0),
        'beta1'   : dists.Gamma(2, 5),
        'beta'    : dists.Uniform(0, 1),
        # ----- K‑vector parameters (one value per realised measure) --------
        'gamma'   : dists.IID(dists.Beta(2, 5),    K),
        'xi'      : dists.IID(dists.Gamma(1, 1),   K),
        'phi'     : dists.IID(dists.Gamma(1, 1),   K),
        'tau1'    : dists.IID(dists.Normal(0, .1), K),
        'tau2'    : dists.IID(dists.Normal(0, .1), K),
        'sigmau2' : dists.IID(dists.Gamma(1, 5),   K),
    })

# ---------------------------------------------------------------------
# helper: gamma·RV for each particle
# ---------------------------------------------------------------------

def _gamma_dot_rv(gamma: np.ndarray, rv_row: np.ndarray) -> np.ndarray:
    """Compute \sum_k gamma_{p,k} * rv_{k,t} for every particle *p*.

    *gamma* shape: (P, K)   from theta['gamma']
    *rv_row* shape: (K,)     realised‑vol vector at time t
    Returns shape: (P,)
    """
    return np.einsum('pk,k->p', gamma, rv_row)

# =====================================================================
# 2.  In‑sample class with multi‑RV variance update
# =====================================================================

class RealRECH_2LSTM(SMC):
    """Multi‑RV version of the in‑sample SMC class (K realised measures).

    Only the variance recursion is modified; everything else is inherited
    unchanged from the original implementation.
    """
    def __init__(self, data, **kwargs):
        # *data* is a tuple (Y, RV) where
        #   Y  : (T,1)
        #   RV : (T,K)
        self.Y, self.RV = data
        super().__init__(data=data, **kwargs)

    # ---- unchanged helper ------------------------------------------------
    @staticmethod
    def safe_sqrt(x):
        return np.sqrt(np.maximum(x, 1e-10))

    def calculate_variance(self, omega, beta, _gamma_dummy, rv_dot, var_prev):
        min_var = np.percentile(self.Y**2, 1)
        return np.maximum(min_var, omega + beta*var_prev + rv_dot)

    # ------------------------------------------------------------------
    #                         log‑likelihood
    # ------------------------------------------------------------------
    def loglik(self, theta, get_v=False):
        N = self.T
        P = theta.shape[0]
        var    = np.zeros((N, P))
        var[0] = np.var(self.Y)
        omega  = np.zeros_like(var)
        omega[0] = theta['beta0']
        h      = np.zeros_like(var)
        c      = np.zeros_like(var)

        for n in range(N-1):
            # ---------- LSTM gates (unchanged, elided here) --------------
            # ...
            raw_w     = theta['beta0'] + theta['beta1'] * h[n+1]
            omega[n+1] = ut.relu(raw_w) * theta.get('ω_scale', 1.0)

            # --------- *** multi‑RV variance update *** -----------------
            rv_dot    = _gamma_dot_rv(theta['gamma'], self.RV[n])   # (P,)
            var[n+1]  = self.calculate_variance(
                            omega[n+1], theta['beta'], 1.0,
                            rv_dot, var[n])

        # original Gaussian likelihood on returns -------------------------
        llik_y = np.sum(stats.norm.logpdf(self.Y[1:], 0, self.safe_sqrt(var[1:])), 0)
        return llik_y


class RealRECHD_2LSTM(SMCD):
    """Multi‑RV out‑of‑sample (data‑annealing) class."""

    def __init__(self, data, **kwargs):
        # data = (Y_train, Y_test, RV_train, RV_test)
        self.Y_train, self.Y_test, self.RV_train, self.RV_test = data
        super().__init__(data=data, **kwargs)
        self.T = self.Y_test.shape[0]

    # helpers -------------------------------------------------------------
    @staticmethod
    def safe_sqrt(x):
        return np.sqrt(np.maximum(x, 1e-10))

    def calculate_variance(self, omega, beta, _gamma_dummy, rv_dot, var_prev):
        min_var = np.percentile(self.Y_train**2, 1)
        return np.maximum(min_var, omega + beta*var_prev + rv_dot)

    # ------------------------------------------------------------------
    def loglik(self, theta, t=None, lpyt=False):
        # concatenate training + test up to t
        Y  = np.concatenate((self.Y_train, self.Y_test[:t+1]))
        RV = np.concatenate((self.RV_train, self.RV_test[:t+1]))
        N  = Y.shape[0]
        P  = theta.shape[0]

        var    = np.zeros((N, P))
        var[0] = np.var(Y)
        omega  = np.zeros_like(var)
        omega[0] = theta['beta0']
        h      = np.zeros_like(var)
        c      = np.zeros_like(var)

        for n in range(N-1):
            # LSTM gates ... (unchanged)
            raw_w      = theta['beta0'] + theta['beta1'] * h[n+1]
            omega[n+1] = ut.relu(raw_w) * theta.get('ω_scale', 1.0)

            rv_dot     = _gamma_dot_rv(theta['gamma'], RV[n])
            var[n+1]   = self.calculate_variance(
                            omega[n+1], theta['beta'], 1.0,
                            rv_dot, var[n])
            # measurement‑eqn for RV (unchanged) ...

        llik_y = np.sum(stats.norm.logpdf(Y[1:], 0, self.safe_sqrt(var[1:])), 0)
        return llik_y
