import numpy as np
import scipy.stats as stats
import scipy.special as sp
import warnings

# Suppress specific warnings that might occur during calculation
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='overflow encountered')

def safe_sqrt(x):
    """Safer square root with input validation"""
    x_safe = np.array(x, dtype=np.float64, copy=True)
    x_safe[~np.isfinite(x_safe)] = 1e-8  # Handle NaN/Inf
    return np.sqrt(np.maximum(x_safe, 1e-10))

def safe_logpdf_t(x, df, loc, scale):
    """Robust t-distribution log PDF with full safeguards"""
    # Convert inputs to numpy arrays
    x = np.asarray(x, dtype=np.float64)
    df = np.maximum(np.asarray(df, dtype=np.float64), 2.1)
    loc = np.asarray(loc, dtype=np.float64)
    scale = np.maximum(np.asarray(scale, dtype=np.float64), 1e-10)
    
    # Handle invalid inputs
    valid_mask = np.isfinite(x) & np.isfinite(loc) & np.isfinite(scale)
    x_safe = np.where(valid_mask, x, loc)  # Replace invalid with loc
    
    # Standardization with clipping
    z = (x_safe - loc) / scale
    z_safe = np.clip(z, -50, 50)  # Conservative clipping
    
    try:
        # Vectorized calculation using scipy's implementation
        log_pdf = stats.t.logpdf(z_safe, df) - np.log(scale)
    except Exception:
        # Fallback using accurate gamma function calculations
        log_gamma_df_half = sp.gammaln(df/2)
        log_gamma_df_plus_half = sp.gammaln((df + 1)/2)
        
        log_term = -0.5 * np.log(df * np.pi) - log_gamma_df_half
        kernel = -((df + 1)/2) * np.log1p(z_safe**2 / df)
        
        log_pdf = log_term + log_gamma_df_plus_half + kernel - np.log(scale)
    
    # Ensure finite outputs
    return np.where(valid_mask, log_pdf, -1e20)  # Extreme penalty for invalid inputs

def calculate_variance(Y, omega, beta, gamma, rv, var_prev):
    """GARCH variance calculation with stationarity enforcement"""
    # Input validation
    omega = np.maximum(omega, 1e-10)
    beta = np.clip(beta, 0, 0.9999)
    gamma = np.clip(gamma, 0, 0.9999)
    
    # Enforce stationarity: beta + gamma < 1
    total = beta + gamma
    adj_factor = np.where(total > 0.9999, 0.9999 / total, 1.0)
    beta_adj = beta * adj_factor
    gamma_adj = gamma * adj_factor
    
    # Calculate baseline variance
    min_var = np.percentile(Y**2, 1, axis=0)  # 1st percentile per series
    var_next = omega + beta_adj*var_prev + gamma_adj*rv
    
    # Apply variance floor and ceiling
    return np.clip(var_next, min_var, 1e8)

# Activation functions (preserved from original)
def safe_sigmoid(x):
    x_safe = np.clip(x, -100, 100)
    pos = x_safe >= 0
    result = np.empty_like(x_safe)
    result[pos] = 1.0 / (1.0 + np.exp(-x_safe[pos]))
    exp_x = np.exp(x_safe[~pos])
    result[~pos] = exp_x / (1.0 + exp_x)
    return result

def safe_tanh(x):
    return np.tanh(np.clip(x, -100, 100))

def safe_relu(x):
    return np.maximum(0, x)