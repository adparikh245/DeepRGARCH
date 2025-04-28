import numpy as np
import scipy.stats as stats
import warnings

# Suppress specific warnings that might occur during calculation
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='overflow encountered')

def safe_sqrt(x):
    """Ensures values are positive before square root with improved safety"""
    # Handle arrays with potential NaN values
    if isinstance(x, np.ndarray):
        x_safe = np.copy(x)
        # Replace NaN and negative values with small positive values
        x_safe[~np.isfinite(x_safe)] = 1e-10
        return np.sqrt(np.maximum(x_safe, 1e-10))
    else:
        # Handle scalar values
        if not np.isfinite(x):
            return np.sqrt(1e-10)
        return np.sqrt(max(x, 1e-10))

def safe_logpdf_norm(x, loc, scale):
    """Safe version of normal log pdf with additional safeguards"""
    # Handle non-finite inputs
    if isinstance(x, np.ndarray):
        x_safe = np.copy(x)
        x_safe[~np.isfinite(x_safe)] = loc
    else:
        x_safe = loc if not np.isfinite(x) else x
    
    # Ensure positive scale with minimum threshold
    scale = np.maximum(scale, 1e-10)
    
    # Handle extremely large differences to prevent overflow
    z = (x_safe - loc) / scale
    z_safe = np.clip(z, -100, 100)
    
    # Use logarithm arithmetic for numerical stability
    log_2pi = 1.83787706640935  # pre-compute log(2π)
    log_pdf = -0.5 * log_2pi - np.log(scale) - 0.5 * (z_safe ** 2)
    
    # Handle potential remaining NaN values
    if isinstance(log_pdf, np.ndarray):
        log_pdf[~np.isfinite(log_pdf)] = -1000.0  # large negative number
    elif not np.isfinite(log_pdf):
        log_pdf = -1000.0
        
    return log_pdf

def safe_logpdf_t(x, df, loc, scale):
    """Safe version of t distribution log pdf with comprehensive safeguards"""
    # Handle non-finite inputs
    if isinstance(x, np.ndarray):
        x_safe = np.copy(x)
        x_safe[~np.isfinite(x_safe)] = loc
    else:
        x_safe = loc if not np.isfinite(x) else x
    
    # Ensure positive scale and valid df
    scale = np.maximum(scale, 1e-10)
    df = np.maximum(df, 2.1)  # Ensure df > 2 for finite variance
    
    # Standardize and clip to prevent overflow
    z = (x_safe - loc) / scale
    z_safe = np.clip(z, -100, 100)
    
    try:
        # Try using scipy's implementation first
        log_pdf = stats.t.logpdf(z_safe, df) - np.log(scale)
        
        # Handle any remaining NaN values
        if isinstance(log_pdf, np.ndarray):
            log_pdf[~np.isfinite(log_pdf)] = -1000.0
        elif not np.isfinite(log_pdf):
            log_pdf = -1000.0
            
        return log_pdf
        
    except Exception:
        # Fallback calculation with safeguards
        # Using more stable log-sum-exp trick for logarithms
        log_gamma_half_df_plus_half = np.log(np.sqrt(np.pi)) + np.log(0.5 * (df + 1))
        log_gamma_half_df = np.log(np.sqrt(np.pi)) + np.log(0.5 * df)
        
        # Calculate log(1 + z²/df) safely
        z_squared_over_df = np.clip(z_safe**2 / df, 0, 1e10)
        log_term = np.log1p(z_squared_over_df)  # more stable than log(1+x)
        
        log_pdf = log_gamma_half_df_plus_half - log_gamma_half_df - 0.5 * np.log(df) - \
                  np.log(scale) - ((df + 1) / 2) * log_term
        
        # Final safety check
        if isinstance(log_pdf, np.ndarray):
            log_pdf[~np.isfinite(log_pdf)] = -1000.0
        elif not np.isfinite(log_pdf):
            log_pdf = -1000.0
            
        return log_pdf

def calculate_variance(omega, beta, gamma, rv, var_prev):
    """Calculate variance with comprehensive safeguards against numerical issues"""
    # Ensure all inputs have reasonable values
    omega_safe = np.maximum(omega, 0)
    beta_safe = np.clip(beta, 0, 0.999)  # Stationary constraint
    gamma_safe = np.clip(gamma, 0, 0.999)  # Stationary constraint
    
    # Ensure rv is non-negative (realized variance should be positive)
    rv_safe = np.maximum(rv, 0)
    var_prev_safe = np.maximum(var_prev, 1e-10)
    
    # Calculate next variance with constraint to ensure stationarity
    # (beta + gamma should be < 1 for GARCH-type models)
    if np.any(beta_safe + gamma_safe >= 1):
        beta_gamma_sum = beta_safe + gamma_safe
        scaling = np.ones_like(beta_gamma_sum)
        mask = beta_gamma_sum > 0
        scaling[mask] = 0.99 / beta_gamma_sum[mask]
        beta_safe = beta_safe * scaling
        gamma_safe = gamma_safe * scaling
    
    # Final variance calculation with safety floor
    var_next = omega_safe + beta_safe * var_prev_safe + gamma_safe * rv_safe
    
    # Ensure variance is always positive and reasonably bounded
    return np.clip(var_next, 1e-10, 1e10)

# Add a safe sigmoid function to replace any in your utils module
def safe_sigmoid(x):
    """Numerically stable sigmoid function"""
    # Clip input to prevent overflow
    x_safe = np.clip(x, -100, 100)
    # Use exp directly for positive values, and 1/(1+exp(-x)) for negative
    # This prevents overflow for large positive x
    pos = x_safe >= 0
    result = np.empty_like(x_safe)
    result[pos] = 1.0 / (1.0 + np.exp(-x_safe[pos]))
    # For negative values, use exp(x)/(1+exp(x)) to prevent underflow
    neg = ~pos
    exp_x = np.exp(x_safe[neg])
    result[neg] = exp_x / (1.0 + exp_x)
    return result

# Safe tanh implementation
def safe_tanh(x):
    """Numerically stable tanh function"""
    # Clip input to prevent overflow
    x_safe = np.clip(x, -100, 100)
    return np.tanh(x_safe)

# Safe ReLU implementation
def safe_relu(x):
    """Numerically stable ReLU function"""
    return np.maximum(0, x)

# Monkey patch the functions in scipy.stats to use our safe versions
original_norm_logpdf = stats.norm.logpdf
original_t_logpdf = stats.t.logpdf

def safe_norm_logpdf_wrapper(x, loc=0, scale=1):
    return safe_logpdf_norm(x, loc, scale)

def safe_t_logpdf_wrapper(x, df, loc=0, scale=1):
    return safe_logpdf_t(x, df, loc, scale)

# Replace standard functions with safe versions
stats.norm.logpdf = safe_norm_logpdf_wrapper
stats.t.logpdf = safe_t_logpdf_wrapper

# Add a function to restore original functions if needed
def restore_original_functions():
    stats.norm.logpdf = original_norm_logpdf
    stats.t.logpdf = original_t_logpdf