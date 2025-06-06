from __future__ import division, print_function

from collections import OrderedDict  # see prior
import numpy as np
import numpy.random as random
import scipy.stats as stats
from scipy.linalg import cholesky, solve_triangular, inv

HALFLOG2PI = 0.5 * np.log(2. * np.pi)


class ProbDist(object):
    """Base class for probability distributions.

    To define a probability distribution class, subclass ProbDist, and define
    methods:

    * ``logpdf(self, x)``: the log-density at point x
    * ``rvs(self, size=None)``: generates *size* variates from distribution
    * ``ppf(self, u)``: the inverse CDF function at point u

    and attributes:

        * ``dim``: dimension of variates (default is 1)
        * ``dtype``: the dtype of inputs/outputs arrays (default is 'float64')

    """
    dim = 1  # distributions are univariate by default
    dtype = 'float64'  # diacstributions are continuous by default

    def shape(self, size):
        if size is None:
            return None
        else:
            return (size,) if self.dim == 1 else (size, self.dim)

    def logpdf(self, x):
        raise NotImplementedError

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def rvs(self, size=None):
        raise NotImplementedError

    def ppf(self, u):
        raise NotImplementedError

##############################
# location-scale distributions
##############################


class LocScaleDist(ProbDist):
    """Base class for location-scale distributions.
    """
    def __init__(self, loc=0., scale=1.):
        self.loc = loc
        self.scale = scale


class Normal(LocScaleDist):
    """N(loc, scale^2) distribution.
    """
    def rvs(self, size=None):
        return random.normal(loc=self.loc, scale=self.scale,
                             size=self.shape(size))

    def logpdf(self, x):
        return stats.norm.logpdf(x, loc=self.loc, scale=self.scale)

    def ppf(self, u):
        return stats.norm.ppf(u, loc=self.loc, scale=self.scale)

    def posterior(self, x, sigma=1.):
        """Model is X_1,...,X_n ~ N(theta, sigma^2), theta~self, sigma fixed.
        """
        pr0 = 1. / self.scale**2  # prior precision
        prd = x.size / sigma**2  # data precision
        varp = 1. / (pr0 + prd)  # posterior variance
        mu = varp * (pr0 * self.loc + prd * x.mean())
        return Normal(loc=mu, scale=np.sqrt(varp))


class Logistic(LocScaleDist):
    """Logistic(loc, scale) distribution.
    """
    def rvs(self, size=None):
        return random.logistic(loc=self.loc, scale=self.scale,
                               size=self.shape(size))

    def logpdf(self, x):
        return stats.logistic.logpdf(x, loc=self.loc, scale=self.scale)

    def ppf(self, u):
        return stats.logistic.ppf(u, loc=self.loc, scale=self.scale)


class Laplace(LocScaleDist):
    """Laplace(loc,scale) distribution.
    """

    def rvs(self, size=None):
        return random.laplace(loc=self.loc, scale=self.scale,
                              size=self.shape(size))

    def logpdf(self, x):
        return stats.laplace.logpdf(x, loc=self.loc, scale=self.scale)

    def ppf(self, u):
        return stats.laplace.ppf(u, loc=self.loc, scale=self.scale)


################################
# Other continuous distributions
################################

class Beta(ProbDist):
    """Beta(a,b) distribution.
    """
    def __init__(self, a=1., b=1.):
        self.a = a
        self.b = b

    def rvs(self, size=None):
        return random.beta(self.a, self.b, size=size)

    def logpdf(self, x):
        return stats.beta.logpdf(x, self.a, self.b)

    def ppf(self, x):
        return stats.beta.ppf(x, self.a, self.b)


class Gamma(ProbDist):
    """Gamma(a,b) distribution, scale=1/b.
    """
    def __init__(self, a=1., b=1.):
        self.a = a
        self.b = b
        self.scale = 1. / b

    def rvs(self, size=None):
        return random.gamma(self.a, scale=self.scale, size=size)

    def logpdf(self, x):
        return stats.gamma.logpdf(x, self.a, scale=self.scale)

    def ppf(self, u):
        return stats.gamma.ppf(u, self.a, scale=self.scale)

    def posterior(self, x):
        """Model is X_1,...,X_n ~ N(0, 1/theta), theta ~ Gamma(a, b)"""
        return Gamma(a=self.a + 0.5 * x.size,
                     b=self.b + 0.5 * np.sum(x**2))


class InvGamma(ProbDist):
    """Inverse Gamma(a,b) distribution.
    """
    def __init__(self, a=1., b=1.):
        self.a = a
        self.b = b

    def rvs(self, size=None):
        return stats.invgamma.rvs(self.a, scale=self.b, size=size)

    def logpdf(self, x):
        return stats.invgamma.logpdf(x, self.a, scale=self.b)

    def ppf(self, u):
        return stats.invgamma.ppf(u, self.a, scale=self.b)

    def posterior(self, x):
        " Model is X_1,...,X_n ~ N(0, theta), theta ~ InvGamma(a, b) "
        return InvGamma(a=self.a + 0.5 * x.size,
                        b=self.b + 0.5 * np.sum(x**2))


class Uniform(ProbDist):
    """Uniform([a,b]) distribution.
    """
    def __init__(self, a=0, b=1.):
        self.a = a
        self.b = b
        self.scale = b - a

    def rvs(self, size=None):
        return random.uniform(low=self.a, high=self.b, size=size)

    def logpdf(self, x):
        return stats.uniform.logpdf(x, loc=self.a, scale=self.scale)

    def ppf(self, u):
        return stats.uniform.ppf(u, loc=self.a, scale=self.scale)


class Student(ProbDist):
    """Student distribution.
    """
    def __init__(self, df=3., loc=0., scale=1.):
        self.df = df
        self.loc = loc
        self.scale = scale

    def rvs(self, size=None):
        return stats.t.rvs(self.df, loc=self.loc, scale=self.scale, size=size)

    def logpdf(self, x):
        return stats.t.logpdf(x, self.df, loc=self.loc, scale=self.scale)

    def ppf(self, u):
        return stats.t.ppf(u, self.df, loc=self.loc, scale=self.scale)


class Dirac(ProbDist):
    """Dirac mass.
    """
    def __init__(self, loc=0.):
        self.loc = loc

    def rvs(self, size=None):
        if isinstance(self.loc, np.ndarray):
            return self.loc.copy()
            # seems safer to make a copy here
        else:  # a scalar
            N = 1 if size is None else size
            return np.full(N, self.loc)

    def logpdf(self, x):
        return np.where(x==self.loc, 0., -np.inf)

    def ppf(self, u):
        return self.rvs(size=u.shape[0])


class TruncNormal(ProbDist):
    """Normal(mu, sigma^2) truncated to [a, b] interval.
    """
    def __init__(self, mu=0., sigma=1., a=0., b=1.):
        self.mu = mu
        self.sigma = sigma
        self.a = a
        self.b = b
        self.au = (a - mu) / sigma
        self.bu = (b - mu) / sigma

    def rvs(self, size=None):
        return stats.truncnorm.rvs(self.au, self.bu, loc=self.mu,
                                   scale=self.sigma, size=size)

    def logpdf(self, x):
        return stats.truncnorm.logpdf(x, self.au, self.bu, loc=self.mu,
                                      scale=self.sigma)

    def ppf(self, u):
        return stats.truncnorm.ppf(u, self.au, self.bu, loc=self.mu,
                                   scale=self.sigma)

    def posterior(self, x, s=1.):
        """Model is X_1,...,X_n ~ N(theta, s^2), theta~self, s fixed"""
        pr0 = 1. / self.sigma**2  # prior precision
        prd = x.size / s**2  # data precision
        varp = 1. / (pr0 + prd)  # posterior variance
        mu = varp * (pr0 * self.mu + prd * x.mean())
        return TruncNormal(mu=mu, sigma=np.sqrt(varp), a=self.a, b=self.b)

########################
# Discrete distributions
########################


class DiscreteDist(ProbDist):
    """Base class for discrete probability distributions.
    """
    dtype = 'int64'


class Poisson(DiscreteDist):
    """Poisson(rate) distribution.
    """
    def __init__(self, rate=1.):
        self.rate = rate

    def rvs(self, size=None):
        return random.poisson(self.rate, size=size)

    def logpdf(self, x):
        return stats.poisson.logpmf(x, self.rate)

    def ppf(self, u):
        return stats.poisson.ppf(u, self.rate)


class Binomial(DiscreteDist):
    """Binomial(n,p) distribution.
    """
    def __init__(self, n=1, p=0.5):
        self.n = n
        self.p = p

    def rvs(self, size=None):
        return random.binomial(self.n, self.p, size=size)

    def logpdf(self, x):
        return stats.binom.logpmf(x, self.n, self.p)

    def ppf(self, u):
        return stats.binom.ppf(u, self.n, self.p)


class Geometric(DiscreteDist):
    """Geometric(p) distribution.
    """
    def __init__(self, p=0.5):
        self.p = p

    def rvs(self, size=None):
        return random.geometric(self.p, size=size)

    def logpdf(self, x):
        return stats.geom.logpmf(x, self.p)

    def ppf(self, u):
        return stats.geom.ppf(u, self.p)

class NegativeBinomial(DiscreteDist):
    """Negative Binomial distribution.

    Parameters
    ----------
    n:  int, or array of ints
        number of failures until the experiment is run
    p:  float, or array of floats
        probability of success

    Note:
        Returns the distribution of the number of successes: support is
        0, 1, ...

    """
    def rvs(self, size=None):
        return random.negative_binomial(self.n, self.p, size=size)

    def logpdf(self, x):
        return stats.nbinom.logpmf(x, self.p, self.n)

    def ppf(self, u):
        return stats.nbinom.ppf(u, self.p, self.n)


class Categorical(DiscreteDist):
    """Categorical distribution.

    Parameter
    ---------
    p:  (k,) or (N,k) float array
        vector(s) of k probabilities that sum to one
    """
    def __init__(self, p=None):
        self.p = p

    def logpdf(self, x):
        return np.log(self.p[x])

    def rvs(self, size=None):
        if self.p.ndim == 1:
            N = 1 if size is None else size
            u =random.rand(N)
            return np.searchsorted(np.cumsum(self.p), u)
        else:
            N = self.p.shape[0] if size is None else size
            u = random.rand(N)
            cp = np.cumsum(self.p, axis=1)
            return np.array([np.searchsorted(cp[i], u[i])
                             for i in range(N)])

class DiscreteUniform(DiscreteDist):
    """Discrete uniform distribution.

    Parameters
    ----------
    lo, hi: int
        support is lo, lo + 1, ..., hi - 1

    """

    def __init__(self, lo=0, hi=2):
        self.lo, self.hi = lo, hi
        self.log_norm_cst = np.log(hi - lo)

    def logpdf(self, x):
        return np.where((x >= self.lo) & (x<self.hi), -self.log_norm_cst, -np.inf) 

    def rvs(self, size=None):
        return random.randint(self.lo, high=self.hi, size=size)

#########################
# distribution transforms
#########################

class TransformedDist(ProbDist):
    """Base class for transformed distributions.

    A transformed distribution is the distribution of Y=f(X) for a certain
    function f, and a certain (univariate) base distribution for X.
    To define a particular class of transformations, sub-class this class, and
    define methods:

        * f(self, x): function f 
        * finv(self, x): inverse of function f
        * logJac(self, x): log of Jacobian of the inverse of f 

    """

    def __init__(self, base_dist):
        self.base_dist = base_dist

    def error_msg(self, method):
        return 'method %s not defined in class %s' % (method, self.__class__)

    def f(self, x):
        raise NotImplementedError(self.error_msg('f'))

    def finv(self, x):
        """ Inverse of f."""
        raise NotImplementedError(self.error_msg('finv'))

    def logJac(self, x):
        """ Log of Jacobian.

        Obtained by differentiating finv, and then taking the log."""
        raise NotImplementedError(self.error_msg('logJac'))

    def rvs(self, size=None):
        return self.f(self.base_dist.rvs(size=size))

    def logpdf(self, x):
        return self.base_dist.logpdf(self.finv(x)) + self.logJac(x)

    def ppf(self, u):
        return self.f(self.base_dist.ppf(u))


class LinearD(TransformedDist):
    """Distribution of Y = a*X + b.

    See TransformedDist.

    Parameters
    ----------
    base_dist: ProbDist
        The distribution of X

    a, b: float (a should be != 0)
    """
    def __init__(self, base_dist, a=1., b=0.):
        self.a, self.b = a, b
        self.base_dist = base_dist

    def f(self, x):
        return self.a * x + self.b

    def finv(self, x):
        return (x - self.b) / self.a

    def logJac(self, x):
        return -np.log(self.a)


class LogD(TransformedDist):
    """Distribution of Y = log(X).

    See TransformedDist.

    Parameters
    ----------
    base_dist: ProbDist
        The distribution of X

    """
    def f(self, x):
        return np.log(x)

    def finv(self, x):
        return np.exp(x)

    def logJac(self, x):
        return x


class LogitD(TransformedDist):
    """Distributions of Y=logit((X-a)/(b-a)).

    See base class `TransformedDist`.

    Parameters
    ----------
    base_dist: ProbDist
        The distribution of X
    a, b: float
        interval [a, b] is the support of base_dist

    """

    def __init__(self, base_dist, a=0., b=1.):
        self.a, self.b = a, b
        self.base_dist = base_dist

    def f(self, x):
        p = (x - self.a) / (self.b - self.a)
        return np.log(p / (1. - p))  # use built-in?

    def finv(self, x):
        return self.a + (self.b - self.a) / (1. + np.exp(-x))

    def logJac(self, x):
        return np.log(self.b - self.a) + x - 2. * np.log(1. + np.exp(x))


############################
# Multivariate distributions
############################

class MvNormal(ProbDist):
    """Multivariate Normal distribution.

    Parameters
    ----------
    loc: ndarray
        location parameter (default: 0.)
    scale: ndarray
        scale parameter (default: 1.)
    cov: (d, d) ndarray
        covariance matrix (default: identity, with dim determined by loc)

    Notes
    -----
    The dimension d is determined either by argument ``cov`` (if it's a dxd 
    array), or by argument loc (if ``cov`` is not specified). In the latter 
    case, the covariance matrix is set to the identity matrix. 

    If ``scale`` is set to ``1.`` (default value), we use the standard 
    parametrisation of a Gaussian, with mean ``loc`` and covariance 
    matrix ``cov``. Otherwise::

        x = dists.MvNormal(loc=m, scale=s, cov=Sigma).rvs(size=30)

    is equivalent to::

        x = m + s * dists.MvNormal(cov=Sigma).rvs(size=30)

    The idea is that they are many cases when we may want to pass
    varying means and scales (but a fixed correlation matrix). Note that
    ``cov`` does not need to be a correlation matrix; e.g.::

        MvNormal(loc=m, scale=s, cov=C)

    correspond to N(m, diag(s)*C*diag(s))

    In addition, note that m and s may be (N, d) vectors;
    i.e for each n=1...N we have a different mean, and a different scale.
    """

    def __init__(self, loc=0., scale=1., cov=None):
        self.cov = np.eye(loc.shape[-1]) if cov is None else cov
        self.loc = loc
        self.scale = scale
        err_msg = 'mvnorm: argument cov must be a dxd ndarray, \
                with d>1, defining a symmetric positive matrix'
        try:
            self.L = cholesky(self.cov, lower=True)  # L*L.T = cov
            self.halflogdetcor = np.sum(np.log(np.diag(self.L)))
        except:
            raise ValueError(err_msg)
        assert self.cov.shape == (self.dim, self.dim), err_msg

    @property
    def dim(self):
        return self.cov.shape[0]

    def linear_transform(self, z):
        return self.loc + self.scale * np.dot(z, self.L.T)

    def logpdf(self, x):
        z = solve_triangular(self.L, np.transpose((x - self.loc) / self.scale),
                             lower=True)
        # z is dxN, not Nxd
        if np.asarray(self.scale).ndim == 0:
            logdet = self.dim * np.log(self.scale)
        else:
            logdet = np.sum(np.log(self.scale), axis=-1)
        logdet += self.halflogdetcor
        return - 0.5 * np.sum(z * z, axis=0) - logdet - self.dim * HALFLOG2PI

    def rvs(self, size=None):
        if size is None:
            sh = np.broadcast(self.loc, self.scale).shape
            # sh=() when both loc and scale are scalars
            N = 1 if len(sh) == 0 else sh[0]
        else:
            N = size
        z = stats.norm.rvs(size=(N, self.dim))
        return self.linear_transform(z)

    def ppf(self, u):
        """
        Note: if dim(u) < self.dim, the remaining columns are filled with 0
        Useful in case the distribution is partly degenerate.
        """
        N, du = u.shape
        if du < self.dim:
            z = np.zeros((N, self.dim))
            z[:, :du] = stats.norm.ppf(u)
        else:
            z = stats.norm.ppf(u)
        return self.linear_transform(z)

    def posterior(self, x, Sigma=None):
        """Posterior for model: X1, ..., Xn ~ N(theta, Sigma), theta ~ self.

        Parameters
        ----------
        x: (n, d) ndarray
            data
        Sigma: (d, d) ndarray
            covariance matrix in the model (default: identity matrix)

        Notes
        -----
        Scale must be set to 1.
        """
        if self.scale != 1.:
            raise ValueError('posterior of MvNormal: scale must be one.')
        n = x.shape[0]
        Sigma = np.eye(self.dim) if Sigma is None else Sigma
        Siginv = inv(Sigma)
        covinv = inv(self.cov)
        Qpost = covinv + n * Siginv
        Sigpost = inv(Qpost)
        m = np.full(self.dim, self.loc) if np.isscalar(self.loc) else self.loc
        mupost = Sigpost @ (m @ covinv + Siginv @ np.sum(x, axis=0))
        # m @ covinv works wether the shape of m is (N, d) or (d)
        return MvNormal(loc=mupost, cov=Sigpost)

##################################
# product of independent dists


class IndepProd(ProbDist):
    """Product of independent univariate distributions.

    The inputs/outputs of IndeProd are numpy ndarrays of shape (N,d), or (d),
    where d is the number of univariate distributions that are
    passed as arguments.

    Parameters
    ----------
    dists: list of `ProbDist` objects
        The probability distributions of each component

    Example
    -------
    To define a bivariate distribution::

        biv_dist = IndepProd(Normal(scale=2.), Gamma(2., 3.))
        samples = biv_dist.rvs(size=9)  # returns a (9, 2) ndarray

    Note
    ----
    This is used mainly to define multivariate state-space models,
    see module `state_space_models`. To specify a prior distribution, you
    should use instead `StructDist`.

    """
    def __init__(self, *dists):
        self.dists = dists
        self.dim = len(dists)
        if all(d.dtype == 'int64' for d in dists):
            self.dtype = 'int64'
        else:
            self.dtype = 'float64'

    def logpdf(self, x):
        return sum([d.logpdf(x[..., i]) for i, d in enumerate(self.dists)])
        # ellipsis: in case x is of shape (d) instead of (N, d)

    def rvs(self, size=None):
        return np.stack([d.rvs(size=size) for d in self.dists], axis=1)

    def ppf(self, u):
        return np.stack([d.ppf(u[..., i]) for i, d in enumerate(self.dists)],
                        axis=1)

def IID(law, k):
    """Joint distribution of k iid (independent and identically distributed) variables. 

    Parameters
    ----------
    law:  ProbDist object
        the univariate distribution of each component
    k: int (>= 2)
        number of components
    """
    return IndepProd(*[law for _ in range(k)])

###################################
# structured array distributions
# (mostly to define prior distributions)
###################################

class Cond(ProbDist):
    """Conditional distributions.

    A conditional distribution acts as a function, which takes as input the
    current value of the samples, and returns a probability distribution. 

    This is used to specify conditional distributions in `StructDist`; see the
    documentation of that class for more details. 
    """
    def __init__(self, law, dim=1, dtype='float64'):
        self.law = law
        self.dim = dim
        self.dtype = dtype

    def __call__(self, x):
        return self.law(x)

class StructDist(ProbDist):
    """A distribution such that inputs/outputs are structured arrays.

    A structured array is basically a numpy array with named fields.
    We use structured arrays to represent particles that are
    vectors of (named) parameters; see modules :mod:`smc_samplers`
    and :mod:`mcmc`. And we use StructDist to define prior distributions
    with respect to such parameters.

    To specify a distribution such that parameters are independent,
    we pass a dictionary::

        prior = StructDist({'mu':Normal(), 'sigma':Gamma(a=1., b=1.)})
        # means mu~N(0,1), sigma~Gamma(1, 1) independently
        x = prior.rvs(size=30)  # returns a stuctured array of length 30
        print(x['sigma'])  # prints the 30 values for sigma

    We may also define a distribution using a chain rule decomposition.
    For this, we pass an ordered dict, since the order of components
    become relevant::

        chain_rule = OrderedDict()
        chain_rule['mu'] = Normal()
        chain_rule['tau'] = Cond(lambda x: Normal(loc=x['mu'])
        prior = StructDist(chain_rule)
        # means mu~N(0,1), tau|mu ~ N(mu,1)

    In the third line, ``Cond`` is a ``ProbDist`` class that represents
    a conditional distribution; it is initialized with a function that
    returns for each ``x`` a distribution that may depend on fields in ``x``.

    Parameters
    ----------
    laws: dict or ordered dict (as explained above)
        keys are parameter names, values are `ProbDist` objects

    """

    def __init__(self, laws):
        if isinstance(laws, OrderedDict):
            self.laws = laws
        elif isinstance(laws, dict):
            self.laws = OrderedDict([(key, laws[key])
                                     for key in sorted(laws.keys())])
        else:
            raise ValueError('recdist class requires a dict or'
                             ' an ordered dict to be instantiated')
        self.dtype = []
        for key, law in self.laws.items():
            if law.dim == 1:
                typ = (key, law.dtype)  # avoid FutureWarning about (1,) fields
            else:
                typ = (key, law.dtype, law.dim)
            self.dtype.append(typ)

    def logpdf(self, theta):
        l = 0.
        for par, law in self.laws.items():
            cond_law = law(theta) if callable(law) else law
            l += cond_law.logpdf(theta[par])
        return l

    def rvs(self, size=1):  # Default for size is 1, not None
        out = np.empty(size, dtype=self.dtype)
        for par, law in self.laws.items():
            cond_law = law(out) if callable(law) else law
            out[par] = cond_law.rvs(size=size)
        return out

    def ppf(self, theta):
        out = np.empty(size, dtype=self.dtype)
        for par, law in self.laws.items():
            cond_law = law(out) if callable(law) else law
            out[par] = cond_law.ppf(theta[par])
        return out
