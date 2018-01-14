import numpy as np


def gr_indicators(chain, alpha=0.05):
    """Calculate the Gelman Rubin indicator of convergence.  Also,
    calculate the interval based indicator presented in Brooks &
    Gelman 1998
    """
    nw, nstep, ndim = chain.shape
    # mean within each chain
    mean = chain.mean(axis=1)
    # variance within each chain
    var = chain.var(axis=1, ddof=1)
    # mean over chains of the variance within each chain
    W = var.mean(axis=0)
    # variance over chains of the mean within each chain, mutiplied by nstep
    B = nstep * mean.var(axis=0, ddof=1)
    # estimate of true variance: weighted sum of variances
    sigmasq = (1 - 1/nstep) * W + B/nstep
    # accounting for sampling variability
    V = sigmasq + B/(nw*nstep)
    R = V / W

    #Now do the interval based method
    p = [100.0*(alpha/2), 100.0*(1-alpha/2)]
    Wp = np.percentile(chain, p, axis=1)
    Wp = Wp[1,...] - Wp[0,...]
    Bp =  np.percentile(chain.reshape(nw*nstep, ndim), p, axis=0)
    Bp = Bp[1,...] - Bp[0,...]
    Rint = Bp/Wp.mean(axis=0)
    
    return R, Rint

def correlation_time(chain, window=None, c=10, fast=False):
    from emcee.autocorr import integrated_time
    nw, nstep, ndim = chain.shape
    x = np.mean(chain, axis=0)
    m = 0
    if window is None:
        for m in np.arange(10, nstep):
            tau = integrated_time(x, axis=0, fast=fast,
                                   window=m)
            if np.all(tau * c < m) and np.all(tau > 0):
                break
        window = m
    else:
        tau = integrated_time(x, axis=0, fast=fast,
                              window=window)
        
    if m == (nstep-1) or (np.any(tau < 0)):
        raise(ValueError)
    
    return tau, window
    
def raftery_lewis(chain, q, tol=None, p = 0.95):
    pass

def heidelberg_welch(chain, alpha):
    pass

def geweke(chain):
    pass
