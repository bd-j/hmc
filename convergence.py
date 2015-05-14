import numpy as np


def gr_indicator_chain(chain, alpha=0.05):
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

def correlation_time(chain):
    pass
    
def raftery_lewis(chain, q, tol=None, p = 0.95):
    pass

def heidelberg_welch(chain, alpha):
    pass

