import numpy as np

def gelman_rubin_R(inchains, nsplit = 2):
    """chains: Ntheta X Nchain X Niterations"""
    ndim, n, m = inchains.shape
    #split the chains in half (or fourths or whatever)
    m *= nsplit
    n /= nsplit
    print(m,n)
    chains = inchains.reshape(ndim, m, n)
    B = chains.mean(axis = -1).var(-1) #between chain variance
    W = chains.var(axis = -1).mean(-1) #within chain variance
    print(B,W)
    varhat = (n-1.0)/n * W + 1.0/n * B
    Rhat = np.sqrt(varhat / W)
    return Rhat
