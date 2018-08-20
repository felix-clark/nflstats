# some helper functions
import numpy as np

def corr_spearman(x, y, weights=None):
    """
    calculates the spearman rank coefficient between arrays, using weights.
    this version runs in linearithmic time (the naive n^2 is quite slow)
    it might also be a candidate for Cython.
    """
    assert(len(x) == len(y))
    mask = ~(np.isnan(x) | np.isnan(y))
    xvals = x[mask]
    yvals = y[mask]
    wts = weights[mask] if weights is not None else np.ones(xvals.shape)
    xsort = np.sort(xvals)
    ysort = np.sort(yvals)

    # keeping the indices will be useful for labelling with the ranks
    xargsorted = np.argsort(xvals)
    yargsorted = np.argsort(yvals)

    xrank = np.empty(xsort.shape)
    yrank = np.empty(ysort.shape)

    # # for testing:
    # xrank[:] = np.nan
    # yrank[:] = np.nan
    
    i=0
    while i < len(xsort):
        val = xsort[i]
        j = i+1
        while j < len(xsort) and xsort[j] == val:
            j += 1
        # if there are duplicate values, set the rank of all of them to half
        rk = 0.5*(i+j)
        for ix in xargsorted[i:j]:
            xrank[ix] = rk
        i = j

    i=0
    while i < len(ysort):
        val=ysort[i]
        j = i+1
        while j < len(ysort) and ysort[j] == ysort[i]:
            j += 1
        # if there are duplicate values, set the rank of all of them to half
        rk = 0.5*(i+j)
        for iy in yargsorted[i:j]:
            yrank[iy] = rk
        i = j

    # assert(not np.isnan(xrank).any())
    # assert(not np.isnan(yrank).any())

    # compute the weighted means of the ranks
    xmrk = np.average(xrank, weights=weights)
    ymrk = np.average(yrank, weights=weights)

    xvarrk = np.average((xrank-xmrk)**2, weights=weights)
    yvarrk = np.average((yrank-ymrk)**2, weights=weights)
    xycorrrk = np.average((xrank-xmrk)*(yrank-ymrk), weights=weights)
    
    result = xycorrrk / np.sqrt(xvarrk*yvarrk)
    return result
