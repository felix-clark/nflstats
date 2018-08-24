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


def get_k_partition_boundaries(data, k):
    if k >= len(data):
        print('error: need k less than the size of the data')
        return None
    sortdata = np.sort(data)
    # gaps is an array of size N-1 listing
    gaps = np.array(sortdata[1:]) - np.array(sortdata[:-1])
    # part_idsx are the indices of the k largest gaps
    part_idxs = np.argpartition(gaps, k)[-k:] # [::-1] # don't think we really need to re-order these
    # define the boundaries as the means
    part_boundaries = [0.5*(sortdata[i] + sortdata[i+1]) for i in part_idxs]
    return np.sort(part_boundaries)


def get_team_abbrev(full_team_name, team_abbrevs):
    up_name = full_team_name.upper().strip()
    for ta in team_abbrevs:
        un_split = up_name.split(' ')
        # these are typically the first letter of the 1st two words:
        # e.g. KC, TB, NE, ...
        # can also be 1st letter of 3 words: LAR, LAC, ...
        if ''.join([w[0] for w in un_split[:len(ta)]]) == ta:
            # print full_team_name, ta
            return ta
        # the other class is the 1st 3 letters of the city
        if up_name[:3] == ta:
            # print full_team_name, ta
            return ta
    # it's possible the string is of the following form: Los Angeles (LAR)
    # in which case the abbreviation already exists
    if len(up_name) > 5:
        if up_name[-5] == '(' and up_name[-1] == ')':
            return up_name[-4:-1]
    
    logging.error('could not find abbreviation for {}'.format(full_team_name))
    
def rm_name_suffix(name):
    spln = name.split(' ')
    last = spln[-1].strip('.')
    if last.lower() in ['jr', 'sr'] or all((l in 'IVX' for l in last)):
        sfl = ' '.join(spln[:-1])
        return sfl
    return name
