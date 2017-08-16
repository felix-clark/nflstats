#!/usr/bin/env python

import numpy as np
from numpy import sqrt, log, exp, pi
from scipy.special import gammaln, betaln, comb, digamma
from scipy.misc import factorial
import scipy.optimize as opt
import matplotlib.pyplot as plt

# non-negative discrete distributions

def poisson( k, lamb ):
    # return (lamb**x/factorial(x)) * exp(-lamb)
    return exp( log_poisson( k, lamb ) )

def log_poisson( k, lamb ):
    return k*log(lamb) - lamb - gammaln( k+1 )

def geometric( k, p ):
    return p*(1-p)**k

def neg_binomial( k, r, p ):
    # return exp( log_neg_binomial( k, r, p ) )
    return ( k >= 0 ) * ( comb( k + r - 1, k) * p**k * (1-p)**r )

def log_neg_binomial( k, r, p ):
    # return -gammaln( k+1 ) + gammaln( k + r ) - gammaln( r ) + k*log(p) + r*log(1-p)
    if k < 0: return -np.inf
    if k == 0: return r*log(1-p)
    return k*log(p) + r*log(1-p) - log(k) - betaln( k, r )

# def sum_log_neg_binomial( ks, r, p ):
#     if not ks:
#         'empty list to sum_log_neg_binomial'
#         return 0
#     N = len( ks )
#     return N*( r*log( 1-p ) - gammaln( r ) ) \
#         + sum( ( gammaln( k+r ) - gammaln( k+1 ) + k*log(p) for k in ks) )

def grad_sum_log_neg_binomial( ks, r, p ):
    if not ks:
        'empty list to grad_sum_log_neg_binomial'
        return 0
    N = len( ks )
    dldp = sum( ks ) / p - N*r/(1-p)
    dldr = N*(log(1-p) - digamma(r)) + sum( ( digamma(k+r) for k in ks ) )
    return (dldr, dldp)


## gaussians are not good for integer support, or at least the normalization must be changed
# def gaussian( x, mu, sigma ):
#     return exp( -(x - mu)**2 / (2*sigma**2) ) / sqrt( 2*pi*sigma**2 )
# def log_gaussian( x, mu, sigma ):
#     return -(x-mu)**2 / (2*sigma**2) - 0.5* log( 2*pi*sigma**2 )

def sum_log_gaussian_int( ks, bounds, mu, sigma):
    N = len( ks )
    bounds = (low_bound,up_bound)
    # norm_factor = -N*0.5*log( 2*pi*sigma**2 ) # only holds in continuous case
    c = 1.0/(2.0*sigma**2)
    norm_factor = log( sum(
        ( exp(-c*(k-mu)**2) for k in range(*bounds) )
    ) )
    print norm_factor, ' -- ', N*0.5*log( 2*pi*sigma**2 ) # temporary just to see how different
    sum_ll = - sum( ( c*(k-mu)**2 for k in ks ) )
    return sum_ll - norm_factor

def grad_sum_log_gaussian_int( ks, bounds, mu, sigma):
    N = len( ks )
    bounds = (low_bound,up_bound)
    # norm_factor = -N*0.5*log( 2*pi*sigma**2 ) # only holds in continuous case
    c = 1.0/(2.0*sigma**2)
    norm_factor = log( sum(
        ( exp(-c*(k-mu)**2) for k in range(*bounds) )
    ) )
    # sum_ll = - sum( ( c*(k-mu)**2 for k in ks ) )
    sum_ll_dmu = sum( ( (k-mu)/sigma**2 for k in ks ) )
    sum_ll_dsigma = # finish this and norm terms later
    return (dll_dmu, dll_dsigma)


def to_poisson( data=[] ):
    if not data:
        print 'error: empty data set'
        exit(1)
    n = len( data )
    mu = float( sum( data ) ) / n
    err_mu = sqrt( mu / n )
    log_L_per_ndf = mu * ( log(mu) - 1 ) - sum( (gammaln( x+1 ) for x in data) ) / n
    return (mu, err_mu), log_L_per_ndf

def to_geometric( data=[] ):
    if not data:
        print 'error: empty data set'
        exit(1)
    n = len( data )
    p = float(n) / ( n + sum( data ) )
    err_p = sqrt( p**2 * (1-p) / n )
    log_L_per_ndf = log( p ) + log( 1-p )*(1-p)/p
    return (p, err_p), log_L_per_ndf

# note that these is an equation that can be solved for r, but it is not closed-form.
# it would reduce the dimensionality of the fit algorithm, however.
# then p is easily expressed in terms of the mean and r.
def to_neg_binomial( data=[] ):
    if not data:
        print 'error: empty data set'
        exit(1)
    for x in data:
        if x < 0:
            print 'warning: negative value in data set. negative binomial may not be appropriate.'
    n = len( data )
    mean = float( sum( data ) ) / n
    rp0 = (mean, 0.5) # initial guess. r > 0 and 0 < p < 1
    allowed_methods = ['L-BFGS-B', 'TNC', 'SLSQP'] # these are the only ones that can handle bounds. they can also all handle jacobians. none of them can handle hessians.
    # only LBFGS returns Hessian, in form of "LbjgsInvHessProduct"
    method = allowed_methods[0]
    
    func = lambda pars: sum( ( - log_neg_binomial( k, *pars ) for k in data ) )
    # func = lambda pars: sum( [ - log_neg_binomial( k, *pars ) for k in data ] )
    grad = lambda pars: - np.array( grad_sum_log_neg_binomial( data, *pars ) )
    opt_result = opt.minimize( func, rp0, method=method, jac=grad, bounds=[(0,None),(0,1)] )
    # print opt_result.message
    if not opt_result.success:
        print 'negative binomial fit did not succeed.'
    r,p = opt_result.x
    print 'jacobian = ', opt_result.jac # should be zero, or close to it
    cov = opt_result.hess_inv
    cov_array = cov.todense()  # dense array
    err_r = sqrt(cov_array[0][0])
    err_p = sqrt(cov_array[1][1]) # ?
    neg_ll = opt_result.fun
    return (r,err_r),(p,err_p),-neg_ll/n
    
    
def to_gaussian( data=[] ):
    if not data:
        print 'error: empty data set'
        exit(1)
    if len( data ) == 1:
        print 'need more than 1 data point to fit gaussian'
        exit(1)
    n = len( data )
    mu = float( sum( data ) ) / n
    var = sum( ( (x - mu)**2 for x in data ) ) / (n-1) # sample (as opposed to population) variance
    sigma = sqrt( var )
    err_mu = sigma / sqrt(n)
    err_sigma = sigma / sqrt( 2*(n-1) ) # report sample standard deviation
    log_L_per_ndf = (1 - log(2*pi*var))/2
    return (mu, err_mu), (sigma, err_sigma), log_L_per_ndf

# appropriate with non-negative integer data
def plot_counts( data=[], label='', norm=False, fits=['poisson', 'neg_binomial'] ):
    ndata = len( data )
    maxdata = max( data )

    # # probably don't need to save all return values
    entries, bin_edges, patches = plt.hist( data, bins=np.arange(-0.0,maxdata+2,1),
                                            # range=[-0.5, maxtds+1.5],
                                            align='left',
                                            normed=norm,
                                            label=label
    )
    
    yerrs = [ sqrt( x / ndata ) for x in entries ] if norm else [ sqrt( x ) for x in entries ]
    plt.errorbar( np.arange(0,maxdata+1), entries, yerr=yerrs, align='left', fmt='none', color='black' )
    
    xfvals = np.linspace(0, maxdata+1, 1000)

    # do likelihood fits for each type

    if 'geometric' in fits:             
        (p,errp),logl = to_geometric( data )
        print '  Geometric fit:'
        print '    p = {:.3} '.format(p) + u'\u00B1' + ' {:.2}'.format( errp )
        print '    log(L)/N = {:.3}'.format( logl )    
        plt.plot(xfvals, ndata*geometric( xfvals, p ), 'g-', lw=2)

    if 'poisson' in fits:
        (mu,errmu),logl = to_poisson( data )
        print '  Poisson fit:'
        print '    ' + u'\u03BC' + ' = {:.3} '.format(mu) +  u'\u00B1' + ' {:.2}'.format( errmu )
        print '    log(L)/N = {:.3}'.format( logl )
        plt.plot(xfvals, ndata*poisson( xfvals, mu ), 'r-', lw=2)

    if 'neg_binomial' in fits:
        (r,errr),(p,errp),logl = to_neg_binomial( data )
        print '  Negative binomial fit:'
        print '    r = {:.3} '.format(r) + u'\u00B1' + ' {:.2}'.format( errr )
        print '    p = {:.3} '.format(p) + u'\u00B1' + ' {:.2}'.format( errp )
        print '    log(L)/N = {:.3}'.format( logl )
        # yfvals = ( ndata*neg_binomial( x, p, r ) for x in xfvals ) # conditional in neg binomial
        # plt.plot(xfvals, yfvals, 'v-', lw=2 )
        plt.plot(xfvals, ndata*neg_binomial( xfvals, r, p ), '--', lw=2, color='violet' )
        
    plt.show()

