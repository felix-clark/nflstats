import logging
import numpy as np
from numpy import sqrt, log, exp, pi
from scipy.special import gammaln, betaln, beta, comb, digamma
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
    # wikipedia has the opposite convention of everywhere else >:(
    # use the non-wikipedia convention
    # return ( k >= 0 ) * ( comb( k + r - 1, k) * (1-p)**k * p**r )
    karr = np.array(k)
    result = comb( karr + r - 1, karr) * (1-p)**k * p**r
    result[karr<0] = 0
    return result

def log_neg_binomial( k, r, p ):
    # return -gammaln( k+1 ) + gammaln( k + r ) - gammaln( r ) + k*log(p) + r*log(1-p)
    karr = np.array(k) # make sure it's an array
    result = k*log(1-p) + r*log(p)
    result[k>0] += - log(k) - betaln( k, r )
    result[k<0] = -np.inf
    return result

# discrete in range [0,n]
def beta_binomial( k, n, a, b ):
    # return comb(n, k) * exp( betaln(k + a, n - k + b) - betaln(a,b) )
    return comb(n, k) * beta(k + a, n - k + b) / beta(a,b)

def log_beta_binomial( k, n, a, b ):
    # if k < 0 or k > n: return -np.inf # is this check necessary? it makes operations on arrays annoying
    # if k == 0: return r*log(1-p)
    return log( comb(n,k) ) + betaln(k+a, n-k+b) - betaln(a,b)

# discrete (can be non-negative)
def gaussian_int( bounds, k, mu, sigma ):
    c = 1.0/(2*sigma**2)
    norm_factor = sum( exp(-c*( np.arange(*bounds) - mu)**2) )
    return exp( -c*(k-mu)**2 ) / norm_factor

def exp_poly_ratio( bounds, k, pis, qis):
    # p(x) = 1 + p1*x + ...
    px = lambda x: 1 + sum( ( pi*x**(i+1) for i,pi in enumerate(pis) ) )
    # q(x) = 1 + q1*x + ...
    qx = lambda x: 1 + sum( ( qi*x**(i+1) for i,qi in enumerate(qis) ) )
    polyr = lambda x: px(x)/qx(x)
    dom = np.arange(*bounds) # domain of distribution
    norm_factor = sum( exp( - polyr( dom ) ) )
    return exp( - polyr(k) ) / norm_factor


def sum_log_neg_binomial( ks, r, p ):
    N = len( ks )
    # print(p) # we are getting 0 or 1 here for some data
    return N*( r*log( p ) - gammaln( r ) ) + sum( gammaln( ks+r ) - gammaln( ks+1 ) + ks*log(1-p) )

    # karr = np.array(k) # make sure it's an array
    # result = k*log(1-p) + r*log(p)
    # result[k>0] += - log(k) - betaln( k, r )
    # result[k<0] = -np.inf
    # return result

def grad_sum_log_neg_binomial( ks, r, p ):
    N = len( ks )
    dldp =  N*r/p - sum( ks ) / (1-p)
    dldr = N*(log(p) - digamma(r)) + sum( digamma(ks+r) )
    return np.array((dldr, dldp))


def sum_log_beta_binomial( ks, n, a, b ):
    N = len(ks)
    return -N * betaln(a,b) + sum( log(comb(n,ks)) + betaln(ks+a, n-ks+b) )
    # return log( comb(n,k) ) + betaln(k+a, n-k+b) - betaln(a,b)

def grad_sum_log_beta_binomial( ks, n, a, b ):
    N = len(ks)
    common = N*(digamma(a+b) - digamma(n+a+b))
    dlda = sum(digamma(ks+a)) - N*digamma(a) + common
    dldb = sum(digamma(n-ks+b)) - N*digamma(b) + common
    return np.array((dlda, dldb))


def sum_log_gaussian_int( bounds, ks, mu, sigma):
    N = len( ks )
    c = 1.0/(2.0*sigma**2)
    norm_factor = N*log( sum( exp( -c*( np.arange(*bounds) - mu)**2) ) )
    sum_ll = - sum( c*(ks-mu)**2 )
    return sum_ll - norm_factor

def grad_sum_log_gaussian_int( bounds, ks, mu, sigma):
    N = len( ks )
    c = 1.0/(2.0*sigma**2)
    dom = np.arange(*bounds) # domain of distribution
    norm_fact = sum( exp( -c*(dom-mu)**2) )
    dll_dmu = sum( 2*c*(ks-mu) ) - N*sum( 2*c*(dom-mu)*exp( -c*(dom-mu)**2) )/norm_fact
    dll_dsigma = 2*c/sigma*(
        sum( (ks-mu)**2 )
        - N*sum( (dom-mu)**2 * exp( -c*(dom-mu)**2) )/norm_fact
        )
    return (dll_dmu, dll_dsigma)


def sum_log_exp_poly_ratio( bounds, ks, pis, qis):
    N = len( ks )
    # p(x) = 1 + p1*x + ...
    px = lambda x: 1 + sum( ( pi*x**(i+1) for i,pi in enumerate(pis) ) )
    # q(x) = 1 + q1*x + ...
    qx = lambda x: 1 + sum( ( qi*x**(i+1) for i,qi in enumerate(qis) ) )
    polyr = lambda x: px(x)/qx(x)
    dom = np.arange(*bounds) # domain of distribution
    norm_factor = N*log( sum( exp( - polyr( dom ) ) ) )
    # if np.isnan( norm_factor ):
    # print dom
    # print polyr( dom ) # these values get too big
    sum_ll = sum( - polyr( ks ) )
    return sum_ll - norm_factor

def _grad_sum_log_exp_poly_noratio( bounds, ks, pis ):
    N = len( ks )
    # p(x) = 1 + p1*x + ...
    px = lambda x: 1 + sum( ( pi*x**(i+1) for i,pi in enumerate(pis) ) )
    grad_px = lambda x: np.array([ x**(i+1) for i,_ in enumerate(pis) ]) # dp(x)/dpi = i*pi*x**(i-1)
    grad_sumll = - sum( grad_px( ks ).T )
    dom = np.arange(*bounds) # domain of distribution
    fdom = exp( - px( dom ) ) # array of f(k) mapped over domain
    fgraddom = ( - exp( -px(k) ) * grad_px(k) for k in dom )
    # fgraddom = - exp( -px(dom) ) * grad_px(dom)
    grad_norm_fact = N * sum( fgraddom ) / sum( fdom )
    return grad_sumll - grad_norm_fact

# should switch to a parameterization for p(x) and q(x) ~ 1 + a*x + ...
#  that is manifestly positive, e.g.
#  (a*x**2 + b*x + sin(t))**2 + (c*x + cos(t))**2
def grad_sum_log_exp_poly_ratio( bounds, ks, pis, qis ):
    N = len( ks )
    # p(x) = 1 + p1*x + ...
    px = lambda x: 1 + sum( ( pi*x**(i+1) for i,pi in enumerate(pis) ) )
    grad_px = lambda x: np.array([ x**(i+1) for i in np.arange(len(pis)) ]) # dp(x)/dpi = x**i
    # q(x) = 1 + q1*x + ...
    qx = lambda x: 1 + sum( ( qi*x**(i+1) for i,qi in enumerate(qis) ) )
    grad_qx = lambda x: np.array([ x**(i+1) for i in np.arange(len(qis)) ])
    polyr = lambda x: px(x)/qx(x)
    grad_polyr = lambda x: np.append( grad_px(x)/qx(x),
                                      -grad_qx(x)*px(x)/qx(x)**2 )
    grad_sumll = - sum( grad_polyr( ks ).T )
    dom = np.arange(*bounds) # domain of distribution
    fdom = exp( - polyr( dom ) ) # array of f(k) mapped over domain
    fgraddom = ( - exp( - polyr(k) ) * grad_polyr(k) for k in dom )
    # grad_norm_fact = N * sum( fdomain * grad_polyr( dom ).T ) / sum( fdomain )
    grad_norm_fact = N * sum( fgraddom ) / sum( fdom )
    return grad_sumll - grad_norm_fact

                         
# functions to return maximum-likelihood estimators for various distributions

def to_poisson( data=[] ):
    if not data:
        logging.error('empty data set')
        exit(1)
    n = len( data )
    mu = float( sum( data ) ) / n
    err_mu = sqrt( mu / n )
    log_L_per_ndf = mu * ( log(mu) - 1 ) * n/(n-1) \
                    - sum( gammaln( np.asarray(data)+1 ) ) / (n-1)
    return (mu, err_mu), log_L_per_ndf

def to_geometric( data=[] ):
    if not data:
        logging.error('empty data set')
        exit(1)
    n = len( data )
    p = float(n) / ( n + sum( data ) )
    err_p = sqrt( p**2 * (1-p) / n )
    log_L_per_ndf = ( log( p ) + log( 1-p )*(1-p)/p ) * n/(n-1)
    return (p, err_p), log_L_per_ndf

# note that these is an equation that can be solved for r, but it is not closed-form.
# it would reduce the dimensionality of the fit algorithm, however.
# then p is easily expressed in terms of the mean and r.
def to_neg_binomial( data=[] ):
    # if not data:
    #     logging.error('empty data set')
    #     exit(1)
    for x in data:
        if x < 0:
            logging.warning('negative value in data set. negative binomial may not be appropriate.')
    arr_ks = np.asarray( data )
    n = len( arr_ks )
    mean = float( sum( arr_ks ) ) / n
    rp0 = (mean, 0.5) # initial guess. r > 0 and 0 < p < 1
    allowed_methods = ['L-BFGS-B', 'TNC', 'SLSQP'] # these are the only ones that can handle bounds. they can also all handle jacobians. none of them can handle hessians.
    # only LBFGS returns Hessian, in form of "LbjgsInvHessProduct"
    method = allowed_methods[0]
    
    func = lambda pars: - sum_log_neg_binomial( arr_ks, *pars )
    grad = lambda pars: - grad_sum_log_neg_binomial( arr_ks, *pars )
    opt_result = opt.minimize( func, rp0, method=method, jac=grad, bounds=[(0,None),(0,1)] )
    isSuccess = opt_result.success
    if not isSuccess:
        logging.error('negative binomial fit did not succeed.')
    r,p = opt_result.x
    logging.debug('jacobian = {}'.format(opt_result.jac)) # should be zero, or close to it
    cov = opt_result.hess_inv
    cov_array = cov.todense()  # dense array
    neg_ll = opt_result.fun
    return isSuccess,(r,p),cov_array,-neg_ll/(n-2)

def to_beta_binomial( bounds, data ):
    """
    bounds = (min,max) where [min,max] are integers determining the (inclusive) domain
    """
    kmin,kmax = bounds
    n = kmax - kmin
    for x in data:
        if x < kmin or x > kmax:
            logging.warning('data out of domain for beta-binomial')
    arr_ks = np.asarray(data) - kmin # subtract lower edge to do fit
    N = len(arr_ks)
    m1 = float(sum( arr_ks )) / N
    m2 = float(sum( arr_ks**2 )) / N
    # use these moments for good initial guesses
    ab0 = np.array((0,0))
    if m1 > 0:
        denom = (n*(m2/m1 - m1 - 1) + m1)
        ab0 = np.array((n*m1-m2, (n-m1)*(n-m2/m1)))/denom
    else:
        logging.warning('all terms are zero:')
        logging.warning(arr_ks)
        ab0 = np.array((1.0/N,1.0/N))
    # if denom <= 0:
    #     logging.warning('m1 = {}, m2 = {}, n = {}, N = {}'.format(m1, m2, n, N))
    method = 'L-BFGS-B'    
    func = lambda pars: - sum_log_beta_binomial( arr_ks, n, *pars )
    grad = lambda pars: - grad_sum_log_beta_binomial( arr_ks, n, *pars )
    opt_result = opt.minimize( func, ab0, method=method, jac=grad, bounds=[(0,None),(0,None)] )
    logging.debug(opt_result.message)
    isSuccess = opt_result.success
    if not isSuccess:
        logging.error('beta binomial fit did not succeed with {} data points.'.format(N))
    a,b = opt_result.x
    # logging.debug('jacobian = {}'.format(opt_result.jac)) # should be zero, or close to it
    cov = opt_result.hess_inv
    cov_array = cov.todense()  # dense array
    neg_ll = opt_result.fun
    return isSuccess,(a,b),cov_array,-neg_ll/(N-2)

def to_gaussian_int( bounds, data=[] ):
    # if not data:
    #     print 'error: empty data set'
    #     exit(1)
    arr_ks = np.asarray( data )
    n = len( arr_ks )
    mean = float( sum( arr_ks ) ) / n
    stddev = sum( (arr_ks - mean)**2 ) / n # just for initial guess -- we don't need to worry about n-1
    rp0 = (mean, sqrt(stddev)) # initial guess. r > 0 and 0 < p < 1
    method = 'L-BFGS-B'
    
    func = lambda pars: - sum_log_gaussian_int( bounds, arr_ks, *pars )
    grad = lambda pars: - np.asarray( grad_sum_log_gaussian_int( bounds, arr_ks, *pars ) )
    opt_result = opt.minimize( func, rp0, method=method, jac=grad, bounds=[(0,None),(1,bounds[1]-bounds[0])] )
    # print opt_result.message
    if not opt_result.success:
        logging.error('integer gaussian fit did not succeed.')
    mu,sigma = opt_result.x
    # print 'jacobian = ', opt_result.jac # should be zero, or close to it
    cov = opt_result.hess_inv
    cov_array = cov.todense()  # dense array
    neg_ll = opt_result.fun
    return (mu,sigma), cov_array, -neg_ll/(n-2)

def to_exp_poly_ratio( npq, dom_bounds, data=[] ):
    """
    fits to exp( (1 + p1*x + ... + pnp*x**np)/(1 + q1*x + ... + qnq*x**nq) )
    npq = (n_p,n_q)
    n_p: order of polynomial p(x)
    n_q: order of polynomial q(x)
    dom_bounds: boundaries of integral domain (e.g. (2,6) -> [2,3,4,5])
    data: integral data to fit
    """
    (n_p,n_q) = npq
    if n_p <= n_q:
        logging.error('require higher degree of polynomial in numerator than denominator')
        exit(1)
    assert( n_p >= 2 )
    # don't technically need even order on p(x) for truly finite domain
    if n_p % 2 != 0:
        logging.error('need even order of p')
        exit(1)
    # even order on q(x) to avoid guaranteeing a zero. if coefficients are small enough this might be okay, if n_q < n_p.
    if n_q % 2 != 0:
        logging.error('need even order of q')
        exit(1)
    arr_ks = np.asarray( data )
    n = len( arr_ks )
    if n_p+n_q > n:
        logging.error('polynomial is under-constrained')
        exit(1)
    mean = float( sum( arr_ks ) ) / n
    var = sum( (arr_ks - mean)**2 ) / (n-1)

    # start with a gaussian:
    pi0 = np.append( np.array( [-mean/var, 1.0/(2*var)] ), np.zeros( n_p-2 ) )
    qi0 = np.zeros( n_q )                                    
    par0 = np.append( pi0, qi0 ) # initial guesses
    # print par0
    sdx = 0.5*abs(dom_bounds[1] - dom_bounds[0]) # gives width scale of problem
    # the highest-order p term should be negative
    #   (not necessary in general, but we want distributions that go to zero at |k| >> 1)
    fit_bounds = [(-10.0/sdx**(i+1),10.0/sdx**(i+1)) for i in range(n_p-1)] + [(0,None)] + [(-0.1/sdx**(i+1),0.1/sdx**(i+1)) for i in range(n_q)]
    logging.debug(fit_bounds)
    # fit_bounds = None
    allowed_methods = ['L-BFGS-B', 'TNC', 'SLSQP'] # these are the only ones that can handle bounds. they can also all handle jacobians. none of them can handle hessians.
    # only LBFGS returns Hessian, in form of "LbjgsInvHessProduct"
    method = allowed_methods[0]

    assert( n_p + n_q == len(par0) )
    # it could be worthwhile to use a parameterization that makes p(x) and q(x) explicitly positive
    # e.g. (a*x**2 + b*x + sin(t))**2 + (c*x + cos(t))^2 for order 4
    func = lambda pars: - sum_log_exp_poly_ratio( dom_bounds, arr_ks, pars[:n_p], pars[n_p:] )
    grad = lambda pars: - grad_sum_log_exp_poly_ratio( dom_bounds, arr_ks, pars[:n_p], pars[n_p:] )
    if n_q == 0: grad = lambda pars: - _grad_sum_log_exp_poly_noratio( dom_bounds, arr_ks, pars ) # this special case makes the general one not vectorizable
    # grad = None
    # in theory a constraint can be defined to keep q(x) non-zero.
    #   this might be difficult to define for arbitrary n_q.
    opt_result = opt.minimize( func, par0, method=method, jac=grad, bounds=fit_bounds )
    # print opt_result.message
    if not opt_result.success:
        logging.error('integer exponential polynomial ratio fit did not succeed.')
        logging.error('jacobian = {}'.format(opt_result.jac)) # should be zero, or close to it
    result = opt_result.x
    cov = opt_result.hess_inv
    cov_array = cov.todense()  # dense array
    neg_ll = opt_result.fun
    return result, cov_array, -neg_ll/(n-n_p-n_q)


# # for floating point data
# def to_gaussian( data=[] ):
#     if not data:
#         print 'error: empty data set'
#         exit(1)
#     if len( data ) == 1:
#         print 'need more than 1 data point to fit gaussian'
#         exit(1)
#     n = len( data )
#     mu = float( sum( data ) ) / n
#     var = sum( ( (x - mu)**2 for x in data ) ) / (n-1) # sample (as opposed to population) variance
#     sigma = sqrt( var )
#     err_mu = sigma / sqrt(n)
#     err_sigma = sigma / sqrt( 2*(n-1) ) # report sample standard deviation
#     log_L_per_ndf = (1 - log(2*pi*var))/2
#     return (mu, err_mu), (sigma, err_sigma), log_L_per_ndf


## functions to generate the plots of data with appropriate fits

# appropriate with non-negative integer data
def plot_counts( data=[], label='', norm=False, fits=['poisson', 'neg_binomial'] ):
    ndata = len( data )
    maxdata = max( data )

    # # instead of fitting unbinned likelihood fits, since the results are all integers
    # #   we may get speedup by fitting to histograms (which will require an additional implementation)
    # # probably don't need to save all return values
    entries, bin_edges, patches = plt.hist( data, bins=np.arange(-0.0,maxdata+4,1),
                                            # range=[-0.5, maxtds+1.5],
                                            align='left',
                                            normed=norm,
                                            label=label
    )
    
    # yerrs = [ sqrt( x / ndata ) for x in entries ] if norm else [ sqrt( x ) for x in entries ]
    yerrs = sqrt( entries ) / ndata if norm else sqrt( entries )
    
    xfvals = np.linspace(0, maxdata+3, 1000) # get from bin_edges instead?

    # do likelihood fits for each type

    if 'geometric' in fits:             
        (p,errp),logl = to_geometric( data )
        logging.info('  Geometric fit:')
        logging.info('    p = {:.3} '.format(p) + u'\u00B1' + ' {:.2}'.format( errp ))
        logging.info('    log(L)/NDF = {:.3}'.format( logl ))
        plt.subplot(121)
        plt.errorbar( np.arange(0,maxdata+3), entries, yerr=yerrs, align='left', fmt='none', color='black' )
        plt.plot(xfvals, ndata*geometric( xfvals, p ), 'g-', lw=2)
        plt.subplot(122)
        plt.errorbar( np.arange(0,maxdata+3), entries, yerr=yerrs, align='left', fmt='none', color='black' )
        plt.plot(xfvals, ndata*geometric( xfvals, p ), 'g-', lw=2)
        plt.yscale('log')

    if 'poisson' in fits:
        (mu,errmu),logl = to_poisson( data )
        logging.info('  Poisson fit:')
        logging.info('    ' + u'\u03BC' + ' = {:.3} '.format(mu) +  u'\u00B1' + ' {:.2}'.format( errmu ))
        logging.info('    log(L)/NDF = {:.3}'.format( logl ))
        plt.subplot(121)
        plt.plot(xfvals, ndata*poisson( xfvals, mu ), 'r-', lw=2)
        plt.subplot(122)
        plt.plot(xfvals, ndata*geometric( xfvals, mu ), 'r-', lw=2)
        plt.yscale('log')

    if 'neg_binomial' in fits:
        # (r,errr),(p,errp),logl = to_neg_binomial( data )
        _,(r,p),cov,logl = to_neg_binomial( data )
        errr = sqrt( cov[0][0] )
        errp = sqrt( cov[1][1] )
        logging.info('  Negative binomial fit:')
        logging.info('    r = {:.3} '.format(r) + u'\u00B1' + ' {:.2}'.format( errr ))
        logging.info('    p = {:.3} '.format(p) + u'\u00B1' + ' {:.2}'.format( errp ))
        logging.info('    log(L)/NDF = {:.3}'.format( logl ))
        # yfvals = ( ndata*neg_binomial( x, p, r ) for x in xfvals ) # conditional in neg binomial
        # plt.plot(xfvals, yfvals, 'v-', lw=2 )
        plt.subplot(121)
        plt.errorbar( np.arange(0,maxdata+3), entries, yerr=yerrs, align='left', fmt='none', color='black' )
        plt.plot(xfvals, ndata*neg_binomial( xfvals, r, p ), '--', lw=2, color='violet' )
        plt.subplot(122)
        plt.errorbar( np.arange(0,maxdata+3), entries, yerr=yerrs, align='left', fmt='none', color='black' )
        plt.plot(xfvals, ndata*neg_binomial( xfvals, r, p ), '--', lw=2, color='violet' )
        plt.yscale('log')

    plt.show()

# appropriate with any integer data
# distributions proportional to exponentials of polynomial ratios
def plot_counts_poly( data=[], bounds=(-100,100), label='', norm=False ):
    ndata = len( data )
    mindata = min( data )
    maxdata = max( data )

    # # probably don't need to save all return values
    entries, bin_edges, patches = plt.hist( data, bins=np.arange(mindata-2,maxdata+4,1),
                                            # range=[-0.5, maxtds+1.5],
                                            align='left',
                                            normed=norm,
                                            label=label
    )
    
    yerrs = sqrt( entries ) / ndata if norm else sqrt( entries )
    plt.subplot(121)
    plt.errorbar( np.arange(mindata-2,maxdata+3), entries, yerr=yerrs, align='left', fmt='none', color='black' )
    plt.subplot(122)
    plt.errorbar( np.arange(mindata-2,maxdata+3), entries, yerr=yerrs, align='left', fmt='none', color='black' )
    
    xfvals = np.linspace(mindata-5, maxdata+6, 1000)

    (mu,sigma),cov,logl = to_gaussian_int( bounds, data )
    errmu = sqrt( cov[0][0] )
    errsigma = sqrt( cov[1][1] )
    logging.info('    ' + u'\u03BC' + ' = {:.3} '.format(mu) +  u'\u00B1' + ' {:.2}'.format( errmu ))
    logging.info('    ' + u'\u03C3' + ' = {:.3} '.format(sigma) +  u'\u00B1' + ' {:.2}'.format( errsigma ))
    logging.info('    log(L)/NDF = {:.3}'.format( logl ))
    plt.subplot(121)
    plt.plot(xfvals, ndata*gaussian_int( bounds, xfvals, mu, sigma ), '--', lw=2, color='blue' )
    plt.subplot(122)
    plt.plot(xfvals, ndata*gaussian_int( bounds, xfvals, mu, sigma ), '--', lw=2, color='blue' )
    plt.yscale('log', nonposy='clip')

    n_p = 6
    n_q = 0
    pars,cov,logl = to_exp_poly_ratio( (n_p,n_q), bounds, data )
    errs = sqrt( np.diagonal(cov) )
    assert( len(errs) == n_p + n_q )
    pis,qis = pars[:n_p], pars[n_p:]
    errpis,errqis = errs[:n_p], errs[n_p:]
    for i,(p,dp) in enumerate(zip(pis,errpis), start=1):
        logging.info('    p{} = {:.3} '.format( i,p ) +  u'\u00B1' + ' {:.2}'.format( dp ))
    for i,(q,dq) in enumerate(zip(qis,errqis), start=1):
        logging.info('    q{} = {:.3} '.format( i,q ) +  u'\u00B1' + ' {:.2}'.format( dq ))
    logging.info('    log(L)/NDF = {:.3}'.format( logl ))
    plt.subplot(121)
    plt.plot(xfvals, ndata*exp_poly_ratio( bounds, xfvals, pis, qis ), '-', lw=2, color='green' )
    plt.subplot(122)
    plt.plot(xfvals, ndata*exp_poly_ratio( bounds, xfvals, pis, qis ), '-', lw=2, color='green' )
    plt.yscale('log', nonposy='clip')
    xlow,xup,ylow,yup = plt.axis()
    plt.axis( (xlow,xup,0.5,yup) )
    
    plt.show()
