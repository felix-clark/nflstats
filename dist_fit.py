import logging
import numpy as np
from numpy import sqrt, log, exp, pi
from scipy.special import gammaln, betaln, beta, comb, digamma, gamma, factorial
import scipy.optimize as opt
import scipy.stats as st
import matplotlib.pyplot as plt
from math import floor, ceil

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
    result = np.array(karr*log(1-p) + r*log(p))
    result[karr>0] += - log(karr) - betaln( karr, r )
    result[karr<0] = -np.inf
    return result

# discrete in range 0,infinity
# more variance than neg. bin.
def beta_neg_binomial( k, r, a, b ):
    return gamma(r+k)/(factorial(k) * gamma(r)) * beta(a+r,b+k) / beta(a,b)

def log_beta_neg_binomial( k, r, a, b ):
    return gammaln(r+k) - gammaln(r) - log(factorial(k)) + betaln(a+r,b+k) - betaln(a,b)

# discrete in range [0,n]
def beta_binomial( k, n, a, b ):
    return comb(n, k) * exp( betaln(k + a, n - k + b) - betaln(a,b) ) # this can avoid overflow
    # result = comb(n, k) * beta(k + a, n - k + b) / beta(a,b)
    # return result

def log_beta_binomial( k, n, a, b ):
    # if k < 0 or k > n: return -np.inf # is this check necessary? it makes operations on arrays annoying
    # if k == 0: return r*log(1-p)
    # print(n,k)
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

def sum_log_beta_neg_binomial( ks, r, a, b ):
    N = len(ks)
    norm = N*(betaln(a,b) + gammaln(r))
    terms = gammaln(r+ks) - log(factorial(ks)) + betaln(a+r,b+ks) 
    return sum(terms) - norm

def grad_sum_log_beta_neg_binomial( ks, r, a, b):
    N = len(ks)
    dg_ar = digamma(a+r)
    dg_ab = digamma(a+b)
    sum_dg_all = sum(digamma(a+r+b+ks))
    dldr = sum(digamma(r+ks)) - N*digamma(r) + dg_ar - sum_dg_all
    dlda = N*(dg_ar + dg_ab - digamma(a)) - sum_dg_all
    dldb = N*(dg_ab - digamma(b)) + sum(digamma(b+ks)) - sum_dg_all
    return np.array((dldr, dlda, dldb))

# this does not account for the distribution of ns
def sum_log_beta_binomial( ks, ns, a, b ):
    # ns may or may not be variable
    # if n is variable, we should weight w.r.t. n
    N = len(ks)
    weights = np.full(shape=ks.shape, fill_value=1.0, dtype=float)
    if np.shape(ns) != ():
        weights = np.array(ns).astype(float)/np.mean(ns)
    # return -N * betaln(a,b) + sum( log(comb(ns,ks)) + betaln(ks+a, ns-ks+b) )
    # assert( a > 0 and b > 0) # why does this assertion fail for L-BFGS-B w/ gradient?
    assert( (ns>0).all() )
    result = -N * betaln(a,b) + sum( weights * (log(comb(ns,ks)) + betaln(ks+a, ns-ks+b)) )
    return result

def grad_sum_log_beta_binomial( ks, ns, a, b ):
    N = len(ks)
    weights = np.full(shape=ks.shape, fill_value=1.0, dtype=float)
    if np.shape(ns) != ():
        weights = np.array(ns).astype(float)/np.mean(ns)
    common = N*digamma(a+b) - sum(weights*digamma(ns+a+b))
    dlda = sum(weights*digamma(ks+a)) - N*digamma(a) + common
    dldb = sum(weights*digamma(ns-ks+b)) - N*digamma(b) + common
    # result = -N * betaln(a,b) + sum( weights* (log(comb(ns,ks)) + betaln(ks+a, ns-ks+b)) )
    # print( 'a, b, result, dlda, dldb = {}, {}, {}, {}, {}'.format( a, b, result, dlda, dldb))
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

def to_poisson( data ):
    n = len( data )
    if n == 0:
        logging.error('empty data set')
        exit(1)
    mu = float( sum( data ) ) / n
    err_mu = sqrt( mu / n )
    log_L_per_ndf = mu * ( log(mu) - 1 ) * n/(n-1) \
                    - sum( gammaln( np.asarray(data)+1 ) ) / (n-1)
    return (mu, err_mu), log_L_per_ndf

def to_geometric( data ):
    n = len( data )
    p = float(n) / ( n + sum( data ) )
    err_p = sqrt( p**2 * (1-p) / n )
    log_L_per_ndf = ( log( p ) + log( 1-p )*(1-p)/p ) * n/(n-1)
    return (p, err_p), log_L_per_ndf

# note that these is an equation that can be solved for r, but it is not closed-form.
# it would reduce the dimensionality of the fit algorithm, however.
# then p is easily expressed in terms of the mean and r.
def to_neg_binomial( data ):
    # if not data:
    #     logging.error('empty data set')
    #     exit(1)
    log = logging.getLogger(__name__)
    arr_ks = np.asarray( data )
    if (arr_ks < 0).any():
            log.warning('negative value in data set. negative binomial may not be appropriate.')
    n = len( arr_ks )
    mean = arr_ks.mean()
    var = arr_ks.var()
    p0 = mean/var
    r0 = mean**2/(var-mean) # initial guess. r > 0 and 0 < p < 1
    logging.info('r0,p0 = {:.3f}, {:.3f}'.format(r0,p0))
    assert((r0 > 0) and p0 < 1)
    allowed_methods = ['L-BFGS-B', 'TNC', 'SLSQP'] # these are the only ones that can handle bounds. they can also all handle jacobians. none of them can handle hessians.
    # only LBFGS returns Hessian, in form of "LbjgsInvHessProduct"
    method = allowed_methods[0]
    
    func = lambda pars: - sum_log_neg_binomial( arr_ks, *pars )
    grad = lambda pars: - grad_sum_log_neg_binomial( arr_ks, *pars )
    opt_result = opt.minimize( func, (r0,p0), method=method, jac=grad, bounds=[(0,None),(0,1)] )
    isSuccess = opt_result.success
    if not isSuccess:
        log.error('negative binomial fit did not succeed.')
    r,p = opt_result.x
    log.debug('jacobian = {}'.format(opt_result.jac)) # should be zero, or close to it
    cov = opt_result.hess_inv
    cov_array = cov.todense()  # dense array
    neg_ll = opt_result.fun
    return isSuccess,(r,p),cov_array,-neg_ll/(n-2)

def to_beta_neg_binomial( data ):
    arr_ks = np.asarray( data, dtype=float )
    if np.any(arr_ks < 0):
        logging.warning('negative value in data set. beta negative binomial may not be appropriate.')
    n = len( arr_ks )
    mean = arr_ks.mean()
    var = arr_ks.var()
    # p0 = mean/var # the EV for a/(a+b)
    a0,b0 = (mean, var-mean) # start w/ low variance. make sure a > 1
    r0 = mean*(a0-1)/b0 # initial guess for r,a,b # mean = r*b/(a-1) for a > 1
    allowed_methods = ['L-BFGS-B', 'TNC', 'SLSQP'] # these are the only ones that can handle bounds. they can also all handle jacobians. none of them can handle hessians.
    # only LBFGS returns Hessian, in form of "LbjgsInvHessProduct"
    method = allowed_methods[0]
    
    func = lambda pars: - sum_log_beta_neg_binomial( arr_ks, *pars )
    grad = lambda pars: - grad_sum_log_beta_neg_binomial( arr_ks, *pars )
    opt_result = opt.minimize( func, (r0,a0,b0), method=method, jac=grad, bounds=[(1e-6,None),(0,None),(0,None)],
                               options={'disp':False, # print convergence message
                                        'ftol':1e-12, # get better tolerance
                                        'gtol':1e-10 # these don't seem to help
                               }
    )
    isSuccess = opt_result.success
    if not isSuccess:
        logging.error('beta negative binomial fit did not succeed.')
    r,a,b = opt_result.x
    logging.debug('jacobian for beta-neg-binomial = {}'.format(opt_result.jac)) # should be zero, or close to it
    cov = opt_result.hess_inv # seems to just return 1 here...
    cov_array = cov.todense()  # dense array
    neg_ll = opt_result.fun
    return isSuccess,(r,a,b),cov_array,-neg_ll/(n-3)

def to_beta_binomial( ks, ns ):
    """
    bounds = (0,n) where determines the (inclusive) domain
    ns may be variable
    """
    if np.shape(ns) == ():
        ns = np.full(shape=ks.shape, fill_value=ns, dtype=float)
    for x,n in zip(ks,ns):
        if x < 0 or x > n:
            logging.warning('data out of domain for beta-binomial')
    arr_ks = np.asarray(ks, dtype=float)
    N = len(arr_ks)
    # see "Further bayesian considerations" under beta-binomial wiki 
    mu = sum( arr_ks ) / sum(ns)
    # s2 = np.var( arr_ks/ns ) / N
    s2 = sum( ns*(arr_ks/ns-mu)**2 ) / sum(ns) * 1.0/(1-1.0/N)
    # M = ( mu*(1-mu) - s2 ) / ( s2 - mu*(1-mu)*np.mean(1.0/ns) )
    M = mu*(1-mu)/s2 - 1
    if M <= 0.0: # don't want this to be too small
        logging.debug('natural M = {} < 0.'.format(M))
        logging.debug('mu(1-mu)/var = {} (is it > 1?)'.format(mu*(1-mu)/s2))
        # M = 1.0/(1.0-mu)
    logging.debug('bayesian mu,mu*(1-mu), s2,M = {}, {}, {}, {}'.format(mu, mu*(1-mu), s2, M))
    # use these moments for good initial guesses
    ab0 = np.array((M*mu, M*(1-mu))) # should be able to at least get the mean
    logging.debug('bayesian a0,b0 = {}, {}'.format(*ab0))
    
    # if denom <= 0:
    #     logging.warning('m1 = {}, m2 = {}, n = {}, N = {}'.format(m1, m2, n, N))
    allowed_methods = ['L-BFGS-B', 'TNC', 'SLSQP'] # these are the only ones that can handle bounds. they can also all handle jacobians. none of them can handle hessians.
    method = allowed_methods[0]
    func = lambda pars: - sum_log_beta_binomial( arr_ks, ns, *pars )
    grad = lambda pars: - grad_sum_log_beta_binomial( arr_ks, ns, *pars )
    minopts = {
        'maxcor':20, # maximum of variable metric corrections (default 10)
        'ftol':1e-12, # tolerance of objective function. (default ~2.22e-9)
        'gtol':1e-8 # gradient tolerance (default 1e-5)
    }
    opt_result = opt.minimize( func, ab0, method=method, jac=grad, bounds=[(0,None),(0,None)], options=minopts )
    # opt_result = opt.minimize( func, ab0, method=method, bounds=[(0,None),(0,None)] )
    logging.debug(opt_result.message)
    isSuccess = opt_result.success
    if not isSuccess:
        logging.error('beta binomial fit did not succeed with {} data points.'.format(N))
    a,b = opt_result.x
    logging.debug('jacobian = {}'.format(opt_result.jac)) # should be zero, or close to it
    # cov = opt_result.hess_inv
    # cov_array = cov.todense()  # dense array
    neg_ll = opt_result.fun
    return isSuccess,(a,b),None,-neg_ll/(N-2)

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


## functions to generate the plots of data with appropriate fits

# appropriate with non-negative integer data
def plot_counts( data, label='', norm=False, fits=None ):
    if fits is None:
        fits = ['poisson', 'neg_binomial', 'beta_neg_binomial']
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
    plt.subplot(121)
    plt.errorbar( np.arange(0,maxdata+3), entries, yerr=yerrs, align='left', fmt='none', color='black' )
    plt.subplot(122)
    plt.errorbar( np.arange(0,maxdata+3), entries, yerr=yerrs, align='left', fmt='none', color='black' )
    
    # do likelihood fits for each type

    if 'geometric' in fits:             
        (p,errp),logl = to_geometric( data )
        logging.info('  Geometric fit:')
        logging.info('    p = {:.3} '.format(p) + u'\u00B1' + ' {:.2}'.format( errp ))
        logging.info('    log(L)/NDF = {:.3}'.format( logl ))
        plt.subplot(121)
        plt.plot(xfvals, ndata*geometric( xfvals, p ), 'g-', lw=2)
        plt.subplot(122)
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
        plt.plot(xfvals, ndata*poisson( xfvals, mu ), 'r-', lw=2)
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
        plt.plot(xfvals, ndata*neg_binomial( xfvals, r, p ), '--', lw=2, color='violet' )
        plt.subplot(122)
        plt.plot(xfvals, ndata*neg_binomial( xfvals, r, p ), '--', lw=2, color='violet' )
        plt.yscale('log')

    if 'beta_neg_binomial' in fits:
        _,(r,a,b),cov,logl = to_beta_neg_binomial(data)
        errr = sqrt( cov[0][0] )
        erra = sqrt( cov[1][1] )
        errb = sqrt( cov[2][2] )
        logging.info('  Beta negative binomial fit:')
        logging.info('    r = {:.3} '.format(r) + u'\u00B1' + ' {:.2}'.format( errr ))
        logging.info('    a = {:.3} '.format(a) + u'\u00B1' + ' {:.2}'.format( erra ))
        logging.info('    b = {:.3} '.format(b) + u'\u00B1' + ' {:.2}'.format( errb ))
        logging.info('    log(L)/NDF = {:.3}'.format( logl ))
        # yfvals = ( ndata*neg_binomial( x, p, r ) for x in xfvals ) # conditional in neg binomial
        # plt.plot(xfvals, yfvals, 'v-', lw=2 )
        plt.subplot(121)
        plt.plot(xfvals, ndata*beta_neg_binomial( xfvals, r, a, b ), '--', lw=2, color='blue' )
        plt.subplot(122)
        plt.plot(xfvals, ndata*beta_neg_binomial( xfvals, r, a, b ), '--', lw=2, color='blue' )
        plt.yscale('log', nonposy='clip')

    plt.show()

# this will just do a gaussian fit right now
# a student-t would be a good addition
# and perhaps some skewed dists as well?
def plot_avg_per( data, bounds=(-5,40), label='', weights=None, norm=True ):
    ndata = len( data )
    mindata = floor(min( data ))
    maxdata = ceil(max( data ))
    
    # # probably don't need to save all return values
    entries, bin_edges, patches = plt.hist( data, bins=np.arange(mindata-2,maxdata+4,1),
                                            # range=[-0.5, maxtds+1.5],
                                            align='left',
                                            normed=norm,
                                            weights=weights,
                                            label=label
    )
    sqerrs, _, _ = plt.hist( data, bins=np.arange(mindata-2,maxdata+4,1),
                             align='left',
                             # normed=norm,
                             weights=weights**2,
    )
    yerrs = sqrt( sqerrs ) / weights.sum()
    
    plt.subplot(121)
    plt.errorbar( np.arange(mindata-2,maxdata+3), entries, yerr=yerrs, align='left', fmt='none', color='black' )
    plt.subplot(122)
    plt.errorbar( np.arange(mindata-2,maxdata+3), entries, yerr=yerrs, align='left', fmt='none', color='black' )
    
    xfvals = np.linspace(mindata-5, maxdata+6, 1000)

    # (mu,sigma),cov,logl = to_gaussian_int( bounds, data )
    mu,sigma = st.norm.fit( data )
    errmu = '?' # sqrt( cov[0][0] )
    errsigma = '?' # sqrt( cov[1][1] )
    logging.info('    ' + u'\u03BC' + ' = {:.3} '.format(mu) +  u'\u00B1' + ' {:.2}'.format( errmu ))
    logging.info('    ' + u'\u03C3' + ' = {:.3} '.format(sigma) +  u'\u00B1' + ' {:.2}'.format( errsigma ))
    logging.info('    -log(L)/NDF = {:.3}'.format( -st.norm.logpdf(data, mu, sigma).sum()/(len(data)-2) ))
    mu_wt = sum(weights*data)/weights.sum()
    sigma_wt = np.sqrt( sum(weights*(data-mu_wt)**2)/weights.sum() )
    logging.info('  weighted results:')
    logging.info('    ' + u'\u03BC' + ' = {:.3} '.format(mu_wt) )
    logging.info('    ' + u'\u03C3' + ' = {:.3} '.format(sigma_wt))
    logging.info('    -log(L)/NDF = {:.3}'.format( -sum(weights*st.norm.logpdf(data, mu_wt, sigma_wt))/weights.sum()*len(data)/(len(data)-2.) ))
    plt.subplot(121)
    plt.plot(xfvals, st.norm.pdf( xfvals, mu, sigma ), '--', lw=2, color='blue' )
    plt.plot(xfvals, st.norm.pdf( xfvals, mu_wt, sigma_wt ), 'r-')
    plt.subplot(122)
    plt.plot(xfvals, st.norm.pdf( xfvals, mu, sigma ), '--', lw=2, color='blue' )
    plt.plot(xfvals, st.norm.pdf( xfvals, mu_wt, sigma_wt ), 'r-')
    plt.yscale('log', nonposy='clip')
    xlow,xup,ylow,yup = plt.axis()
    plt.axis( (xlow,xup,0.001,0.5) )

    logging.info('non-centered t:')
    df,nc,loc,scale = st.nct.fit( data )
    logging.info('    df = {:.3} '.format(df))
    logging.info('    nc = {:.3} '.format(nc))
    logging.info('    loc = {:.3} '.format(loc))
    logging.info('    scale = {:.3} '.format(scale))
    
    logging.info('    -log(L)/NDF = {:.3}'.format( -st.nct.logpdf(data, df, nc, loc, scale).sum()/(len(data)-4) ))
    logging.info('    mean (scipy / me) = {:.3f} / {:.3f}'.format(st.nct.mean(df, nc, loc, scale),
                                                                  loc + scale*nc*np.sqrt(df/2)*gamma((df-1)/2)/gamma(df/2)))
    plt.subplot(121)
    plt.plot(xfvals, st.nct.pdf( xfvals, df, nc, loc, scale ), '-', lw=2, color='green' )
    plt.subplot(122)
    plt.plot(xfvals, st.nct.pdf( xfvals, df, nc, loc, scale ), '-', lw=2, color='green' )
    plt.yscale('log', nonposy='clip')
    xlow,xup,ylow,yup = plt.axis()
    plt.axis( (xlow,xup,1e-6,0.5) )

    
    plt.show()
    
# distributions proportional to exponentials of polynomial ratios
# we will likely not use these distributions in our currnet model
def plot_counts_poly( data, bounds=(-100,100), label='', norm=False ):
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

## pretty much only a beta distribution is appropriate here (?)
def plot_fraction( data_num, data_den, label='', fits=None, step=0.002 ):
    # will fit to fraction num/den to get alpha and beta parameters
    if fits is None:
        fits = ['beta_binomial']
    ndata = len( data_num )
    data_num = np.array(data_num).astype(float)
    data_den = np.array(data_den).astype(float)
    data_ratio = data_num.astype(float)/data_den.astype(float)
    # # instead of fitting unbinned likelihood fits, since the results are all integers
    # #   we may get speedup by fitting to histograms (which will require an additional implementation)
    # # probably don't need to save all return values
    entries, bin_edges, patches = plt.hist( data_ratio, bins=np.arange(0,1+step,step),
                                            align='left',
                                            normed=True,
                                            weights=data_den, # weight by number of attempts
                                            label=label
    )
    sqerrs, _, _ = plt.hist( data_ratio, bins=np.arange(0,1+step,step),
                             align='left',
                             normed=False,
                             weights=data_den**2,
    )
    yerrs = sqrt( sqerrs ) / data_den.sum()
    
    # print([(y,e) for y,e in zip(entries,yerrs) if y > 0])# these errors are very small. are we doing this properly?

    #check the gradient: <- it's good
    # atest,btest,ep = 1.4,9.2,1e-6
    # dl_g = grad_sum_log_beta_binomial(data_num, data_den, atest, btest)
    # dlda_n = (sum_log_beta_binomial(data_num, data_den, atest+ep, btest)
    #           - sum_log_beta_binomial(data_num, data_den, atest-ep, btest))/(2*ep)
    # dldb_n = (sum_log_beta_binomial(data_num, data_den, atest, btest+ep)
    #           - sum_log_beta_binomial(data_num, data_den, atest, btest-ep))/(2*ep)
    # logging.debug('analytic, numberic along alpha: {}, {}'.format(dl_g[0], dlda_n))
    # logging.debug('analytic, numberic along beta: {}, {}'.format(dl_g[1], dldb_n))
    
    # yerrs = [ sqrt( x / ndata ) for x in entries ] if norm else [ sqrt( x ) for x in entries ]
    ## these errors don't really hold with weights...
    
    xfvals = np.linspace(0, 1, 1000) # get from bin_edges instead?
    plt.subplot(121)
    plt.errorbar( np.arange(0,1,step), entries, yerr=yerrs, align='left', fmt='none', color='black' )
    plt.subplot(122)
    plt.errorbar( np.arange(0,1,step), entries, yerr=yerrs, align='left', fmt='none', color='black' )

    logging.debug('weighted/unweighted mean, stddev: {} / {}, {}'.format(sum(data_num)/sum(data_den), data_ratio.mean(), data_ratio.std()))
    
    # # we checked that all these functions give consistent values:
    # testk,testn = np.array([4]),np.array([10])
    # testa, testb = 2.0, 1.0
    # testpars = testk, testn, testa, testb
    # logging.debug('bb pdf: {}'.format(beta_binomial(*testpars)))
    # logging.debug('bb log pdf: {}, {}, {}'.format(log(beta_binomial(*testpars)), log_beta_binomial(*testpars), sum_log_beta_binomial(*testpars) ))
    
    if 'beta_binomial' in fits:
        _,(a,b),cov,logl = to_beta_binomial(data_num, data_den)
            
        # need to do a contour search or use optimize() to get uncertainties (TODO)
        erra,errb = '?','?'
        if cov is not None:
            erra = sqrt( cov[0][0] )
            errb = sqrt( cov[1][1] )
        logging.info('  Beta binomial fit:')
        logging.info('    a = {:.3} '.format(a) + u'\u00B1' + ' {:.2}'.format( erra ))
        logging.info('    b = {:.3} '.format(b) + u'\u00B1' + ' {:.2}'.format( errb ))
        logging.info('    a/(a+b) = {:.3}'.format(a/(a+b)))
        logging.info('    sqrt( ab/(a+b)**2/(a+b+1) ) = {:.3}'.format(np.sqrt(a*b/(a+b)**2/(a+b+1)))) 
        logging.info('    log(L)/NDF = {:.3} (not including constant term)'.format( logl ) )
        # yfvals = ( ndata*neg_binomial( x, p, r ) for x in xfvals ) # conditional in neg binomial
        # plt.plot(xfvals, yfvals, 'v-', lw=2 )
        plt.subplot(121)
        plt.plot(xfvals, st.beta.pdf( xfvals, a, b ), '--', lw=2, color='blue' )
        res = plt.subplot(122)
        plt.plot(xfvals, st.beta.pdf( xfvals, a, b ), '--', lw=2, color='blue' )
        plt.yscale('log', nonposy='clip')
        # res.set_ylim(0.1,None)

    plt.show()
