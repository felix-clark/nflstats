import logging
import dist_fit
import numpy as np

# we don't really need to define an interface
class bayes_model:
    log = logging.getLogger(__name__)
    def mse(self, data, *pars, **kwargs):
        """
        the data is assumed to be sequential
        """
        return self._mse(np.array(data), *pars, **kwargs)

    def kld(self, data, *pars, **kwargs):
        """
        data is assumed to be sequential
        kullback-leibler divergence for each point, with updating of bayesian parameters if necessary
        """
        return self._kld(np.array(data), *pars, **kwargs)

class const_model_gen(bayes_model):
    """
    This model uses a single constant for all predictions.
    Useful as a baseline (often the simple average of all data)
    """
    def _mse(self, data, mu, var=0.):
        return (mu-data)**2 + var
        pass # don't have weighted case yet, as it hasn't been necessary

    def _kld(self, data, mu, var=0.):
        if var == 0:
            log.warning('zero in variance for KL divergence')
            return np.array([-np.inf if d == mu else np.inf for d in data])
        return (mu-data)**2/(2.0*var) + 0.5*np.log(2.0*np.pi*var)
        
gauss_const = const_model_gen()



# the model is defined by it's posterior predictive distribution, which are compound distributions for easy updating of parameters.
# beta-binomial is a compound of beta with binomial.
# for new observation k, the model is updated:
# alpha -> alpha + k
# beta -> beta + (n-k)
# alpha and beta correspond to those for the beta-binomial (for the posterior predictive),
# or the beta distribution for the prior.
class beta_binomial_model_gen(bayes_model):
    def _mse(self, data, n, alpha0, beta0, lr=1.0):
        # lr can be used to suppress learning
        # we could also apply multiplicative decay after adding, which will result in variance decay even in long careers
        assert((data >= 0).all() and (data <= n).all())
        mses = []
        alpha,beta = alpha0,beta0
        # domain of summation for EV computation:
        support = np.arange(0,n+1)
        for d in data:
            # beta-binomial isn't in scipy ?
            probs = dist_fit.beta_binomial( support, n, alpha, beta )
            mses.append( sum(probs*(support-d)**2) )
            alpha += lr*d
            beta += lr*(n-d)
        # log.debug('alpha, beta = {},{}'.format(alpha,beta))
        return np.array(mses)

    # computes Kullback-Leibler divergence
    def _kld(self, data, n, alpha0, beta0, lr=1.0):
        assert((data >= 0).all() and (data <= n).all())
        mses = []
        alpha,beta = alpha0,beta0
        for d in data:
            mses.append( -dist_fit.log_beta_binomial( d, n, alpha, beta ) )
            alpha += lr*d
            beta += lr*(n-d)
        return np.array(mses)

bbinom = beta_binomial_model_gen()


## the negative binomial posterior predictor is a gamma convolved with a poisson
## the update rules are:
# r -> r + lr*k
# beta -> beta + lr where beta = (1-p)/p so:
#   p -> p/(1 + (2-lr)*p) ... but for lr = 0, this goes to p/(1+2p) != p... (think about this)
# where if lr < 1 we are slowing the update
# we could also apply multiplicative decay after adding, which will result in variance decay even in long careers
class neg_binomial_model_gen(bayes_model):
    def _mse(self, data, r0, p0, lr=1.0):
        mses = []
        r,p = r0,p0
        # domain of summation for EV computation - extend out to 5 significance
        support = np.arange(0,5*(p*r)**(1.5)/(1-p)**2)
        for d in data:
            # neg-binomial is in scipy.stats, but uses a different parameterization.
            # will probably be worth it to switch for the function nbinom.expect()
            probs = dist_fit.neg_binomial( support, alpha, beta )
            mses.append( sum(probs*(support-d)**2) )
            r += lr*d
            p *= 1.0/(1.0 + (2.0-lr)*p) # <- questionable ... what about lr == 0?
        # log.debug('alpha, beta = {},{}'.format(alpha,beta))
        return np.array(mses)

nbinom = neg_binomial_model_gen()



## TODO: work these functions into the class structure above. do we really need MAE?
## the "mean" model can perhaps be made bayesian, using unknown variance -- student's t?

# mean-absolute-error of model that uses average
def mae_model_const(data, const=0, weights=None):
    return np.abs(const-data)

# mean-squared-error of model that uses mean of past
def mse_model_mean(data, default=0):
    mses = []
    if data.size == 0:
        return default
    for i_d in range(data.size):
        mean_so_far = data.iloc[:i_d].mean() if i_d > 0 else default
        mses.append( (mean_so_far-data.iloc[i_d])**2 )
    return np.array(mses)

# mean absolute error
def mae_model_bb(data, alpha0, beta0, lr=1.0, n=16):
    # lr can be used to suppress learning
    # we can also apply multiplicatively after adding, which will result in variance decay even in long careers
    assert((data >= 0).all() and (data <= n).all())
    maes = []
    alpha,beta = alpha0,beta0
    # domain of summation for EV computation:
    support = np.arange(0,n+1)
    for d in data:
        probs = dist_fit.beta_binomial( support, n, alpha, beta )
        maes.append( sum(probs*np.abs(support-d)) )
        alpha += lr*d
        beta += lr*(n-d)
    # logging.debug('alpha, beta = {},{}'.format(alpha,beta))
    return np.array(maes)

