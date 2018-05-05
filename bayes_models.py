import logging
import dist_fit
import numpy as np
import scipy.stats as st

# we don't really need to define an interface
class bayes_model:
    log = logging.getLogger(__name__)
    def me(self, data, *pars, **kwargs):
        """
        the mean error of the model from the data.
        used to check for biases.
        """
        return self._me(data, *pars, **kwargs)
    
    def mse(self, data, *pars, **kwargs):
        """
        the data is assumed to be sequential
        """
        # print(np.array(data))
        return self._mse(np.array(data), *pars, **kwargs)

    def mae(self, data, *pars, **kwargs):
        return self._mae(np.array(data), *pars, **kwargs)
        
    def kld(self, data, *pars, **kwargs):
        """
        data is assumed to be sequential
        kullback-leibler divergence for each point, with updating of bayesian parameters if necessary
        """
        return self._kld(np.array(data), *pars, **kwargs)

    def evse(self, data, *pars, **kwargs):
        """
        returns the squared difference of the data from the distribution's expectation value
        """
        return (np.array(data) - self.evs(data))**2
    
    def evs(self, data):
        # list of expectation values at each point
        return self._evs(np.array(data))

    def vars(self, data):
        # list of variances of distribution at each point
        return self._vars(np.array(data))

    def residuals(self, data):
        return (np.array(data) - self.evs(data))/np.sqrt(self.vars(data))
        
## constant gaussian w/ mean and variance
class const_gauss_model(bayes_model):
    """
    This model uses a single constant for all predictions.
    Useful as a baseline (often the simple average of all data)
    """
    def __init__(self, mu, var=0.):
        self.mu = mu
        self.var = var

    def _evs(self, data):
        return np.full(data.shape, self.mu)
        
    def _vars(self, data):
        return np.full(data.shape, self.var)
    
    def _mse(self, data):
        return (self.mu-data)**2 + self.var

    def _mae(self, data):
        raise NotImplementedError('technically this is implemented, but it is quite slow')
        # we can loop through manually, at least
        maes = []
        # this takes a looong time and may be inaccurate:
        # we should be able to calculate it manually, in terms of erf
        for d in data:
            maes.append(st.norm.expect(lambda x: np.abs(x-d), loc=self.mu, scale=np.sqrt(self.var)))
        return np.array(maes)
    
    def _kld(self, data):
        if self.var == 0:
            log.warning('zero in variance for KL divergence')
            return np.array([-np.inf if d == self.mu else np.inf for d in data])
        return (self.mu-data)**2/(2.0*self.var) + 0.5*np.log(2.0*np.pi*self.var)
        

# the model is defined by its posterior predictive distribution, which are compound distributions for easy updating of parameters.
# beta-binomial is a compound of beta with binomial.
# for new observation k, the model is updated:
# alpha -> alpha + k
# beta -> beta + (n-k)
# alpha and beta correspond to those for the beta-binomial (for the posterior predictive),
# or the beta distribution for the prior.
class beta_binomial_model(bayes_model):
    def __init__(self, n, alpha0, beta0, lr=1.0):
        self.n = n
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.lr = lr
    
    # returns EVs
    def _evs(self, data, n, alpha0, beta0, lr=1.0):
        alphas,betas = _get_abs(data)
        return n*alphas/(alphas + betas)

    # returns variances of model
    def _vars(self, data, n, alpha0, beta0, lr=1.0):
        alphas,betas = _get_abs(data)
        return n*alphas*betas*(alphas+betas+n)/(alphas+betas)**2/(alphas+betas+1)
    
    def _mse(self, data):
        # lr can be used to suppress learning
        # we could also apply multiplicative decay after adding, which will result in variance decay even in long careers
        assert((data >= 0).all() and (data <= n).all())
        mses = []
        alphas,betas = self._get_abs(data)
        # domain of summation for EV computation:
        support = np.arange(0,n+1)
        for d,a,b in zip(data,alphas,betas):
            # beta-binomial isn't in scipy - there is an open ticket; the CDF is difficult to implement
            # it's not so bad to compute manually since the domain is finite
            probs = dist_fit.beta_binomial( support, n, a, b )
            mses.append( sum(probs*(support-d)**2) )
        # log.debug('alpha, beta = {},{}'.format(alpha,beta))
        return np.array(mses)

    # mean absolute error
    def _mae(self, data):
        assert((data >= 0).all() and (data <= n).all())
        maes = []
        alphas,betas = self._get_abs(data)
        support = np.arange(0,n+1)
        for d,a,b in zip(data,alphas,beta):
            probs = dist_fit.beta_binomial( support, n, a, b )
            maes.append( sum(probs*np.abs(support-d)) )
        return np.array(maes)

    # computes Kullback-Leibler divergence
    def _kld(self, data, n):
        assert((data >= 0).all() and (data <= n).all())
        alphas,betas = self._get_abs(data)
        return -dist_fit.log_beta_binomial( data, n, alphas, betas )
    
    def _get_abs(self, data):
        alphas = [self.alpha0]
        betas = [self.beta0]
        assert(len(data.shape) <= 2)
        for d in data[:-1]:
            alphas.append(alphas[-1] + self.lr*d)
            betas.append(betas[-1] + self.lr*(n-d))
        # these have one additional element which is not used. it may or may not matter.
        return np.array(alphas),np.array(betas)

    # return the updated model parameters from after the data
    def get_model_pars(self, data):
        alpha = self.alpha0 + self.lr*sum(data)
        beta = self.beta0 + self.lr*sum(n-data)
        return alpha,beta



## the negative binomial posterior predictor is a gamma convolved with a poisson
## this should be appropriate for a poisson w/ unknown rate, like TDs per game for an RB
## the update rules are:
# r -> r + lr*k
# beta -> beta + lr where beta = p/(1-p) so:
#   p = (beta0+n)/(1+beta0+n)
# where if lr < 1 we are slowing the update
# we could also apply multiplicative decay after adding, which will result in variance decay even in long careers
class neg_binomial_model(bayes_model):
    def __init__(self, r0, p0, lr=1.0):
        self.r0 = r0
        self.p0 = p0
        self.lr = lr
        
    def _evs(self, data):
        rs,ps = self._get_rps(data)
        evs = rs*(1/ps-1)
        assert(len(evs.shape) == 1)
        return evs

    def _vars(self, data):
        rs,ps = self._get_rps(data)
        rdata = data if len(data.shape) == 1 else data[0] / data[1]
        return rs*(1-ps)/ps**2
    
    def evse(self, data, *pars, **kwargs):
        """
        we need to overwrite this for the case where two data cols are provided
        """
        darray = np.array(data)
        rdata = darray if len(darray.shape) == 1 else darray[0] / darray[1]
        return (rdata - self.evs(darray))**2
    
    def residuals(self, data):
        # this also needs to be overwritten
        darray = np.array(data)
        rdata = darray if len(darray.shape) == 1 else darray[0] / darray[1]
        return (rdata - self.evs(data))/np.sqrt(self.vars(data))
    
    def _mse(self, data):
        mses = []
        # note these lists should be 1 longer than the data
        rs,ps = self._get_rps(data)
        r_data = data if len(data.shape) == 1 else data[0] / data[1]
        # domain of summation for EV computation - extend out to 5 significance
        for d,r,p in zip(r_data,rs,ps):
            # note that wikipedia has a different p convention than most... p -> 1-p
            nbm,nbv = st.nbinom.stats(r,p) # get mean and variance of distribution
            # maxct = 5*((1-p)*r)**(1.5)/p**2 # not clear that computing this initially is the best
            maxct = nbm + 5*np.sqrt(nbv)
            mse = st.nbinom.expect(lambda k: (k-d)**2, args=(r,p), maxcount=max(1000,maxct))
            mses.append( mse )
        return np.array(mses)

    def _mae(self, data, *pars):
        raise NotImplementedError

    # computes Kullback-Leibler divergence
    def _kld(self, data):
        # i think the KLD for the per-game distribution cannot be calculated w/out weekly data
        # we only have 1 update per season, so to scale properly this should go like this:?
        rs,ps = self._get_rps(data)
        r_data = data if len(data.shape) == 1 else data[0] # / data[1] # these must be integers!
        # the sum of N variables w/ NB distribution goes like NB(sum(r), p)
        # so we can adjust the rs by scaling up by the number of games played
        # check that this is close to the analytic continuation?
        # if using continuous version, we need to use the once/season update (beta += 1)
        result_ac = -dist_fit.log_neg_binomial(data[0]/data[1], rs, ps) # these are not the same, nor really close
        # if len(data.shape) == 2:
        #     rs = rs*data[1] # scale by denominator (e.g. "games started")
        # result = -st.nbinom.logpmf(r_data, rs, ps)
        # self.log.error('\nAC: {}\nscaled NB: {}'.format(result_ac, result))
        return result_ac
        
    def _get_rps(self, data):
        rs = [self.r0]
        bs = [self.p0/(1-self.p0)]
        assert(len(data.shape) <= 2)
        if len(data.shape) == 2:
            for dn,dd in data.T[:-1]:
                # rs.append(rs[-1]+self.lr*dn)
                # bs.append(bs[-1]+self.lr*dd)
                # this is necessary if using the continuous approximation
                rs.append(rs[-1]+self.lr*dn/dd)
                bs.append(bs[-1]+self.lr)
        else:
            for d in data[:-1]:
                rs.append(rs[-1]+self.lr*d)
                bs.append(bs[-1]+self.lr)
        ps = [b/(1+b) for b in bs] # np.array(bs)/(1.0+np.array(bs))
        # these probably have one additional element which is not used. it may or may not matter.
        return np.array(rs),np.array(ps)

    def get_model_pars(self, data):
        r = self.r0
        b = self.p0/(1-self.p0)
        assert(len(data.shape) <= 2)
        if len(data.shape) == 2:
            # r += self.lr*sum(data[0])
            # b += self.lr*sum(data[1])
            r += self.lr*dn/dd
            b += self.lr*data.shape[1]
        else:
            r += self.lr*sum(data)
            b += self.lr*len(data)
        p = b/(1+b)
        return r,p


# the predictive posterior is a student's t distribution.
# the posterior is a gaussian w/ unknown mean and variance
class t_model(bayes_model):
    def __init__(self, mu0, nu0, alpha0, beta0, lr=1.0):
        self.mu0 = mu0
        self.nu0 = nu0
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.lr = lr
        
    def _evs(self, data):
        mus,_,_,_ = self._get_bps(data)
        # technically the mean is not defined if alpha < 1/2,
        # but this isn't supposed to be taken as a rigorous quantity anyway
        return mus

    def _vars(self, data):
        _,nus,alphas,betas = self._get_bps(data)
        # technically the variance is not defined if alpha < 1
        return betas/(alphas-1)*(1+1.0/nus)
    
    def _get_bps(self, data):
        """
        get the 4 bayes parameters for the normal distribution
        """
        # learn rate is probably redundant for this, since it's effectively a change in nu0 (and alpha0?)
        mus = [self.mu0]
        nus = [self.nu0]
        alphas = [self.alpha0]
        betas = [self.beta0]
        lr = self.lr
        runCount = 0
        runSum = 0
        runSumSq = 0
        for d in data[:-1]:
            runCount += 1
            runSum += d
            runSumSq += d**2
            runMean = runSum/runCount
            nup = nus[-1] + lr
            betap = self.beta0 + lr*0.5*(runSumSq - runCount*runMean**2) + lr*0.5*runCount*self.nu0/nup*(runMean - self.mu0)**2
            mus.append( (nus[-1]*mus[-1] + lr*d)/nup )
            nus.append(nup)
            alphas.append(alphas[-1] + 0.5*lr)
            betas.append(betap)
        return np.array(mus),np.array(nus),np.array(alphas),np.array(betas)

    def _mse(self, data):
        mus,nus,alphas,betas = self._get_bps(data)
        return (data-mus)**2 + betas/(alphas-1)*(1+1.0/nus)

    def _mae(self, data):
        raise NotImplementedError

    # computes Kullback-Leibler divergence
    def _kld(self, data):
        mus,nus,alphas,betas = self._get_bps(data)
        # sigma**2 is not a variance of the t-distribution, it's a scale factor
        sigmasq = betas/alphas*(1+1.0/nus)
        result = -st.t.logpdf(data, 2*alphas, loc=mus,
                              scale=np.sqrt(sigmasq))
        return result

    def get_model_pars(self, data):
        """
        get the 4 bayes parameters for the normal distribution after updating for the data
        """
        ndata = len(data)
        lr = self.lr
        datasum = np.sum(data)
        datamean = datasum/ndata
        mu = (self.nu0*self.mu0 + lr*datasum)/(self.nu0 + lr*ndata)
        nu = self.nu0 + lr
        alpha = self.alpha0 + 0.5*lr
        beta = self.beta0 + 0.5*lr*np.sum((data-datamean)**2) + 0.5*lr*ndata*self.nu0/nu*(datamean-self.mu0)**2
        return mu,nu,alpha,beta
