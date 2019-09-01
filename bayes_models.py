import logging
import dist_fit
import numpy as np
import scipy.stats as st

#### at this point, these are mostly superseded by the code in playermodels,
##   which models game-by-game.

# we don't really need to define an interface
class bayes_model:
    log = logging.getLogger(__name__)
    def me(self, data, *args, **kwargs):
        """
        the mean error of the model from the data.
        used to check for biases.
        """
        return self._me(data, *args, **kwargs)
    
    def mse(self, data, *args, **kwargs):
        """
        the data is assumed to be sequential
        """
        return self._mse(np.array(data), *args, **kwargs)

    def mae(self, data, *args, **kwargs):
        return self._mae(np.array(data), *args, **kwargs)
        
    def kld(self, data, *args, **kwargs):
        """
        data is assumed to be sequential
        kullback-leibler divergence for each point, with updating of bayesian parameters if necessary
        """
        return self._kld(np.array(data), *args, **kwargs)

    def evse(self, data, *args, **kwargs):
        """
        returns the squared difference of the data from the distribution's expectation value
        """
        darr = np.array(data)
        d = darr if len(darr.shape) == 1 else darr[0] / darr[1]
        return (d - self.evs(darr, *args, **kwargs))**2
    
    def evs(self, data, *args, **kwargs):
        # list of expectation values at each point
        return self._evs(np.array(data), *args, **kwargs)

    def vars(self, data, *args, **kwargs):
        # list of variances of distribution at each point
        return self._vars(np.array(data), *args, **kwargs)

    def residuals(self, data, *args, **kwargs):
        darr = np.array(data)
        d = darr if len(darr.shape) == 1 else darr[0] / darr[1]
        return (d - self.evs(darr, *args, **kwargs))/np.sqrt(self.vars(darr, *args, **kwargs))
        
## constant gaussian w/ mean and variance
class const_gauss_model(bayes_model):
    """
    This model uses a single constant for all predictions.
    Useful as a baseline (often the simple average of all data)
    """
    def __init__(self, mu, var=0.):
        self.mu = mu
        self.var = var

    def _evs(self, data, weights=None):
        return np.full(data.shape, self.mu)
        
    def _vars(self, data, weights=None):
        return np.full(data.shape, self.var)
    
    def _mse(self, data, weights=None):
        return (self.mu-data)**2 + self.var

    def _mae(self, data, weights=None):
        raise NotImplementedError('technically this is implemented, but it is quite slow')
        # we can loop through manually, at least
        maes = []
        # this takes a looong time and may be inaccurate:
        # we might be able to calculate it manually, in terms of erf
        for d in data:
            maes.append(st.norm.expect(lambda x: np.abs(x-d), loc=self.mu, scale=np.sqrt(self.var)))
        return np.array(maes)
    
    def _kld(self, data, weights=None, normalize=False):
        if self.var == 0:
            log.warning('zero in variance for KL divergence')
            return np.array([-np.inf if d == self.mu else np.inf for d in data])
        result = (self.mu-data)**2/(2.0*self.var)
        if not normalize: result += 0.5*np.log(2.0*np.pi*self.var)
        return result
        

# the model is defined by its posterior predictive distribution, which are compound distributions for easy updating of parameters.
# beta-binomial is a compound of beta with binomial.
# if n == 1, the predictive distribution is bernoulli.
# for new observation k, the model is updated:
# alpha -> alpha + k
# beta -> beta + (n-k)
# alpha and beta correspond to those for the beta-binomial (for the posterior predictive),
# or the beta distribution for the prior.
class beta_binomial_model(bayes_model):
    def __init__(self, n, alpha0, beta0, lr=1.0, mem=1.0):
        self.n = n
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.lr = lr
        self.mem = mem
    
    # returns EVs
    def _evs(self, data, weights=None):
        alphas,betas = self._get_abs(data, weights)
        return self.n*alphas/(alphas + betas)

    # returns variances of model
    def _vars(self, data, weights=None):
        alphas,betas = self._get_abs(data, weights)
        return self.n*alphas*betas*(alphas+betas+self.n)/(alphas+betas)**2/(alphas+betas+1)
    
    def _mse(self, data, weights=None):
        # lr can be used to suppress learning
        # we could also apply multiplicative decay after adding, which will result in variance decay even in long careers
        assert((data >= 0).all() and (data <= self.n).all())
        mses = []
        alphas,betas = self._get_abs(data, weights)
        # domain of summation for EV computation:
        support = np.arange(0,self.n+1)
        for d,a,b in zip(data,alphas,betas):
            # beta-binomial isn't in scipy - there is an open ticket; the CDF is difficult to implement
            # it's not so bad to compute manually since the domain is finite
            probs = dist_fit.beta_binomial( support, self.n, a, b )
            mses.append( sum(probs*(support-d)**2) )
        # log.debug('alpha, beta = {},{}'.format(alpha,beta))
        return np.array(mses)

    # mean absolute error
    def _mae(self, data, weights=None):
        assert((data >= 0).all() and (data <= self.n).all())
        maes = []
        alphas,betas = self._get_abs(data, weights)
        support = np.arange(0,self.n+1)
        for d,a,b in zip(data,alphas,betas):
            probs = dist_fit.beta_binomial( support, self.n, a, b )
            maes.append( sum(probs*np.abs(support-d)) )
        return np.array(maes)

    # computes Kullback-Leibler divergence
    def _kld(self, data, weights=None, normalize=False):
        assert((data >= 0).all() and (data <= self.n).all())
        alphas,betas = self._get_abs(data, weights)
        result = -dist_fit.log_beta_binomial( data, self.n, alphas, betas )
        if normalize:
            # evs = self.n*alphas/(alphas + betas)
            raise NotImplementedError('not sure how to normalize likelihood for discrete distribution')
        return result
        
    def _get_abs(self, data, weights=None):
        assert(len(data.shape) == 1), 'data must be 1-dimensional'
        ws = np.full(data.shape, self.lr)
        if weights is not None: ws *= weights
        alphas = [self.alpha0]
        betas = [self.beta0]
        for d,w in zip(data[:-1], ws):
            alphas.append(self.mem*alphas[-1] + w*d)
            betas.append(self.mem*betas[-1] + w*(self.n-d))
        return np.array(alphas),np.array(betas)

    # return the updated model parameters from after the data
    def get_model_pars(self, data, weights=None):
        ndata = data.shape
        ws = np.full(ndata, self.lr)
        if weights is not None: ws *= weights
        weights[::-1] *= np.asarray([self.mem**i for i in range(ndata)])
        alpha = self.alpha0*self.mem**ndata + sum(ws*data)
        beta = self.beta0*self.mem**ndata + sum(ws*(self.n-data))
        return alpha,beta


# this model is appropriate for a percentage, like pass completion %.
# it predicts a value between 0 and 1.
# the number of completions for a set number of attempts is beta-binomial distributed.
# to keep from getting too narrow, a reduced learning rate and/or memory is probably appropriate
class beta_model(bayes_model):
    def __init__(self, alpha0, beta0, lr=1.0, mem=1.0):
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.lr = lr
        self.mem = mem
    
    # returns EVs
    def _evs(self, data, weights=None):
        alphas,betas = self._get_abs(data, weights)
        return alphas/(alphas + betas)

    # returns variances of model
    def _vars(self, data, weights=None):
        alphas,betas = self._get_abs(data, weights)
        return alphas*betas/(alphas+betas)**2/(alphas+betas+1)
    
    def _mse(self, data, weights=None):
        alphas,betas = self._get_abs(data, weights)
        rdata = data if len(data.shape) == 1 else data[0]/data[1]
        return rdata**2 + alphas/(alphas+betas)*(-2*rdata + (1+alphas)/(1+alphas+betas))

    # mean absolute error
    def _mae(self, data, weights=None):
        raise NotImplementedError

    # computes Kullback-Leibler divergence
    def _kld(self, data, weights=None, normalize=False):
        alphas,betas = self._get_abs(data, weights)
        rdata = data if len(data.shape) == 1 else data[0] / data[1]
        result = -st.beta.logpdf( rdata, alphas, betas )
        if normalize:
            raise NotImplementedError('not sure how to normalize likelihood for beta distribution')
        return result
        
    def _get_abs(self, data, weights=None):
        # this can support a ratio
        assert(len(data.shape) <= 2)
        ndata = data.shape[-1]
        ws = np.full(ndata, self.lr)
        if weights is not None: ws *= weights
        alphas = [self.alpha0]
        betas = [self.beta0]
        if len(data.shape) == 2:
            for (dn,dd),w in zip(data.T[:-1],ws):
                alphas.append(self.mem*alphas[-1] + w*dn)
                betas.append(self.mem*betas[-1] + w*(dd-dn))
                # alphas.append(self.mem*alphas[-1]+w*dn/dd)
                # betas.append(self.mem*betas[-1]+w)
        else:
            for d,w in zip(data[:-1],ws):
                alphas.append(self.mem*alphas[-1]+w*d)
                betas.append(self.mem*betas[-1]+w*(1-d))            
        return np.array(alphas),np.array(betas)

    # return the updated model parameters from after the data
    def get_model_pars(self, data, weights=None):
        ndata = data.shape[-1]
        ws = np.full(ndata, self.lr)
        if weights is not None: ws *= weights
        weights[::-1] *= np.asarray([self.mem**i for i in range(ndata)])
        alpha = self.alpha0*self.mem**ndata
        beta = self.beta0*self.mem**ndata
        # return alpha,beta
        if len(data.shape) == 2:
            alpha += sum(ws*data[0])
            beta += sum(ws*(data[1]-data[0]))
        else:
            alpha += sum(ws*data)
            beta += sum(ws*(1-data))


## the negative binomial posterior predictor is a gamma convolved with a poisson
## this should be appropriate for a poisson w/ unknown rate, like TDs per game for an RB
## the update rules are:
# r -> r + lr*k
# beta -> beta + lr where beta = p/(1-p) so:
#   p = (beta0+n)/(1+beta0+n)
# where if lr < 1 we are slowing the update
# we could also apply multiplicative decay after adding, which will result in variance decay even in long careers
class neg_binomial_model(bayes_model):
    def __init__(self, r0, p0, lr=1.0, mem=1.0):
        self.r0 = r0
        self.p0 = p0
        self.lr = lr # learn rate
        self.mem = mem # memory factor
        
    def _evs(self, data, weights=None):
        rs,ps = self._get_rps(data, weights)
        evs = rs*(1/ps-1)
        assert(len(evs.shape) == 1)
        return evs

    def _vars(self, data, weights=None):
        rs,ps = self._get_rps(data, weights)
        rdata = data if len(data.shape) == 1 else data[0] / data[1]
        return rs*(1-ps)/ps**2
    
    def _mse(self, data, weights=None):
        mses = []
        # note these lists should be 1 longer than the data
        rs,ps = self._get_rps(data, weights)
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
    def _kld(self, data, weights=None, normalize=False):
        # i think the KLD for the per-game distribution cannot be calculated w/out weekly data
        # we only have 1 update per season, so to scale properly this should go like this:?
        rs,ps = self._get_rps(data, weights)
        r_data = data if len(data.shape) == 1 else data[0] # / data[1] # these must be integers!
        # the sum of N variables w/ NB distribution goes like NB(sum(r), p)
        # so we can adjust the rs by scaling up by the number of games played
        # check that this is close to the analytic continuation?
        # if using continuous version, we need to use the once/season update (beta += 1)
        result_ac = -dist_fit.log_neg_binomial(data[0]/data[1], rs, ps) # these are not the same, nor really close
        if normalize:
            evs = rs*(1/ps-1)
            result_ac += dist_fit.log_neg_binomial(evs, rs, ps)
        # if len(data.shape) == 2:
        #     rs = rs*data[1] # scale by denominator (e.g. "games started")
        # result = -st.nbinom.logpmf(r_data, rs, ps)
        # self.log.error('\nAC: {}\nscaled NB: {}'.format(result_ac, result))
        return result_ac
        
    def _get_rps(self, data, weights=None):
        ndata = data.shape[-1]
        ws = np.full(ndata, self.lr)
        if weights is not None: ws *= weights
        rs = [self.r0]
        bs = [self.p0/(1-self.p0)]
        assert(len(data.shape) <= 2)
        if len(data.shape) == 2:
            for (dn,dd),w in zip(data.T[:-1],ws):
                # rs.append(self.mem*rs[-1]+w*dn)
                # bs.append(self.mem*bs[-1]+w*dd)
                # this is necessary if using the continuous approximation
                rs.append(self.mem*rs[-1]+w*dn/dd)
                bs.append(self.mem*bs[-1]+w)
        else:
            for d,w in zip(data[:-1],ws):
                rs.append(self.mem*rs[-1]+w*d)
                bs.append(self.mem*bs[-1]+w)
        ps = [b/(1+b) for b in bs] # np.array(bs)/(1.0+np.array(bs))
        # these probably have one additional element which is not used. it may or may not matter.
        return np.array(rs),np.array(ps)

    def get_model_pars(self, data, weights=None):
        ndata = data.shape[-1]
        ws = np.full(ndata, self.lr)
        if weights is not None: ws *= weights
        weights[::-1] *= np.asarray([self.mem**i for i in range(ndata)])
        r = self.r0 * self.mem**ndata
        b = self.p0/(1-self.p0) * self.mem**ndata
        assert(len(data.shape) <= 2)
        if len(data.shape) == 2:
            # r += sum(ws*data[0])
            # b += sum(ws*data[1])
            r += sum(ws*data[0]/data[1])
            b += sum(ws)
        else:
            r += sum(ws*data)
            b += len(ws*data)
        p = b/(1+b)
        return r,p

    
# the predictive posterior is a student's t distribution.
# the posterior is a gaussian w/ unknown mean and variance
class t_model(bayes_model):
    def __init__(self, mu0, nu0, alpha0, beta0, lr=1.0, mem=1.0):
        self.mu0 = mu0
        self.nu0 = nu0
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.lr = lr
        self.mem = mem
        
    def _evs(self, data, weights=None):
        mus,_,_,_ = self._get_bps(data, weights)
        # technically the mean is not defined if alpha < 1/2,
        # but this isn't supposed to be taken as a rigorous quantity anyway
        return mus

    def _vars(self, data, weights=None):
        _,nus,alphas,betas = self._get_bps(data, weights)
        # technically the variance is not defined if alpha < 1
        return betas/(alphas-1)*(1+1.0/nus)
    
    def _get_bps(self, data, weights=None):
        """
        get the 4 bayes parameters for the normal distribution
        """
        # learn rate is probably redundant for this, since it's effectively a change in nu0 (and alpha0?)
        ndata = data.shape
        ws = np.full(ndata, self.lr)
        if weights is not None: ws *= weights
        mus = [self.mu0]
        nus = [self.nu0]
        alphas = [self.alpha0]
        betas = [self.beta0]
        runCount = 0
        runSum = 0
        runSumSq = 0
        for d,w in zip(data[:-1],ws):
            wp = self.mem # weight for previous data
            runCount = wp*runCount + w
            runSum = wp*runSum + w*d
            runSumSq = wp*runSumSq + w*d**2
            runMean = runSum/runCount
            nup = wp*nus[-1] + w
            # the beta term is a bit difficult. with w.g. 0.5 memory, the variance blows up.
            # this may just be how it works -- w/ less information the variance should explode
            betap = wp*self.beta0 + 0.5*(runSumSq - runCount*runMean**2) + \
                    0.5*runCount*self.nu0/nup*(runMean - self.mu0)**2 # not sure how to handle these mu0/nu0 w/ forgetting...
            mus.append( (wp*nus[-1]*mus[-1] + w*d)/nup )
            nus.append(nup)
            alphas.append(wp*alphas[-1] + 0.5*w)
            betas.append(betap)
        return np.array(mus),np.array(nus),np.array(alphas),np.array(betas)

    def _mse(self, data, weights=None):
        mus,nus,alphas,betas = self._get_bps(data, weights)
        return (data-mus)**2 + betas/(alphas-1)*(1+1.0/nus)

    def _mae(self, data, weights=None):
        raise NotImplementedError

    # computes Kullback-Leibler divergence
    def _kld(self, data, weights=None, normalize=False):
        mus,nus,alphas,betas = self._get_bps(data, weights)
        # sigma**2 is not a variance of the t-distribution, it's a scale factor
        sigmas = np.sqrt(betas/alphas*(1+1.0/nus))
        result = -st.t.logpdf(data, 2*alphas, loc=mus, scale=sigmas)
        if normalize: result += st.t.logpdf(0, 2*alphas, scale=sigmas)
        return result

    def get_model_pars(self, data, weights=None):
        """
        get the 4 bayes parameters for the normal distribution after updating for the data
        """
        ndata = data.shape
        ws = np.full(ndata, self.lr)
        if weights is not None: ws *= weights
        weights[::-1] *= np.asarray([self.mem**i for i in range(ndata)])
        wsum = np.sum(ws)
        datasum = np.sum(ws*data)
        datamean = datasum/wsum
        w0 = self.mem**ndata
        nu = w0*self.nu0 + wsum
        mu = (w0*self.nu0*self.mu0 + datasum)/nu
        alpha = w0*self.alpha0 + 0.5*wsum
        # i think this way of dealing w/ memory in the last term for beta makes sense...
        beta = w0*self.beta0 + 0.5*np.sum(ws*(data-datamean)**2) + 0.5*wsum*self.nu0/nu*(datamean-self.mu0)**2
        return mu,nu,alpha,beta
