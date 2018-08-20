# will define model templates and provide some common functionality
import numpy as np
import scipy.stats as st
# from scipy.special import gamma, digamma
import dist_fit

class Model:
    """
    base class w/ just a bit of common and default functionality
    """
    @classmethod
    def for_position(self, pos):
        return self(*self._default_hyperpars(pos))

    @property
    def var_names(self):
        var_names = [self.pred_var] + list(self.dep_vars)
        return var_names
    

# TODO: implement scipy.stats.rv_discrete ?
class CountsModel(Model):
    """
    used to model a number of discrete events (like attempts per game).
    the predictive posterior is a negative binomial.
    the learning parameters allow some variance, but it cannot describe different players having different variances.
    a generalized beta-negative binomial could be investigated, but the bayesian update rules are not so clear.
    """
    def __init__(self,
                 a0, b0, # initial bayes parameters
                 lr,     # learn rate per game
                 mem, gmem, # memory per season and game
    ):
        self.ab = np.array((a0,b0))
        # we might want game_lr to be a function of the season?
        self.game_lr = lr
        self.season_mem = mem
        self.game_mem = gmem
    
    @classmethod
    def _hyperpar_bounds(self):
        return [(1e-6,None),(1e-6,None),(0,1),(0.1,1.0),(0.5,1.0)]
    
    # a shortcut for a re-mapping of beta that comes up a lot due to the common convention for negative binomial
    def _p(self):
        # scipy convention for p (not wiki)
        beta = self.ab[1]
        return beta/(1.+beta)

    def update_game(self, att):
        self.ab *= self.game_mem
        self.ab += self.game_lr * np.array((att, 1.))
        # we could accumulate a KLD to diagnose when the model has been very off recently

    def new_season(self):
        self.ab *= self.season_mem
        
    def ppf(self, uni):
        # this yields a gamma convoluted w/ a poisson
        att = st.nbinom.ppf(uni, self.ab[0], self._p())
        return att

    def cdf(self, att):
        cdf = st.nbinom.cdf(att, self.ab[0], self._p())
        return cdf
    
    def ev(self):
        return self.ab[0] / self.ab[1]

    def var(self):
        return self.ev()/self._p()

    # remember that we don't want to minimize the chi-sq, we want to minimize the KLD
    # this will attempt to normalize, but will sometimes still end up negative
    def chi_sq(self, att):
        norm = dist_fit.log_neg_binomial(self.ev(), self.ab[0], self._p())
        return 2.*(self.kld(att) + norm)
    
    def kld(self, att):
        return - st.nbinom.logpmf(att, self.ab[0], self._p())

    def __str__(self):
        pars = u'\u03B1\t= {:.2f}\n\u03B2\t= {:.2f}\n'.format(*self.ab)
        pars += 'lr\t= {:.3f}\nmem\t= {:.3f}\ngmem\t= {:.3f}\n'.format(self.game_lr, self.season_mem, self.game_mem)
        return pars


class TrialModel(Model):
    """
    a model for a discrete number of successes given an integer number of trials.
    e.g. TDs per reception
    the posterior is a beta-binomial, where n is provided as the number of attempts.
    """
    def __init__(self,
                 a0, b0,
                 lr,
                 mem,gmem,
    ):
        self.ab = np.array((a0,b0))
        # we might want game_lr to be a function of the season?
        self.game_lr = lr
        self.season_mem = mem
        self.game_mem = gmem

    @classmethod
    def _hyperpar_bounds(self):
        return [
            (1e-6,None),(1e-6,None), # small positive lower bounds can prevent the fitter from going to invalid locations
            (0.0,1.0), #  should probably cap learn rate to prevent overfitting
            (0.2,1.0),(0.5,1.0)
        ]

    def _p(self):
        return self.ab[0] / self.ab.sum()
        
    def update_game(self, succ, att):
        self.ab *= self.game_mem
        self.ab += self.game_lr * np.array((succ, att - succ))

    def new_season(self):
        self.ab *= self.season_mem
        
    def ev(self, att):
        return self._p() * att

    def var(self, att):
        a,b = self.ab[0],self.ab[1]
        apb = a + b
        var = att*a*b*(apb+att)/(apb**2*(apb+1))
        return var

    def ppf(self, att, uni):
        succ = 0
        cdf = dist_fit.beta_binomial(0, att, *self.ab)
        while cdf < uni:
            succ += 1
            cdf += dist_fit.beta_binomial(succ, att, *self.ab)
        return succ
    
    def cdf(self, succ, att):
        # CDF is the % of the mass that is *at or equal to* the value.
        # these CDFs will not be flat, since most results are 0 and that's most of the way up the CDF already
        # this is the definition we want, however, for analyzing the correlations
        cdf = 0.
        check = 0
        while check <= succ:
            cdf += dist_fit.beta_binomial(check, att, *self.ab)
            check += 1
        return cdf
        
    def chi_sq(self, succ, att):
        # if att == 0: return 0.
        norm = dist_fit.log_beta_binomial( self.ev(att), att, *self.ab)
        return 2.*(self.kld(succ, att) + norm)
    
    def kld(self, succ, att):
        if att == 0: return 0.
        if succ > att: att = succ
        return - dist_fit.log_beta_binomial( succ, att, *self.ab)

    def __str__(self):
        pars = u'{:.2f}% rate\n\u03B1\t= {:.2f}\n\u03B2\t= {:.2f}\n'.format(100*self._p(), *self.ab)
        hpars = 'lr\t= {:.3f}\n'.format(self.game_lr)
        hpars += 'smem\t= {:.3f}\ngmem\t= {:.3f}\n'.format(self.season_mem, self.game_mem)
        return pars + hpars

class YdsPerAttModel(Model):
    """
    model for yards per attempt (though it could be more general than that).
    it uses a non-central student-t distribution to allow positive skew.
    the skewness goes to zero as the number of samples rises.
    """
    def __init__(self,
                 mn0, n0, a0, b0,
                 skew,
                 lr1, lr2,
                 mnmem, # memory decay per season for munu / nu
                 abmem, # seasonal parameter decay for a/b
                 mngmem, # game memory for munu/nu. doesn't seem to help much.
                 abgmem, # game memory for a/b
    ):
        # this represents (mu*nu, nu, alpha, beta). note that we save only mu*nu, for simpler decay.
        self.mnab = np.array((mn0, n0, a0, b0))
        # the skewness will be a constant hyperparameter for now,
        # as it's not clear how to do bayesian updating w/ non-centrality.
        # to get a good value for skewness directly, we'd need play-by-play data.
        self.skew = skew

        # we might want game_lr to be a function of the season?
        # using different learn rates for mu/nu and alpha/beta (i.e. 1 and 2 moments)
        # possibly different learn rates for all of them? tie to memory?
        self.game_lr = np.repeat((lr1,lr2), 2)
        self.season_mem = np.repeat((mnmem, abmem), 2)
        self.game_mem = np.repeat((mngmem, abgmem), 2)

    @classmethod
    def _hyperpar_bounds(self):
        return [
            (1e-6,None),(1e-6,None),(1e-6,None),(1e-6,None),
            (0.0,8.0), # skew
            (0.0,1.0),(0.0,1.0), # learn rates. lr for mean is actually much > 1 for QBs; we'll cap to avoid overfitting
            (0.2,1.0),(0.4,1.0), # season memory
            (0.5,1.0),(0.5,1.0), # game memory - doesn't help much
        ]
        
    def update_game(self, rush_yds, att):
        # assert((0 < self.game_mem).all() and (self.game_mem <= 1.0).all())
        # mu does not decay simply like the others, but mu*nu does
        ev = self.ev(att)
        self.mnab *= self.game_mem
        self.mnab += self.game_lr * np.array((rush_yds, att,
                                              0.5*att,
                                              0.5*(rush_yds-ev)**2/max(1,att)))
        # the max() function is just to avoid a divide-by-zero error when everything is zero
        # we could accumulate a KLD to diagnose when the model has been very wrong recently

    def new_season(self):
        self.mnab *= self.season_mem

    def _df(self, att):
        # we should probably use the # of attempts, but this choice can be overridden
        return att
        
    # helper function for common parameter
    def _sigma2(self):
        nu,alpha,beta = tuple(self.mnab[1:])
        return beta*(nu+1.)/(nu*alpha)

    def _ncmean(self, att):
        # the built-in mean function calls stats, which also computes variance and gives a warning for df = 2
        # ncmean = st.nct.mean(df, nc) if df > 1 else nc
        df = self._df(att)
        nc = self.skew
        if df <= 1: return nc
        # ncmean = nc*np.sqrt(0.5*df)*gamma(0.5*(df-1))/gamma(0.5*df)
        # a good approximation is:
        ncmean = nc/(1 - 3/(4*df-1))
        return ncmean

    def _loc(self, att):
        df = self._df(att)
        ncmean = self._ncmean(att)
        return (self.mnab[0] / self.mnab[1] - ncmean) * att
    
    def _scale(self, att):
        df = self._df(att)
        sigma = np.sqrt(self._sigma2())
         # prevent a zero to remove a warning, even tho it doesn't change any calculations:
        if df == 0: return sigma
        scale = att*sigma
        if df > 2:
            nc = self.skew
            # scale /= ( df*(1+nc**2)/(df-2) - nc**2*df/2*(gamma((df-1)/2)/gamma(df/2))**2 )
            # a good approximation:
            scale /= ( df*(1+nc**2)/(df-2) - nc**2/(1-3/(4*df-1))**2 )
        return scale

    # given a uniformly distributed number 0 < uni < 1, return the yards corresponding to that point on the cdf.
    # correlations can be dealt with externally, since it's much easier to correlate normal variables.
    def ppf(self, att, uni):
        assert(0 < uni < 1)
        if att == 0: return 0 # we won't try to simulate laterals
        df,nc = att,self.skew
        yds = st.nct.ppf(uni, df, nc,
                              loc=self._loc(att),
                              scale=self._scale(att),)
        return yds
    
    def ev(self, att):
        # the mean is mu*nu / nu
        munu = self.mnab[0]
        nu = self.mnab[1]
        ev = munu / nu * att
        return ev

    def var(self, att):
        # sig2 = self._sigma2()
        # nu = self.mnab[1]
        nc = self.skew
        df = self._df(att)
        if df <= 2:
            return np.inf
        return st.nct.var(df, nc, scale=self._scale(att))

    # def std_res(self, yds, att):
    def cdf(self, yds, att):
        # can also just look at the CDF and check that it's flat from 0 to 1
        # but the CDF is not easy to compute analytically
        df,nc = att,self.skew
        if df == 0:
            # there can be nonzero rushing yards w/out an attempt due to laterals. just skip these.
            return 0.
        cdf = st.nct.cdf(yds, df, nc,
                         loc=self._loc(att),
                         scale=self._scale(att))
        assert(not np.isnan(cdf))
        return cdf
    
    def chi_sq(self, yds, att):
        df = self._df(att)
        # using nct.mean results in undefined when df = 1
        nc = self.skew
        ncmean = self._ncmean(att)
        scale = self._scale(att)
        norm = st.nct.logpdf( ncmean, df, self.skew, loc=0., scale=scale )
        
        return 2.*(self.kld(yds, att) + norm)
    
    def kld(self, yds, att):
        df = self._df(att)
        if df == 0:
            # the pdf is undefined, but there is no information lost so just return 0
            # i.e. the data and model are both distributed as a delta function at 0.
            return 0.
        nc = self.skew
        # print(ncmean, st.nct.mean(df, nc)) # these are the same
        # the problem w/ using the mean for the offset is that this blows up for df = 1
        # the skew parameter is in between the mode and mean, so let's just use this
        loc = self._loc(att)
        scale = self._scale(att)
        result = - st.nct.logpdf(yds, df, nc, loc=loc,
                                 scale=scale)
        return result

    def __str__(self):
        parstr = u'\u03BC\t= {:.2f}\n\u03BD\t= {:.2f}\n\u03B1\t= {:.2f}\n\u03B2\t= {:.2f}\n'.format(self.mnab[0]/self.mnab[1], *self.mnab[1:])
        hparstr = 'skew\t= {:.4}\n'.format(self.skew)
        hparstr += 'lr\t= {}\n'.format(self.game_lr) # just print out whole array
        hparstr += 'mem\t= {}\ngmem\t= {}\n'.format(self.season_mem, self.game_mem)
        return parstr + hparstr


# can't figure out how to keep the parameters from going to invalid values, unless we turn off all memory.
# the data may often just be under-dispersed.
# we might be better off trying to inject extra variance into the bayes parameters when needed
class CountsModelMM(Model):
    """
    uses the method of moments to model a number of discrete events (like attempts per game).
    the predictive posterior is a negative binomial.
    the 0-2nd moments will be saved with some memory factors, and the distribution parameters set to those.
    this allows the variance to be learned. at least for now, the mean and variance learn rates are the same.
    """
    def __init__(self,
                 m0, m1, m2, # initial moments (un-normalized: mn = sum(k**n))
                 lr,     # learn rate per game
                 mem, gmem, # memory per season and game
    ):
        # we have a constraint that m2 > m1**2/m0
        self.mom = np.array((m0,m1,m2))
        self.game_lr = lr
        self.season_mem =  mem
        self.game_mem = gmem
    
    @classmethod
    def _hyperpar_bounds(self):
        return [(1e-6,None),(1e-6,None),(1e-6,None),(0,1),
                (0.1,1.0),(0.5,1.0)
        ]
    
    def _rp(self):
        # p = beta/(1+beta), r = alpha
        m0,m1,m2 = self.mom[0], self.mom[1], self.mom[2]
        r = m1**2/(m0*m2 - m0*m1 - m1**2)
        p = m0*m1/(m2*m0 - m1**2)
        if (r <= 0 or p <= 0):
            print('invalid parameter space!')
            print((r,p))
            print(self.mom)
            exit(1)
        return (r,p)
        
    def update_game(self, att):
        self.mom *= self.game_mem
        self.mom += self.game_lr * np.array((1., att, att**2))
        # we could accumulate a KLD to diagnose when the model has been very off recently

    def new_season(self):
        self.mom *= self.season_mem
        
    def ppf(self, uni):
        # this yields a gamma convoluted w/ a poisson
        att = st.nbinom.ppf(uni, *self._rp())
        return att

    def cdf(self, att):
        cdf = st.nbinom.cdf(att, *self._rp())
        return cdf
    
    def ev(self):
        r,p = self._rp()
        return r*(1-p)/p

    def var(self):
        r,p = self._rp()
        return r*(1-p)/p**2

    # remember that we don't want to minimize the chi-sq, we want to minimize the KLD
    # this will attempt to normalize, but will sometimes still end up negative
    def chi_sq(self, att):
        norm = dist_fit.log_neg_binomial(self.ev(), *self._rp())
        return 2.*(self.kld(att) + norm)
    
    def kld(self, att):
        return - st.nbinom.logpmf(att, *self._rp())

    def __str__(self):
        pars = u'r\t= {:.2f}\np\t= {:.2f}\n'.format(*self._rp())
        pars += 'm0\t= {:.2f}\nm1\t= {:.2f}\nm2\t= {:.2f}\n'.format(*self.mom)
        pars += 'lr\t= {:.3f}\nmem\t= {:.3f}\ngmem\t= {:.3f}\n'.format(self.game_lr, self.season_mem, self.game_mem)
        return pars
