# will define model templates and provide some common functionality
import numpy as np
import scipy.stats as st
from scipy.special import gamma, digamma
import dist_fit

# TODO: implement scipy.stats.rv_discrete ?
class CountsModel:
    """
    used to model a number of discrete events (like attempts per game).
    the predictive posterior is a negative binomial.
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
    
    @property
    def var_names(self):
        var_names = [self.pred_var] + list(self.dep_vars)
        return var_names
    
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


class TrialModel:
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
            (0.0,20.0), # uncap the learn rate
            (0.2,1.0),(0.5,1.0)
        ]

    @property
    def var_names(self):
        var_names = [self.pred_var] + list(self.dep_vars)
        return var_names

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
        norm = dist_fit.log_beta_binomial( self.ev(att), att, *self.ab)
        return 2.*(self.kld(succ, att) + norm)
    
    def kld(self, succ, att):
        # if att == 0: return 0.
        return - dist_fit.log_beta_binomial( succ, att, *self.ab)

    def __str__(self):
        pars = u'{:.2f}% rate\n\u03B1\t={:.2f}\n\u03B2\t= {:.2f}\n'.format(100*self._p(), *self.ab)
        hpars = 'lr\t= {:.3f}\n'.format(self.game_lr)
        hpars += 'smem\t= {:.3f}\ngmem\t= {:.3f}\n'.format(self.season_mem, self.game_mem)
        return pars + hpars

class YdsPerAttModel:
    """
    model for yards per attempt (though it could be more general than that).
    it uses a non-central student-t distribution to allow positive skew.
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
    def for_position(self, pos):
        return self(*self._default_hyperpars(pos))
        
    @classmethod
    def _hyperpar_bounds(self):
        return [
            (1e-6,None),(1e-6,None),(1e-6,None),(1e-6,None),
            (0.0,8.0), # skew
            (0.0,1.0),(0.0,1.0), # learn rates. lr for mean is actually much > 1 for QBs;
            (0.2,1.0),(0.4,1.0), # season memory
            (0.5,1.0),(0.5,1.0), # game memory - doesn't help much
        ]
        
    @property
    def var_names(self):
        var_names = [self.pred_var] + list(self.dep_vars)
        return var_names

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

    def _loc(self, att):
        df = self._df(att)
        nc = self.skew
        ncmean = st.nct.mean(df, nc) if df > 1 else nc
        return (self.mnab[0] / self.mnab[1] - ncmean) * att
    
    def _scale(self, att):
        df = self._df(att)
        scale = att*np.sqrt(self._sigma2())
        if df > 2:
            nc = self.skew
            scale /= ( df*(1+nc**2)/(df-2) - nc**2*df/2*(gamma((df-1)/2)/gamma(df/2))**2 )
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
        # ncmean = nc * np.sqrt(df/2)*gamma((df-1)/2)/gamma(df/2)
        # ncmean=0
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
        ncmean = st.nct.mean(df, nc) if df > 1 else nc
        norm = st.nct.logpdf( ncmean, df, self.skew, loc=0., scale=self._scale(att) )
        
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
