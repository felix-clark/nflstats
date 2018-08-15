import dist_fit
import numpy as np
import scipy.stats as st
from scipy.special import gamma, digamma
import logging

# a model rush attempts per game.
# TODO: implement scipy.stats.rv_discrete ?
class RushAttModel:
    """
    statistical model for rushing attempts by RBs, QBs, and WRs.
    """
    def __init__(self, a0, b0, lr, mem, gmem,
                 learn=True
    ):
        self.ab = np.array((a0,b0))
        # we might want game_lr to be a function of the season?
        self.game_lr = lr if learn else 0.0
        self.game_mem = gmem if learn else 1.0
        self.season_mem = mem if learn else 1.0

    @classmethod
    def for_position(self, pos):
        """
        provides an instance of the model with default hyperparameters
        """
        return RushAttModel(*self._default_hyperpars(pos))

    @classmethod
    def _default_hyperpars(self, pos):
        pos = pos.upper()
        if pos == 'RB':
            return np.array((
                2.807, 0.244,
                0.121, 0.677, 0.782))
        if pos == 'QB':
            return np.array((
                2.61, 0.840,
                0.241, 0.703, 0.954))
        if pos == 'WR':
            return np.array((
                0.516, 3.65,
                0.646, 0.523, 0.972
                ))
        if pos == 'TE':
            logging.error('TEs do not rush enough to try to predict them')
        logging.error( 'positional defaults not implemented for {}'.format(pos) )
        pass

    @property
    def var_names(self):
        var_names = [self.pred_var] + list(self.dep_vars)
        return var_names

    # the variable we're predicting
    @property
    def pred_var(self):
        return 'rushing_att'
    
    @property
    def dep_vars(self):
        return ()
        
    # a shortcut for a re-mapping of beta that comes up a lot due to the common convention for negative binomial
    def _p(self):
        # scipy convention for p (not wiki)
        beta = self.ab[1]
        return beta/(1.+beta)

    def update_game(self, rush_att):
        self.ab *= self.game_mem
        self.ab += self.game_lr * np.array((rush_att, 1.))
        # we could accumulate a KLD to diagnose when the model has been very off recently

    def new_season(self):
        self.ab *= self.season_mem
        
    def ppf(self, uni):
        # assert(0 < uni < 1)
        # this yields a gamma convoluted w/ a poisson
        rush_att = st.nbinom.ppf(uni, self.ab[0], self._p())
        return rush_att

    def ev(self):
        return self.ab[0] / self.ab[1]

    def var(self):
        return self.ev()/self._p()

    def scale(self):
        return np.sqrt(self.var())

    def cdf(self, rush_att):
        cdf = st.nbinom.cdf(rush_att, self.ab[0], self._p())
        return cdf
    
    # we may be able to make this more general, and not have to implement it for every model
    # remember that we don't want to minimize the chi-sq, we want to minimize the KLD
    def chi_sq(self, rush_att):
        norm = dist_fit.log_neg_binomial(self.ev(), self.ab[0], self._p())
        return 2.*(self.kld(rush_att) + norm)
    
    def kld(self, rush_att):
        result = - st.nbinom.logpmf(rush_att, self.ab[0], self._p())
        return result

    def __str__(self):
        std = st.nbinom.std(self.ab[0], self._p())
        pars = u'rush_att: \u03B1={:.2f}, \u03B2={:.2f}; {:.1f} pm {:.1f}\n'.format(self.ab[0], self.ab[1],
                                                                                    self.ev(), std)
        pars += 'learn rate, mem (seas/gm): {}, {}, {}\n'.format(self.game_lr, self.season_mem, self.game_mem)
        return pars

class RushYdsModel:
    """
    statistical model for yards per rush
    it uses a non-central student-t distribution to allow positive skew
    """
    def __init__(self,
                 mn0, n0, a0, b0,
                 lr1, lr2,
                 skew,
                 mnmem, # memory decay per season for munu / nu
                 abmem, # seasonal parameter decay for a/b
                 mngmem, # game memory for munu/nu. doesn't seem to help much.
                 abgmem, # game memory for a/b
                 learn=True # can turn this false to shortcut other settings
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
        self.game_lr = np.repeat((lr1,lr2), 2) if learn else 0.
        self.season_mem = np.repeat((mnmem, abmem), 2) if learn else 1.0
        self.game_mem = np.repeat((mngmem, abgmem), 2)
        
    @classmethod
    def for_position(self, pos):
        return RushYdsModel(*self._default_hyperpars(pos))
    
    @classmethod
    def _default_hyperpars(self, pos):
        pos = pos.upper()
        if pos == 'RB':
            return np.array((
                122.3, 36.26, 8.89, 40.08, # initial bayes parameters
                0.00259, 0.0191, # learn rates
                0.793, # skew
                1.0, # munu/nu memory
                0.979,# alpha/beta mem
                1.0,0.966, # game memories
            ))
        if pos == 'QB':
            return np.array((
                112.9, 48.51, 2.85, 50.35, # initial bayes parameters
                6.266, 0.0272, # learn rates
                0.0552, # skew# possibly low due to sacks?
                0.667, # munu/nu memory
                0.769, # alpha/beta mem
                1.0,1.0 # game memories don't help for QBs
            ))
        if pos == 'WR':
            # this has a reasonable flat CDF
            # interesting that there is no memory
            return np.array((
                116.3, 41.28, 3.46, 45.64, # initial bayes parameters
                0.563, 0.0, # learn rates
                0.457, # skew
                1.0, # munu/nu memory
                1.0, # alpha/beta mem
                1.0,1.0 # game memories don't work well for WRs
            ))
        if pos.upper() == 'TE': # TEs rush so rarely we shouldn't even include them
            logging.error('TEs do not run enough to try to predict their rushes')
        logging.error('no default hyperparameters are implemented for {}'.format(pos))
        pass
        
    @property
    def var_names(self):
        var_names = [self.pred_var] + list(self.dep_vars)
        return var_names

    @property
    def pred_var(self):
        return 'rushing_yds'
    
    # dependent variables i.e. those required for prediction
    @property
    def dep_vars(self):
        return ('rushing_att',)# self.__dep_vars
    
    def update_game(self, rush_yds, rush_att):
        # assert((0 < self.game_mem).all() and (self.game_mem <= 1.0).all())
        # mu does not decay simply like the others, but mu*nu does
        ev = self.ev(rush_att)
        self.mnab *= self.game_mem
        self.mnab += self.game_lr * np.array((rush_yds, rush_att,
                                              0.5*rush_att,
                                              0.5*(rush_yds-ev)**2/max(1,rush_att)))
                                              # 0.5,
                                              # 0.5*(rush_yds-ev)**2/rush_att**2))
        # we could accumulate a KLD to diagnose when the model has been very wrong recently

    def new_season(self):
        # if not((0 < self.season_mem).all() and (self.season_mem <= 1.0).all()):
        #     logging.warning('season mem = {}'.format(self.season_mem))
        self.mnab *= self.season_mem

    def _df(self, rush_att):
        # we should either use the # of attempts, or twice alpha.
        # if using rush_att, alpha is not used at all.
        # return 2.0*self.mnab[2]
        return rush_att
        
    # helper function for common parameter
    def _sigma2(self):
        nu,alpha,beta = tuple(self.mnab[1:])
        return beta*(nu+1.)/(nu*alpha)

    # given a uniformly distributed number 0 < uni < 1, return the yards corresponding to that point on the cdf.
    # correlations can be dealt with externally, since it's much easier to correlate normal variables.
    def ppf(self, rush_att, uni):
        assert(0 < uni < 1)
        if rush_att == 0: return 0 # we won't try to simulate laterals
        df,nc = rush_att,self.skew
        rush_yds = st.nct.ppf(uni, df, nc,
                              loc=self.loc(rush_att),
                              scale=self.scale(rush_att),)
        return rush_yds
    
    def ev(self, rush_att):
        # the mean is mu*nu / nu
        munu = self.mnab[0]
        nu = self.mnab[1]
        # df,nc = rush_att,self.skew
        # ncmean = nc * np.sqrt(df/2)*gamma((df-1)/2)/gamma(df/2)
        # ncmean=0
        ev = munu / nu * rush_att
        # print (ev, st.nct.mean(df, nc, loc=(munu/nu-ncmean)*rush_att, scale=rush_att*np.sqrt(self._sigma2())))
        return ev

    def var(self, rush_att):
        # sig2 = self._sigma2()
        # nu = self.mnab[1]
        nc = self.skew
        df = self._df(rush_att)
        if df <= 2:
            return np.inf
        return st.nct.var(df, nc, scale=self.scale(rush_att))

    def loc(self, rush_att):
        df = self._df(rush_att)
        nc = self.skew
        ncmean = st.nct.mean(df, nc) if df > 1 else nc
        return (self.mnab[0] / self.mnab[1] - ncmean) * rush_att
    
    def scale(self, rush_att):
        df = self._df(rush_att)
        scale = rush_att*np.sqrt(self._sigma2())
        if df > 2:
            nc = self.skew
            scale /= ( df*(1+nc**2)/(df-2) - nc**2*df/2*(gamma((df-1)/2)/gamma(df/2))**2 )
        return scale

    # def std_res(self, rush_yds, rush_att):
    def cdf(self, rush_yds, rush_att):
        # can also just look at the CDF and check that it's flat from 0 to 1
        # but the CDF is not easy to compute analytically
        df,nc = rush_att,self.skew
        if df == 0:
            # there can be nonzero rushing yards w/out an attempt due to laterals. just skip these.
            return 0.
        cdf = st.nct.cdf(rush_yds, df, nc,
                         loc=self.loc(rush_att),
                         scale=self.scale(rush_att))
        assert(not np.isnan(cdf))
        return cdf
    
    def chi_sq(self, rush_yds, rush_att):
        df = self._df(rush_att)
        # using nct.mean results in undefined when df = 1
        nc = self.skew
        ncmean = st.nct.mean(df, nc) if df > 1 else nc
        norm = st.nct.logpdf( ncmean, df, self.skew, loc=0., scale=self.scale(rush_att) )
        
        return 2.*(self.kld(rush_yds, rush_att) + norm)
    
    def kld(self, rush_yds, rush_att):
        df = self._df(rush_att)
        if df == 0:
            # the pdf is undefined, but there is no information lost so just return 0
            # i.e. the data and model are both distributed as a delta function at 0.
            return 0.
        nc = self.skew
        # print(ncmean, st.nct.mean(df, nc)) # these are the same
        # the problem w/ using the mean for the offset is that this blows up for df = 1
        # the skew parameter is in between the mode and mean, so let's just use this
        loc = self.loc(rush_att)
        scale = self.scale(rush_att)
        result = - st.nct.logpdf(rush_yds, df, nc, loc=loc,
                                 scale=scale)
        return result

    def __str__(self):
        parstr = u'rush_yds: \u03BC={:.2f}, \u03BD={:.2f}, \u03B1={:.2f}, \u03B2={:.2f}\n'.format(self.mnab[0]/self.mnab[1], *self.mnab[1:])
        hparstr = 'skew: {}\n'.format(self.skew)
        hparstr += 'learn rate: {}\n'.format(self.game_lr)
        hparstr += 'game/season mem = {} / {}\n'.format(self.game_mem, self.season_mem)
        return parstr + hparstr


class RushTdModel:
    """
    statistical model for TDs per rush
    this could again be extended to other positions, with different defaults
    """
    def __init__(self, a0, b0,
                 lr,
                 mem,
                 gmem,
                 learn=True):
        self.ab = np.array((a0,b0))
        # we might want game_lr to be a function of the season?
        self.game_lr = lr if learn else 0. #lr
        self.game_mem = gmem if learn else 1.0 #gmem
        self.season_mem = mem if learn else 1.0 #mem

    @classmethod
    def for_position(self, pos):
        return RushTdModel(*self._default_hyperpars(pos))

    def _default_hyperpars(self, pos):
        pos = pos.upper()
        if pos == 'RB':
            return np.array((
                19.07, 684.7, # initial bayes parameters
                1.84, # learn rate
                0.775, # season memory
                1.0)) # game mem
        if pos == 'QB':
            return np.array((
                12.33, 330.27, # initial bayes parameters
                1.76, # learn rate
                1.0, # season memory
                0.980)) # game mem
        if pos == 'WR':
            # rushing TDs are rare enough for WRs that there's no reason to update a model from an event
            return np.array((
                1.97, 60.4, # initial bayes parameters
                0.0, # learn rate
                1.0, # season memory
                1.0)) # game mem
        if pos == 'TE':
            logging.error('TEs do not rush enough')
        logging.error( 'rushing TD positional defaults not implemented for {}'.format(pos) )
        
    @property
    def var_names(self):
        var_names = [self.pred_var] + list(self.dep_vars)
        return var_names

    @property
    def pred_var(self):
        return 'rushing_tds'
    
    # dependent variables i.e. those required for prediction
    @property
    def dep_vars(self):
        return ('rushing_att',)# self.__dep_vars
    
    def update_game(self, rush_td, rush_att):
        self.ab *= self.game_mem
        self.ab += self.game_lr * np.array((rush_td, rush_att - rush_td))

    def new_season(self):
        self.ab *= self.season_mem
        
    def ev(self, rush_att):
        ev = self.ab[0] / (self.ab[0] + self.ab[1]) * rush_att
        return ev

    def var(self, rush_att):
        a,b = self.ab[0],self.ab[1]
        apb = a + b
        var = rush_att*a*b*(apb+rush_att)/(apb**2*(apb+1))
        return var

    def ppf(self, rush_att, uni):
        assert(0 < uni < 1)
        rush_tds = 0
        cdf = dist_fit.beta_binomial(0, rush_att, *self.ab)
        while cdf < uni:
            rush_tds += 1
            cdf += dist_fit.beta_binomial(rush_tds, rush_att, *self.ab)
        return rush_tds
    
    def cdf(self, rush_tds, rush_att):
        # CDF is the % of the mass that is *at or equal to* the value.
        # these CDFs will not be flat, since most results are 0 and that's most of the way up the CDF already
        # this is the definition we want, however, for analyzing the correlations
        cdf = 0.
        checktd = 0
        while checktd <= rush_tds:
            cdf += dist_fit.beta_binomial(checktd, rush_att, *self.ab)
            checktd += 1
        # asgaussian = st.norm.ppf(cdf)
        # return asgaussian
        return cdf
        
    def scale(self, rush_att):
        return np.sqrt(var(rush_att))

    def chi_sq(self, rush_td, rush_att):
        norm = dist_fit.log_beta_binomial( self.ev(rush_att), rush_att, *self.ab)
        return 2.*(self.kld(rush_td, rush_att) + norm)
    
    def kld(self, rush_td, rush_att):
        result = - dist_fit.log_beta_binomial( rush_td, rush_att, *self.ab)
        return result

    def __str__(self):
        mu = self.ev(100) # % percent of rushes that end in TD
        pars = u'rush_td: \u03B1={:.2f}, \u03B2={:.2f}; {:.2f}%\n'.format(*self.ab, mu)
        hpars = 'learn rate = {}\n'.format(self.game_lr)
        hpars += 'memory (season/game): {:.3f} / {:.3f}\n'.format(self.season_mem, self.game_mem)
        return pars + hpars
