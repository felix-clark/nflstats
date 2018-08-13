import dist_fit
import numpy as np
import scipy.stats as st
from scipy.special import gamma, digamma

# we'll want to split this up into a few types
class RbModel:
    """
    represents a statistical model for a single season for a running back.
    """
    def __init__(self):
        self.rushatt = RushAttModel()

    def gen_game(self):
        game = {}
        rush_att = self.rushatt.gen_game()
        game['rushing_att'] = rush_att
        return game

    def bayes_update_game(self, game):
        """
        update the stats based on the results of 1 game
        we can either do this stochastically using the KLD gradient and a variable learn rate, or using bayesian models
        """
        self.rushatt.bayes_update_game(game['rushing_att'])

    def bayes_new_season(self):
        """
        decay some parameters to account for uncertainty between seasons
        """
        pass

# this is really just like our neg binomial model,
# but with a different interface and week-to-week sensitivity
# TODO: implement scipy.stats.rv_discrete ?
class RushAttModel:
    """
    statistical model for rushing attempts by RBs
    this could be extended to other positions, with different defaults
    """
    # TODO: make these defaults a function of whether this is expected to be an RB1 or RB2/3
    def __init__(self, lr=0.135, mem=0.66, gmem=0.8, ab0=(6.392,0.4694)):
        # self.__pred_var=('rushing_att',)
        
        # these defaults are gotten from top 16 (by ADP) RBs.
        # they won't be good for general RBs down the line.
        # r and p for negative binomial distribution (scipy convention, not wikipedia)
        # self.rusha_r = 10.
        # self.rusha_p = 0.375

        # alpha and beta, using gamma prior for neg.binom. predictive
        # beta = p/(1-p)
        # EV = alpha/beta
        # self.alpha,self.beta = ab0
        self.ab = np.array(ab0)

        # we might want game_lr to be a function of the season?
        self.game_lr = lr
        self.game_mem = gmem
        self.season_memory = mem

    @property
    def var_names(self):
        var_names = [self.pred_var] + list(self.dep_vars)
        return var_names

    # the variable we're predicting
    @property
    def pred_var(self):
        # return ('rushing_att',)
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
        assert(0 < self.game_mem <= 1.0)
        self.ab *= self.game_mem
        self.ab += self.game_lr * np.array((rush_att, 1.))
        # we could accumulate a KLD to diagnose when the model has been very off recently

    def new_season(self):
        assert(0 < self.season_memory <= 1.0)
        self.ab *= self.season_memory
        
    def gen_game(self):
        # this yields a gamma convoluted w/ a poisson
        return st.nbinom.rvs(self.ab[0], self._p())

    def ev(self):
        return self.ab[0] / self.ab[1]

    def var(self):
        # 1 - p = 1/(1+beta)
        ## var/alpha = (1-p)/p**2  = (1+beta)/beta**2
        return self.ab[0]*(1.+self.ab[1])/self.ab[1]**2

    def scale(self):
        return np.sqrt(self.var())

    # we may be able to make this more general, and not have to implement it for every model
    # remember that we don't want to minimize the chi-sq, we want to minimize the KLD
    def chi_sq(self, rush_att):
        norm = dist_fit.log_neg_binomial(self.ev(), self.ab[0], self._p())
        return 2.*(self.kld(rush_att) + norm)
    
    def kld(self, rush_att):
        result = - st.nbinom.logpmf(rush_att, self.ab[0], self._p())
        return result

    def __str__(self):
        # assert (abs(st.nbinom.mean(self.alpha, p) - mu) < 0.001)
        std = st.nbinom.std(self.ab[0], self._p())
        return u'rush_att: \u03B1={:.2f}, \u03B2={:.2f}; {:.1f} pm {:.1f}'.format(self.ab[0], self.ab[1],
                                                                                  self.ev(), std)

# a stochastic model doesn't work right out of the box because the gradients can easily make alpha,beta < 0
class RushAttStochModel(RushAttModel):
    """
    stochiastic model that updates via gradient descent instead of bayes
    """
    def __init__(self, *args):
        # RushAttModel.__init__(self, *args)
        super(RushAttStochModel, self).__init__(*args)
    
    def update_game(self, rush_att):
        self.ab *= self.game_mem
        self.ab += - self.game_lr * self._grad_kld(rush_att)
        if (self.ab <= 0).any():
            logging.error('alpha and beta must be > 0')
        
    def _grad_kld(self, rush_att):
        # could use dist_fit.grad_sum_log_neg_binomial,
        # but we don't need the sum and it uses a different change of variables.
        dlda = digamma(rush_att + self.ab[0]) - digamma(self.ab[0]) + np.log(self.ab[1]/(1.+self.ab[1]))
        dldb = (self.ab[0]/self.ab[1] - rush_att)/(1.+self.ab[1])
        return np.array((dlda, dldb))

class RushYdsModel:
    """
    statistical model for yards per rush
    this could again be extended to other positions, with different defaults
    this model will assume a symmetry that isn't there at low attempts.
      (e.g. some might have a 40 yd/att on 1-2 rushes but -40 yd/att is impossible)
    """
    def __init__(self, lr=(0.04,0.002), mem=0.83, gmem=0.99, skew=1.1, mnab0=(49.8,12.0,2.0,5.35)):
        # this represents (mu*nu, nu, alpha, beta). note that we save only mu*nu, for simpler decay.
        self.mnab = np.array(mnab0)
        # the skewness will be a constant hyperparameter for now (not clear how to do bayesian updating w/ non-centrality)
        # to get a good value for skewness directly, we'd need play-by-play data
        self.skew = skew

        # we might want game_lr to be a function of the season?
        # for this we might have different learn rates for mu/nu and alpha/beta.
        self.game_lr = np.array(lr)
        self.game_mem = gmem
        self.season_memory = mem

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
        assert(0 < self.game_mem <= 1.0)
        # mu does not decay simply like the others, but mu*nu does
        ev = self.ev(rush_att)
        self.mnab *= self.game_mem
        self.mnab[:2] += self.game_lr[0] * np.array((rush_yds, rush_att))
        self.mnab[2:] += self.game_lr[1] * 0.5 * np.array((rush_att, (rush_yds-ev)**2/rush_att))
        # self.mnab += self.game_lr * np.array((rush_yds, rush_att, 0.5*rush_att,
        #                                       # this update rule for beta may not be quite right,
        #                                       # but should approach something reasonable at large n:
        #                                       0.5*(rush_yds-self.ev(rush_att))**2/rush_att)
        #                                      )
        # we could accumulate a KLD to diagnose when the model has been very wrong recently

    def new_season(self):
        assert(0 < self.season_memory <= 1.0)
        self.mnab *= self.season_memory

    def _df(self, rush_att):
        # we should either use the # of attempts, or twice alpha.
        # if using rush_att, alpha is not used at all.
        # return 2.0*self.mnab[2]
        return rush_att
        
    # helper function for common parameter
    def _sigma2(self):
        nu,alpha,beta = tuple(self.mnab[1:])
        return beta*(nu+1.)/(nu*alpha)
        
    def gen_game(self, rush_att):
        df = self._df(rush_att) # if we do this approach, we don't use our alpha at all?
        loc = (self.mnab[0] / self.mnab[1] - st.nct.mean(df, self.skew)) * rush_att # can't use ev since that includes skew
        return st.nct.rvs(df, self.skew, loc=loc, scale=rush_att*np.sqrt(self._sigma2()))

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

    def scale(self, rush_att):
        return rush_att*np.sqrt(self._sigma2())

    def chi_sq(self, rush_yds, rush_att):
        df = self._df(rush_att)
        nu = self.mnab[1]
        # ev = self.ev(1) + self.skew*np.sqrt(nu/2.)*gamma(0.5*(nu-1))/gamma(0.5*nu)
        # ev = self.ev(rush_att) # should be converging to the mean
        norm = st.nct.logpdf( st.nct.mean(df, self.skew), df, self.skew, loc=0., scale=self.scale(rush_att) )
        # kld_calc = st.nct.logpdf(rush_yds, df, self.skew,
        #                          loc=(self.mnab[0] / self.mnab[1] - st.nct.mean(df, self.skew)) * rush_att,
        #                          scale=self.scale(rush_att))
        # print(norm, kld_calc)
        return 2.*(self.kld(rush_yds, rush_att) + norm)
    
    def kld(self, rush_yds, rush_att):
        df = self._df(rush_att)
        nc = self.skew
        # ncmean = nc * np.sqrt(df/2)*gamma((df-1)/2)/gamma(df/2)
        ncmean = st.nct.mean(df, nc) # get the unscaled contribution to the mean from the skew
        # print(ncmean, st.nct.mean(df, nc)) # these are the same
        loc = (self.mnab[0] / self.mnab[1] - ncmean) * rush_att # can't use ev since that includes skew
        scale = self.scale(rush_att) # /( df*(1+nc**2)/(df-2) - nc**2*df/2*(gamma((df-1)/2)/gamma(df/2))**2 )
        result = - st.nct.logpdf(rush_yds, df, nc, loc=loc,
                                 scale=scale)
        return result

    def __str__(self):
        return u'rush_yds: \u03BC={:.2f}, \u03BD={:.2f}, \u03B1={:.2f}, \u03B2={:.2f}'.format(self.mnab[0]/self.mnab[1], *self.mnab[1:])


class RushTdModel:
    """
    statistical model for TDs per rush
    this could again be extended to other positions, with different defaults
    """
    def __init__(self, lr=1.0, mem=0.77, gmem=1.0, ab0=(42.0,1400.)):
        # self.__var_names = ('rushing_tds', 'rushing_att')
        # self.__dep_vars = ('rushing_att',)
    
        self.ab = np.array(ab0)

        # we might want game_lr to be a function of the season?
        self.game_lr = lr
        self.game_mem = gmem
        self.season_memory = mem

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
        assert(0 < self.game_mem <= 1.0)
        self.ab *= self.game_mem
        self.ab += self.game_lr * np.array((rush_td, rush_att - rush_td))
        # print(self.ab)
        # exit(1)
        # we could accumulate a KLD to diagnose when the model has been very off recently

    def new_season(self):
        assert(0 < self.season_memory <= 1.0)
        self.ab *= self.season_memory
        
    def gen_game(self, rush_att):
        p = st.beta.rvs(self.ab[0], self.ab[1])
        return st.binom.rvs(rush_att, p)

    def ev(self, rush_att):
        ev = self.ab[0] / self.ab[1] * rush_att
        return ev

    def var(self, rush_att):
        a,b = self.ab[0],self.ab[1]
        apb = a + b
        var = rush_att*a*b*(apb+rush_att)/(apb**2*(apb+1))
        return var

    def scale(self, rush_att):
        return np.sqrt(var(rush_att))

    def chi_sq(self, rush_td, rush_att):
        norm = dist_fit.log_beta_binomial( self.ev(rush_att), rush_att, self.ab[0], self.ab[1])
        return 2.*(self.kld(rush_td, rush_att) + norm)
    
    def kld(self, rush_td, rush_att):
        alpha,beta = self.ab[0],self.ab[1]
        result = - dist_fit.log_beta_binomial( rush_td, rush_att, alpha, beta)
        return result

    def __str__(self):
        mu = self.alpha / (self.alpha + self.beta)
        # p = self.beta/(1.+self.beta)
        # we can't get the variance for this distribtion w/out the rush attempts
        # we could compute it by compounding variance, but that's not too necessary rn
        # std = st.nbinom.std(self.alpha, p)
        return u'rush_td: \u03B1={:.2f}, \u03B2={:.2f}; {:.2f}%'.format(self.alpha, self.beta, mu*100)
