import dist_fit
import numpy as np
import scipy.stats as st
from scipy.special import digamma

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
        # print (dlda, dldb)
        return np.array((dlda, dldb))

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
        a,b = tuple(self.ab)
        apb = a + b
        var = rush_att*a*b*(apb+rush_att)/(apb**2*(apb+1))
        return var

    def chi_sq(self, rush_td, rush_att):
        norm = dist_fit.log_beta_binomial( self.ev(rush_att), rush_att, self.ab[0], self.ab[1])
        return 2.*(self.kld(rush_td, rush_att) + norm)
    
    def kld(self, rush_td, rush_att):
        # return -st.binom.logpmf(rush_td, rush_att, self.alpha, self.beta)
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
