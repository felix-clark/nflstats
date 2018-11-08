import numpy as np
import scipy.stats as st
import logging

from playermodels.passing import *
from playermodels.rushing import *
from playermodels.receiving import *

def get_stat_model(mname):
    """
    a convenience function to return a model type from its name.
    """
    models = {
        'rush_att': RushAttModel,
        'rush_yds': RushYdsModel,
        'rush_td': RushTdModel,
        'targets': RecTgtModel,
        'rec': RecModel,
        'rec_yds': RecYdsModel,
        'rec_td': RecTdModel,
        'pass_att': PassAttModel,
        'pass_cmp': PassCmpModel,
        'pass_yds': PassYdsModel,
        'pass_td': PassTdModel,
        'pass_int': PassIntModel,
    }
    if mname not in models:
        logging.error('could not provide model with name {}'.format(mname))
    else: return models[mname]
    
def gen_player_model(pos):
    pm = {
        'QB': QbModel,
        'RB': RbModel,
        'WR': WrModel,
        'TE': TeModel,
        }
    if pos.upper() not in pm:
        logging.error('no model for position {}'.format(pos))
    return pm[pos.upper()]()


class PosModel:
    """
    the position models are defined by the models for their stats and the covariance in each.
    most of the funcionatlity can be defined here.
    """
    @property
    def stats(self):
        return [model.name for model in self.models]

    def gen_game(self):
        """
        generate a random statline for a single game
        """
        game = {}
        # call the ppf of each member model
        # the generator should handle the correlations
        urvs = st.norm.cdf(self.stat_gen.rvs())
        for rv,model in zip(urvs, self.models):
            depvars = [game[dv] for dv in model.dep_vars] # get previously generated stats needed for this one
            game[model.pred_var] = model.ppf(*depvars, rv)
        return game

    def update_game(self, game):
        """
        update the stats based on the results of 1 game
        we can either do this stochastically using the KLD gradient and a variable learn rate, or using bayesian models
        """
        for model in self.models:
            stats = [game[v] for v in model.var_names]
            model.update_game(*stats)

    def new_season(self):
        """
        decay some parameters to account for uncertainty between seasons
        """
        for model in self.models:
            model.new_season()

    def revert_evs(self, ev_dict):
        """
        takes a dictionary of values to revert the EV of each model to
        """
        for model in self.models:
            vname = model.pred_var
            if vname in ev_dict:
                newev = ev_dict.pop(vname)
                model.revert_ev(newev)
        if len(ev_dict) > 0:
            logging.error('Could not revert all evs:')
            logging.error(ev_dict)

    def evs(self):
        evs = {}
        for model in self.models:
            depvars = [evs[dv] for dv in model.dep_vars]
            evs[model.pred_var] = model.ev(*depvars)
        return evs

    def __str__(self):
        # outstr = '\n'.join(('\n{}:\n{}\n'.format(mod.pred_var, mod)  for mod in self.models))
        outstr = '\n'.join((mod.summary()  for mod in self.models))
        return outstr

class QbModel(PosModel):
    """
    model for quarterbacks.
    passing and rushing.
    """
    def __init__(self):
        # these must be ordered such that stats come after those they depend on
        self.models = (
            PassAttModel.for_position('QB'),
            PassCmpModel.for_position('QB'),
            PassYdsModel.for_position('QB'),
            PassTdModel.for_position('QB'),
            PassIntModel.for_position('QB'),
            RushAttModel.for_position('QB'),
            RushYdsModel.for_position('QB'),
            RushTdModel.for_position('QB'),
        )
        # it might make sense to fix some of the off-diagonal terms to zero, but maybe moreso for other positions than QBs.
        self.stat_gen = st.multivariate_normal(
            cov = [[ 1.   ,-0.102, 0.164,-0.227,-0.043, 0.09 , 0.148,-0.022],
                    [-0.102, 1.   , 0.149, 0.099,-0.21 , 0.052,-0.124, 0.009],
                    [ 0.164, 0.149, 1.   , 0.228,-0.122, 0.001,-0.037, 0.079],
                    [-0.227, 0.099, 0.228, 1.   ,-0.072, 0.019,-0.025,-0.092],
                    [-0.043,-0.21 ,-0.122,-0.072, 1.   ,-0.075, 0.054, 0.024],
                    [ 0.09 , 0.052, 0.001, 0.019,-0.075, 1.   ,-0.052,-0.322],
                    [ 0.148,-0.124,-0.037,-0.025, 0.054,-0.052, 1.   , 0.093],
                    [-0.022, 0.009, 0.079,-0.092, 0.024,-0.322, 0.093, 1.   ]]
            )

# we'll want to split this up into a few types
class RbModel(PosModel):
    """
    represents a statistical model for a single season for a running back.
    """
    def __init__(self):
        # these must be ordered such that stats come after those they depend on
        self.models = (
            RushAttModel.for_position('RB'),
            RushYdsModel.for_position('RB'),
            RushTdModel.for_position('RB'),
            RecTgtModel.for_position('RB'),
            RecModel.for_position('RB'),
            RecYdsModel.for_position('RB'),
            RecTdModel.for_position('RB'),
        )
        # the order here is important: att, yds, tds
        # the covariance matrix is the spearman (rank) correlations between
        # the model's cdf values for each training data point.
        # the correlation is computed weighting for rushing attempts, which makes the correlations smaller.
        # in theory, this correlation matrix itself could be updated each season for each player individually.
        self.stat_gen = st.multivariate_normal(
            cov = [[ 1.   , 0.177,-0.09 , 0.099, 0.017, 0.026, 0.044],
                   [ 0.177, 1.   , 0.146,-0.048, 0.002, 0.032, 0.053],
                   [-0.09 , 0.146, 1.   ,-0.042, 0.034, 0.037,-0.002],
                   [ 0.099,-0.048,-0.042, 1.   ,-0.145, 0.07 ,-0.337],
                   [ 0.017, 0.002, 0.034,-0.145, 1.   , 0.075,-0.138],
                   [ 0.026, 0.032, 0.037, 0.07 , 0.075, 1.   , 0.055],
                   [ 0.044, 0.053,-0.002,-0.337,-0.138, 0.055, 1.   ]]
        )

class WrModel(PosModel):
    """
    represents the stats to track for wideouts
    """
    
    def __init__(self):
        self.models = (
            RecTgtModel.for_position('WR'),
            RecModel.for_position('WR'),
            RecYdsModel.for_position('WR'),
            RecTdModel.for_position('WR'),
            RushAttModel.for_position('WR'),
            RushYdsModel.for_position('WR'),
            RushTdModel.for_position('WR'),
        )
        self.stat_gen = st.multivariate_normal(
            cov = [[ 1.   ,-0.081, 0.007,-0.199, 0.025,-0.009,-0.007],
                    [-0.081, 1.   , 0.059,-0.148, 0.009, 0.033,-0.05 ],
                    [ 0.007, 0.059, 1.   , 0.205,-0.008,-0.037, 0.078],
                    [-0.199,-0.148, 0.205, 1.   , 0.019, 0.001, 0.007],
                    [ 0.025, 0.009,-0.008, 0.019, 1.   , 0.113,-0.081],
                    [-0.009, 0.033,-0.037, 0.001, 0.113, 1.   , 0.053],
                    [-0.007,-0.05 , 0.078, 0.007,-0.081, 0.053, 1.   ]]
        )

class TeModel(PosModel):
    """
    model for tight ends.
    only tracks receptions.
    """
    def __init__(self):
        self.models = (
            RecTgtModel.for_position('TE'),
            RecModel.for_position('TE'),
            RecYdsModel.for_position('TE'),
            RecTdModel.for_position('TE'),
        )
        self.stat_gen = st.multivariate_normal(
            cov = [[ 1.   ,-0.111, 0.011,-0.184],
                   [-0.111, 1.   , 0.11 ,-0.133],
                   [ 0.011, 0.11 , 1.   , 0.027],
                   [-0.184,-0.133, 0.027, 1.   ]]
        )
