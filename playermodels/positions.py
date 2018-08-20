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

    def evs(self):
        evs = {}
        for model in self.models:
            depvars = [evs[dv] for dv in model.dep_vars]
            evs[model.pred_var] = model.ev(*depvars)
        return evs

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
            cov = [[ 1.   ,-0.118,-0.148,-0.237,-0.028, 0.057, 0.159,-0.023],
                   [-0.118, 1.   ,-0.055, 0.105,-0.214, 0.05 ,-0.125,-0.   ],
                   [-0.148,-0.055, 1.   , 0.392,-0.036, 0.034,-0.066, 0.028],
                   [-0.237, 0.105, 0.392, 1.   ,-0.084, 0.017,-0.018,-0.107],
                   [-0.028,-0.214,-0.036,-0.084, 1.   ,-0.085, 0.061, 0.021],
                   [ 0.057, 0.05 , 0.034, 0.017,-0.085, 1.   ,-0.033,-0.278],
                   [ 0.159,-0.125,-0.066,-0.018, 0.061,-0.033, 1.   , 0.079],
                   [-0.023,-0.   , 0.028,-0.107, 0.021,-0.278, 0.079, 1.   ]]
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
            cov = [[ 1.   , 0.128,-0.093, 0.097, 0.009, 0.023, 0.044],
                   [ 0.128, 1.   , 0.144,-0.055, 0.018, 0.026, 0.052],
                   [-0.093, 0.144, 1.   ,-0.044, 0.039, 0.036, 0.001],
                   [ 0.097,-0.055,-0.044, 1.   ,-0.142, 0.073,-0.337],
                   [ 0.009, 0.018, 0.039,-0.142, 1.   , 0.068,-0.141],
                   [ 0.023, 0.026, 0.036, 0.073, 0.068, 1.   , 0.054],
                   [ 0.044, 0.052, 0.001,-0.337,-0.141, 0.054, 1.   ]]
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
            cov = [[ 1.   ,-0.08 , 0.001,-0.207,-0.003,-0.   , 0.013],
                   [-0.08 , 1.   , 0.049,-0.142,-0.003, 0.026,-0.018],
                   [ 0.001, 0.049, 1.   , 0.189,-0.009,-0.023, 0.076],
                   [-0.207,-0.142, 0.189, 1.   , 0.009,-0.004, 0.02 ],
                   [-0.003,-0.003,-0.009, 0.009, 1.   , 0.105,-0.075],
                   [-0.   , 0.026,-0.023,-0.004, 0.105, 1.   , 0.055],
                   [ 0.013,-0.018, 0.076, 0.02 ,-0.075, 0.055, 1.   ]]
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
            cov = [[ 1.   ,-0.112, 0.013,-0.197],
                   [-0.112, 1.   , 0.08 ,-0.13 ],
                   [ 0.013, 0.08 , 1.   , 0.042],
                   [-0.197,-0.13 , 0.042, 1.   ]]
        )
