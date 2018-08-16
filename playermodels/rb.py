import numpy as np

from playermodels.rushing import *

# we'll want to split this up into a few types
class RbModel:
    """
    represents a statistical model for a single season for a running back.
    """
    def __init__(self):
        self.rushatt = RushAttModel.for_position('RB')
        self.rushyds = RushYdsModel.for_position('RB')
        self.rushtds = RushTdModel.for_position('RB')
        # the order here is important: att, yds, tds
        # the covariance matrix is the spearman (rank) correlations between
        # the model's cdf values for each training data point.
        # the correlation is computed weighting for rushing attempts, which makes the correlations smaller.
        # in theory, this correlation matrix itself could be updated each season for each player individually.
        self.rush_gen = st.multivariate_normal(
        cov=[
            [1.0,     0.1078, -0.1310],
            [0.1078,  1.0,    0.1316 ],
            [-0.1310, 0.1316, 1.0    ]
        ])
        # rushing_att  rushing_ypa  rushing_tdpa

    def gen_game(self):
        game = {}
        # call the ppf of each member model
        # the generator should handle the correlations
        ura,ury,urt = self.rush_gen.rvs()
        rush_att = self.rushatt.ppf(ura)
        rush_yds = self.rushyds.ppf(rush_att, ury)
        rush_tds = self.rushtds.ppf(rush_att, urt)
        game['rushing_att'] = rush_att
        game['rushing_yds'] = rush_yds
        game['rushing_tds'] = rush_tds
        return game

    def update_game(self, game):
        """
        update the stats based on the results of 1 game
        we can either do this stochastically using the KLD gradient and a variable learn rate, or using bayesian models
        """
        rush_att = game['rushing_att']
        rush_yds = game['rushing_yds']
        rush_tds = game['rushing_tds']
        self.rushatt.update_game(rush_att)
        self.rushyds.update_game(rush_yds, rush_att)
        self.rushtds.update_game(rush_tds, rush_att)

    def new_season(self):
        """
        decay some parameters to account for uncertainty between seasons
        """
        # TODO: if we change these in the future, we have to remember to update this here.
        # figure out a cleaner way of doing this.
        self.rushatt.new_season()
        self.rushyds.new_season()
        self.rushtds.new_season()

