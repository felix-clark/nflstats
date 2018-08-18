import numpy as np

from playermodels.receiving import *
from playermodels.rushing import *

class WrModel:
    """
    represents the stats to track for wideouts
    """
    def __init__(self):
        self.recrec = RecRecModel.for_position('RB')
        self.recyds = RecYdsModel.for_position('RB')
        self.rectds = RecTdModel.for_position('RB')
        self.rushatt = RushAttModel.for_position('RB')
        self.rushyds = RushYdsModel.for_position('RB')
        self.rushtds = RushTdModel.for_position('RB')
        # mind the order
        # we may or may not want to correlate rush and rec
        # self.rush_gen = st.multivariate_normal(
        # cov=[
        #     [1.0,     0.1078, -0.1310],
        #     [0.1078,  1.0,    0.1316 ],
        #     [-0.1310, 0.1316, 1.0    ]
        # ])
        # self.rec_gen = st.multivariate_normal()

    def update_game(self, game):
        """
        update the stats based on the results of 1 game
        we can either do this stochastically using the KLD gradient and a variable learn rate, or using bayesian models
        """
        rec_rec = game['receiving_rec']
        rec_yds = game['receiving_yds']
        rec_tds = game['receiving_tds']
        rush_att = game['rushing_att']
        rush_yds = game['rushing_yds']
        rush_tds = game['rushing_tds']
        self.recrec.update_game(rec_rec)
        self.recyds.update_game(rec_yds, rec_rec)
        self.rectds.update_game(rec_tds, rec_rec)
        self.rushatt.update_game(rush_att)
        self.rushyds.update_game(rush_yds, rush_att)
        self.rushtds.update_game(rush_tds, rush_att)
