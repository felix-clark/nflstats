from playermodels.template import *
import numpy as np
import logging

# a model rush attempts per game.
class RushAttModel(CountsModel):
    """
    statistical model for rushing attempts by RBs, QBs, and WRs.
    """
    name = 'rush_att'
    pred_var = 'rush_att'
    dep_vars = ()
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def _default_hyperpars(self, pos):
        pos = pos.upper()
        if pos == 'RB':
            return np.array((
                0.680, 0.0835,
                0.625, 0.0825,
                0.125,
                0.382, # seasonal parameter decay
                0.720, # game memory # this means the memory is only about 4 games. need more info.
                ))
        if pos == 'QB':
            return np.array((
                1.40, 0.52,
                0.90, 0.43,
                0.233,
                0.433,
                0.953, # 0.0282,
            ))
        if pos == 'WR':
            return np.array((
                0.78, 6.29,
                0.78, 6.29,
                0.759,
                0.718,
                1.0, # 0.128
                ))
        if pos == 'TE':
            logging.error('TEs do not rush enough to try to predict them')
        logging.error( 'positional defaults not implemented for {}'.format(pos) )


class RushYdsModel(YdsPerAttModel):
    """
    statistical model for yards per rush attempt
    """
    name = 'rush_yds'
    pred_var = 'rush_yds'
    dep_vars = ('rush_att',)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @classmethod
    def _default_hyperpars(self, pos):
        pos = pos.upper()
        if pos == 'RB':
            return np.array((
                4.07, 1.08, 3.45, 4.36, # initial bayes parameters
                1.46, # skew
                0.0025, 0.0010, # learn rates
                0.833, 0.967, # munu/nu, a/b season memory
                1.0, 1.0, # game memories
            ))
        if pos == 'QB':
            return np.array(( # TODO: here and other positions
                2.15, 3.04, 2.12, 38.83, # initial bayes parameters
                0.157, # skew # low due to sacks? and QBs don't often break away w/ big runs
                0.872,  3.0e-7, # learn rates
                0.747, 0.948, # munu/nu;a/b season memory
                0.982, 0.905
            ))
        if pos == 'WR':
            # this has a reasonable flat CDF
            # interesting that there is no memory
            return np.array((
                116.2, 41.84, 2.45, 46.42, # initial bayes parameters
                0.348, # skew
                0.332, 0.00092, # learn rates
                1.0,1.0, # munu/nu alpha/beta mem
                1.0,1.0 # game memories don't work well for WRs
            ))
        if pos.upper() == 'TE': # TEs rush so rarely we shouldn't even include them
            logging.error('TEs do not run enough to try to predict their rushes')
        logging.error('no default hyperparameters are implemented for {}'.format(pos))


class RushTdModel(TrialModel):
    """
    statistical model for TDs per rush
    """
    name = 'rush_td'
    pred_var = 'rush_td'
    dep_vars = ('rush_att',)# self.__dep_vars
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def _default_hyperpars(self, pos):
        pos = pos.upper()
        if pos == 'RB':
            return np.array((
                5.41, 188.2, # initial bayes parameters
                6.12, 209.9, # attractor bayes parameters
                1.0, # learn rate
                0.422, # season memory
                1.0 # game mem
            ))
        if pos == 'QB':
            return np.array((
                5.84, 126.0, # initial bayes parameters
                5.45, 176.1, # attractor bayes parameters
                1.0, # learn rate
                0.704, # season memory
                1.0)) # game mem
        if pos == 'WR':
            return np.array((
                1.14, 60.63, # initial bayes parameters
                1.14, 60.63, # attractor bayes parameters
                0.413, # learn rate
                0.997, # season memory
                0.989)) # game mem
        if pos == 'TE':
            logging.error('TEs do not rush enough')
        logging.error( 'rushing TD positional defaults not implemented for {}'.format(pos) )

