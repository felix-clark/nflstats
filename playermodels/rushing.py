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
                0.641, 0.0746,
                0.121,
                0.464, # seasonal parameter decay
                0.754, # game memory # this means the memory is only about 4 games. need more info.
                ))
        if pos == 'QB':
            return np.array((
                1.76, 0.657,
                0.225,
                0.545,
                0.953, # 0.0282,
            ))
        if pos == 'WR':
            return np.array((
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
                111.59, 42.13, 2.82, 50.32, # initial bayes parameters
                0.0472, # skew # low due to sacks? and QBs don't often break away w/ big runs
                0.940,  4.2e-5, # learn rates
                0.805, 0.985, # munu/nu;a/b season memory
                0.972, 0.942
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
                20.51, 684.67, # initial bayes parameters
                1.0, # learn rate
                0.633, # season memory
                1.0)) # game mem
        if pos == 'QB':
            return np.array((
                12.67, 330.26, # initial bayes parameters
                1.0, # learn rate
                0.862, # season memory
                1.0)) # game mem
        if pos == 'WR':
            return np.array((
                1.14, 60.63, # initial bayes parameters
                0.413, # learn rate
                0.997, # season memory
                0.989)) # game mem
        if pos == 'TE':
            logging.error('TEs do not rush enough')
        logging.error( 'rushing TD positional defaults not implemented for {}'.format(pos) )

