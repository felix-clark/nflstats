from playermodels.template import *
import numpy as np
import logging

# a model rush attempts per game.
class RecRecModel(CountsModel):
    """
    statistical model for receptions.
    this should be split into targets first, but targets weekly data isn't currently part of our scraping.
    we should be able to get it from pro-football-reference's game logs but that'll take some messing.
    """
    name = 'rec' # do we actually need this?
    pred_var = 'rec' # the variable we're predicting
    dep_vars = ('targets') # variables this prediction depends on
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def _default_hyperpars(self, pos):
        pos = pos.upper()
        if pos == 'WR':
            return np.array((
                16.20, 4.026,
                0.533, 0.568, 0.958
                ))
            # return np.array(( # this is for the moment method
            #     2.0, 4.0, 100, # can't keep the parameters from going to invalid values
            #     1.0,
            #     1.0, 1.0
            #     ))
        if pos == 'TE':
            return np.array((
                16.18, 4.09,
                0.673, 0.935, 0.938
                ))
        if pos == 'RB':
            return np.array((
                1.71, 0.84,
                0.263, 0.603, 0.914))
        if pos == 'QB':
            logging.error('we aren\'t modeling receptions for QBs')
            logging.error(' since not even Tom Brady can catch a pass.')
        logging.error( 'positional defaults not implemented for {}'.format(pos) )
        pass

    
class RecTdModel(TrialModel):
    """
    TD rate per reception
    """
    name = 'rec_td'
    pred_var = 'rec_td'
    dep_vars = ('rec',)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def _default_hyperpars(self, pos):
        pos = pos.upper()
        if pos == 'WR':
            return np.array((
                36.67, 397.42, # initial bayes parameters
                1.0, # learn rate
                1.0, 0.976 # season,game memory
            ))
        if pos == 'TE':
            # this large learn rate with perfect memory seems a bit strange.
            # this type of thing suggests learn rate decay would be useful.
            return np.array((
                9.94, 105.5, # initial bayes parameters
                1.0, # learn rate
                1.0, 1.0 # season,game memory
            )) # game mem
        if pos == 'RB':
            return np.array((
                11.49, 361.51, # initial bayes parameters
                1.0, # learn rate
                1.0, 0.989 # season,game memory
            ))
        logging.error( 'rushing TD positional defaults not implemented for {}'.format(pos) )

   
class RecYdsModel(YdsPerAttModel):
    """
    receiving yards per catch
    """
    name = 'rec_yds'
    pred_var = 'rec_yds'
    dep_vars = ('rec',)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def _default_hyperpars(self, pos):
        pos = pos.upper()
        if pos == 'WR':
            return np.array((
                128.52, 11.96, 1.00, 55.83, # initial bayes parameters
                0.390, # skew
                0.0233, 0.0138, # learn rates
                1.0,0.994, # munu/nu, alpha/beta season memory
                1.0,0.999, # game memories
            ))
        if pos.upper() == 'TE':
            return np.array((
                117.8, 11.54, 1.51, 47.92, # initial bayes parameters
                0.1869, # skew
                0.0172, 0.000808, # learn rates
                0.920, # munu/nu memory
                1.0, # alpha/beta mem
                0.956,0.965 # game memories don't work well for WRs
            ))            
        if pos == 'RB':
            return np.array((
                12.01, 1.89, 152.6, 2650., # initial bayes parameters
                0.2635, # skew
                0.00262, 0.0184, # learn rates
                1.0,0.985, # season memory
                1.0,0.999 # game memories don't work well for WRs
            ))
        logging.error('no default hyperparameters are implemented for {}'.format(pos))
        pass
