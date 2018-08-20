from playermodels.template import *
import numpy as np
import logging
    
# a model rush attempts per game.
class RecTgtModel(CountsModel):
    """
    statistical model for targets
    """
    name = 'targets' # do we actually need this?
    pred_var = 'targets' # the variable we're predicting
    dep_vars = () # variables this prediction depends on
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def _default_hyperpars(self, pos):
        pos = pos.upper()
        if pos == 'WR':
            return np.array((
                1.42, 0.315,
                0.305,
                0.421, 0.865
                ))
        if pos == 'TE':
            return np.array((
                1.03, 0.443,
                0.339,
                0.375, 0.889
                ))
        if pos == 'RB':
            return np.array((
                1.05, 0.457,
                0.279,
                0.442, 0.891
            ))
        if pos == 'QB':
            logging.error('we aren\'t modeling receptions for QBs')
            logging.error(' since not even Tom Brady can catch a pass.')
        logging.error( 'positional defaults not implemented for {}'.format(pos) )
        pass


class RecModel(TrialModel):
    """
    model for receptions
    """
    name = 'rec' # do we actually need this?
    pred_var = 'rec' # the variable we're predicting
    dep_vars = ('targets',) # variables this prediction depends on

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def _default_hyperpars(self, pos):
        pos = pos.upper()
        if pos == 'WR':
            return np.array((
                30.10, 23.88, # initial bayes parameters
                0.400, # learn rate
                0.660, 0.998 # season,game memory
            ))
        if pos == 'TE':
            return np.array((
                33.47, 19.39, # initial bayes parameters
                0.245, # learn rate
                0.900, 1.0 # season,game memory
            ))
        if pos == 'RB':
            return np.array((
                37.55, 13.71, # initial bayes parameters
                0.173, # learn rate
                0.941, 1.0 # season,game memory
            ))
        logging.error( 'rushing TD positional defaults not implemented for {}'.format(pos) )
    
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
                9.22, 100.15, # initial bayes parameters
                0.476, # learn rate
                0.918, 1.0 # season,game memory
            ))
        if pos == 'TE':
            # this large learn rate with perfect memory seems a bit strange.
            # this type of thing suggests learn rate decay would be useful.
            return np.array((
                4.92, 52.78, # initial bayes parameters
                0.464, # learn rate
                1.0, 1.0 # season,game memory
            )) # game mem
        if pos == 'RB':
            return np.array((
                11.22, 361.5, # initial bayes parameters
                0.582, # learn rate
                0.975, 0.988 # season,game memory
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
                33.13, 3.07, 0.35, 13.95, # initial bayes parameters
                0.408, # skew
                0.0021, 0.0018, # learn rates
                1.0,0.991, # munu/nu, alpha/beta season memory
                1.0,0.995, # game memories
            ))
        if pos.upper() == 'TE': # TODO: HERE and beyond
            return np.array((
                117.8/5, 11.54/5, 1.51/5, 47.92/5, # initial bayes parameters
                0.1869, # skew
                0.0172, 0.000808, # learn rates
                0.920, # munu/nu memory
                1.0, # alpha/beta mem
                0.956,0.965 # game memories don't work well for WRs
            ))            
        if pos == 'RB':
            return np.array((
                12.01, 1.89, 152.6/20, 2650./20, # initial bayes parameters
                0.2635, # skew
                0.00262, 0.0184, # learn rates
                1.0,0.985, # season memory
                1.0,0.999 # game memories don't work well for WRs
            ))
        logging.error('no default hyperparameters are implemented for {}'.format(pos))
        pass
