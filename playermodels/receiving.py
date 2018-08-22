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
                1.153, 0.294,
                0.289, # lr
                0.378, # season mem
                0.858, # game mem
                ))
        if pos == 'TE':
            return np.array((
                0.839, 0.409,
                0.304,
                0.374,
                0.887,
                ))
        if pos == 'RB':
            return np.array((
                0.88, 0.457,
                0.276,
                0.401,
                0.895,
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
                30.04, 23.96, # initial bayes parameters
                0.398, # learn rate
                0.698, 0.995 # season,game memory
            ))
        if pos == 'TE':
            return np.array((
                37.72, 21.87, # initial bayes parameters
                0.280, # learn rate
                0.899, 1.0 # season,game memory
            ))
        if pos == 'RB':
            return np.array((
                37.47, 13.95, # initial bayes parameters
                0.185, # learn rate
                0.922, 1.0 # season,game memory
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
                8.90, 100.19, # initial bayes parameters
                0.542, # learn rate
                0.903, 1.0 # season,game memory
            ))
        if pos == 'TE':
            # this large learn rate with perfect memory seems a bit strange.
            # this type of thing suggests learn rate decay would be useful.
            return np.array((
                4.87, 53.26, # initial bayes parameters
                0.403, # learn rate
                1.0, 1.0 # season,game memory
            )) # game mem
        if pos == 'RB':
            return np.array((
                11.84, 361.48, # initial bayes parameters
                0.659, # learn rate
                0.784, 1.0 # season,game memory
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
                6.66, 0.546, 0.645, 3.49, # initial bayes parameters -- consider reducing and re-fitting
                0.940, # skew
                0.00394, 3.85e-4, # learn rates
                0.967,1.0, # munu/nu, alpha/beta season memory
                0.991,1.0, # game memories
            ))
        if pos.upper() == 'TE':
            return np.array((
                5.86, 0.603, 0.596, 2.40, # initial bayes parameters -- not much sensitivity to these!
                0.498, # skew
                0.0052, 0.00040, # learn rates
                0.992, 1.0, # munu/nu memory, alpha/beta mem
                0.983, 1.0
            ))
        if pos == 'RB':
            return np.array((
                12.12, 1.93, 8.16, 132.47, # initial bayes parameters
                0.256, # skew
                0.0017, 0.0060, # learn rates
                1.0, 0.964, # season memory
                1.0, 1.0 # game memory
            ))
        logging.error('no default {} hyperparameters are implemented for {}'.format(self.name, pos))
        pass
