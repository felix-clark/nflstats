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
                1.36, 0.304,
                0.306, # lr
                0.422, # season mem
                0.865, # game mem
                ))
        if pos == 'TE':
            return np.array((
                1.03, 0.442,
                0.339,
                0.375,
                0.889,
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
                9.22, 100.16, # initial bayes parameters
                0.476, # learn rate
                0.920, 1.0 # season,game memory
            ))
        if pos == 'TE':
            # this large learn rate with perfect memory seems a bit strange.
            # this type of thing suggests learn rate decay would be useful.
            return np.array((
                4.93, 52.79, # initial bayes parameters
                0.463, # learn rate
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
                6.66, 0.532, 0.645, 3.49, # initial bayes parameters -- consider reducing and re-fitting
                0.940, # skew
                0.0037, 3.45e-4, # learn rates
                0.967,1.0, # munu/nu, alpha/beta season memory
                0.990,1.0, # game memories
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
