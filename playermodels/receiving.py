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
    name = 'rec_rec' # do we actually need this?
    pred_var = 'receiving_rec' # the variable we're predicting
    dep_vars = () # variables this prediction depends on
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def for_position(self, pos):
        """
        provides an instance of the model with default hyperparameters
        """
        return self(*self._default_hyperpars(pos))

    @classmethod
    def _default_hyperpars(self, pos):
        pos = pos.upper()
        if pos == 'WR':
            return np.array((
                16.20, 4.026,
                0.533, 0.568, 0.958
                ))
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
    name = 'rec_tds'
    pred_var = 'receiving_tds'
    dep_vars = ('receiving_rec',)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def for_position(self, pos):
        return self(*self._default_hyperpars(pos))

    @classmethod
    def _default_hyperpars(self, pos):
        pos = pos.upper()
        if pos == 'WR':
            return np.array((
                36.65, 397.42, # initial bayes parameters
                1.955, # learn rate
                0.748, 1.0 # season,game memory
            ))
        if pos == 'TE':
            # this large learn rate with perfect memory seems strange.
            # this type of thing suggests learn rate decay would be useful.
            return np.array((
                91.99, 979.48, # initial bayes parameters
                9.272, # learn rate
                1.0, 1.0 # season,game memory
            )) # game mem
        if pos == 'RB':
            return np.array(( # TODO
                1.0, 50.0, # initial bayes parameters
                1.0, # learn rate
                0.8, 1.0 # season,game memory
            ))
        logging.error( 'rushing TD positional defaults not implemented for {}'.format(pos) )

class RecYdsModel(YdsPerAttModel):
    """
    receiving yards per catch
    """
    pred_var = 'receiving_yds'
    dep_vars = ('receiving_rec',)
    # let's leave out name for now
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def for_position(self, pos):
        return self(*self._default_hyperpars(pos))
    
