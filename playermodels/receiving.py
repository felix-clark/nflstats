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
            return np.array((
                11.85, 371.78, # initial bayes parameters
                1.221, # learn rate
                1.0, 0.994 # season,game memory
            ))
        logging.error( 'rushing TD positional defaults not implemented for {}'.format(pos) )

   
class RecYdsModel(YdsPerAttModel):
    """
    receiving yards per catch
    """
    name = 'rec_yds'
    pred_var = 'receiving_yds'
    dep_vars = ('receiving_rec',)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def _default_hyperpars(self, pos):
        pos = pos.upper()
        if pos == 'WR': # TODO: check this again; probably need to cap the variance learn rate
            # this seems like a very low learn rate for the mean and a very high learn rate for the variance...
            return np.array((
                128.54, 11.94, 0.978, 55.77, # initial bayes parameters
                0.432, # skew
                0.00516, 8.89, # learn rates
                1.0, # munu/nu memory
                0.617, # alpha/beta mem
                0.971,1.0, # game memories
            ))
        if pos.upper() == 'TE': # TODO - refine? it ran out of iterations but didn't change much and the parameters don't seem crazy
            return np.array((
                117.8, 11.54, 1.51, 47.92, # initial bayes parameters
                0.1869, # skew
                0.0172, 0.000808, # learn rates
                0.920, # munu/nu memory
                1.0, # alpha/beta mem
                0.956,0.965 # game memories don't work well for WRs
            ))            
        if pos == 'RB':
            return np.array(( # TODO? check and/or refine? full memory is weirdish, but rest seems fine
                116.3, 41.28, 3.46, 45.64, # initial bayes parameters
                0.457, # skew
                0.563, 0.1, # learn rates
                1.0, # munu/nu memory
                1.0, # alpha/beta mem
                1.0,1.0 # game memories don't work well for WRs
            ))
        logging.error('no default hyperparameters are implemented for {}'.format(pos))
        pass
