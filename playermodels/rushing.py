from playermodels.template import *
import numpy as np
import logging

# a model rush attempts per game.
class RushAttModel(CountsModel):
    """
    statistical model for rushing attempts by RBs, QBs, and WRs.
    """
    name = 'rush_att'
    
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
        if pos == 'RB':
            return np.array((
                2.807, 0.244,
                0.121, 0.677, 0.782))
        if pos == 'QB':
            return np.array((
                2.61, 0.840,
                0.241, 0.703, 0.954))
        if pos == 'WR':
            return np.array((
                0.516, 3.65,
                0.646, 0.523, 0.972
                ))
        if pos == 'TE':
            logging.error('TEs do not rush enough to try to predict them')
        logging.error( 'positional defaults not implemented for {}'.format(pos) )
        pass

    # the variable we're predicting
    @property
    def pred_var(self):
        return 'rushing_att'
    
    @property
    def dep_vars(self):
        return ()


class RushYdsModel(YdsPerAttModel):
    """
    statistical model for yards per rush attempt
    """
    name = 'rush_yds'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @classmethod
    def for_position(self, pos):
        return self(*self._default_hyperpars(pos))
    
    @classmethod
    def _default_hyperpars(self, pos):
        pos = pos.upper()
        if pos == 'RB':
            return np.array((
                122.3, 36.26, 8.89, 40.08, # initial bayes parameters
                0.793, # skew
                0.00259, 0.0191, # learn rates
                1.0, # munu/nu memory
                0.979,# alpha/beta mem
                1.0,0.966, # game memories
            ))
        if pos == 'QB':
            return np.array((
                112.9, 48.51, 2.85, 50.35, # initial bayes parameters
                0.0552, # skew# possibly low due to sacks?
                6.266, 0.0272, # learn rates
                0.667, # munu/nu memory
                0.769, # alpha/beta mem
                1.0,1.0 # game memories don't help for QBs
            ))
        if pos == 'WR':
            # this has a reasonable flat CDF
            # interesting that there is no memory
            return np.array((
                116.3, 41.28, 3.46, 45.64, # initial bayes parameters
                0.457, # skew
                0.563, 0.0, # learn rates
                1.0, # munu/nu memory
                1.0, # alpha/beta mem
                1.0,1.0 # game memories don't work well for WRs
            ))
        if pos.upper() == 'TE': # TEs rush so rarely we shouldn't even include them
            logging.error('TEs do not run enough to try to predict their rushes')
        logging.error('no default hyperparameters are implemented for {}'.format(pos))
        pass

    @property
    def pred_var(self):
        return 'rushing_yds'
    
    # dependent variables i.e. those required for prediction
    @property
    def dep_vars(self):
        return ('rushing_att',)# self.__dep_vars


class RushTdModel(TrialModel):
    """
    statistical model for TDs per rush
    """
    name = 'rush_tds'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def for_position(self, pos):
        return self(*self._default_hyperpars(pos))
    
    @classmethod
    def _default_hyperpars(self, pos):
        pos = pos.upper()
        if pos == 'RB':
            return np.array((
                19.07, 684.7, # initial bayes parameters
                1.84, # learn rate
                0.775, # season memory
                1.0)) # game mem
        if pos == 'QB':
            return np.array((
                12.33, 330.27, # initial bayes parameters
                1.76, # learn rate
                1.0, # season memory
                0.980)) # game mem
        if pos == 'WR':
            # rushing TDs are rare enough for WRs that there's no reason to update a model from an event
            return np.array((
                1.97, 60.4, # initial bayes parameters
                0.0, # learn rate
                1.0, # season memory
                1.0)) # game mem
        if pos == 'TE':
            logging.error('TEs do not rush enough')
        logging.error( 'rushing TD positional defaults not implemented for {}'.format(pos) )

    @property
    def pred_var(self):
        return 'rushing_tds'
    
    # dependent variables i.e. those required for prediction
    @property
    def dep_vars(self):
        return ('rushing_att',)# self.__dep_vars
    
