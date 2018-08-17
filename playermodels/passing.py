from playermodels.template import *
import numpy as np
import logging

class PassAttModel(CountsModel):
    name = 'pass_att'
    pred_var = 'passing_att'
    dep_vars = ()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def _default_hyperpars(self, pos):
        pos = pos.upper()
        if pos == 'QB':
            return np.array((
                22.20, 0.656,
                0.0550, 0.839, 0.956))
        else:
            logging.error('no passing models for {}'.format(pos))
        pass


class PassCmpModel(TrialModel):
    name = 'pass_cmp'
    pred_var = 'passing_cmp'
    dep_vars = ('passing_att',)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @classmethod
    def _default_hyperpars(self, pos):
        pos = pos.upper()
        if pos == 'QB':
            return np.array((
                93.07, 55.72, # initial bayes parameters
                0.218, # learn rate
                0.966, 0.964 # season,game memory
            ))
        else:
            logging.error('only QBs have passing models (not {}s)'.format(pos))
        pass

class PassYdsModel(YdsPerAttModel):
    """
    receiving yards per catch
    """
    name = 'pass_yds'
    pred_var = 'passing_yds'
    dep_vars = ('passing_cmp',)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def _default_hyperpars(self, pos):
        pos = pos.upper()
        if pos == 'QB':
            return np.array(( #TODO: run this
                60.09, 5.39, 9.08, 55.51, # initial bayes parameters
                0.396, # skew
                0.00145, 0.00296, # learn rates
                1.0,0.964, # munu/nu, alpha/beta season memory
                1.0,0.963, # game memories
                ))
        else:
            logging.error('only QBs will have passing models (not {}s)'.format(pos))
        pass

class PassTdModel(TrialModel):
    name = 'pass_tds'
    pred_var = 'passing_tds'
    dep_vars = ('passing_cmp',)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @classmethod
    def _default_hyperpars(self, pos):
        pos = pos.upper()
        if pos == 'QB':
            return np.array((
                98.06, 1209.5, # initial bayes parameters
                1.0, # learn rate
                1.0, 0.988 # season,game memory
            ))
        else:
            logging.error('only QBs have passing models (not {}s)'.format(pos))
        pass

class PassIntModel(TrialModel):
    name = 'pass_int'
    pred_var = 'passing_int'
    dep_vars = ('passing_att',)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @classmethod
    def _default_hyperpars(self, pos):
        pos = pos.upper()
        if pos == 'QB':
            return np.array((
                15.81, 630.0, # initial bayes parameters
                0.271, # learn rate
                1.0, 0.987 # season,game memory
            ))
        else:
            logging.error('only QBs have passing models (not {}s)'.format(pos))
        pass
