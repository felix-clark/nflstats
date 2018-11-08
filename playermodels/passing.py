from playermodels.template import *
import numpy as np
import logging

class PassAttModel(CountsModel):
    name = 'pass_att'
    pred_var = 'pass_att'
    dep_vars = ()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def _default_hyperpars(self, pos):
        pos = pos.upper()
        if pos == 'QB':
            return np.array((
                0.727, 0.032, # initial bayes parameters
                0.286, 0.010, # attractor bayes parameters
                0.070, # learn rate
                0.193,# season mem
                0.836, # game mem
            ))
        else:
            logging.error('no passing models for {}'.format(pos))
        pass


class PassCmpModel(TrialModel):
    name = 'pass_cmp'
    pred_var = 'pass_cmp'
    dep_vars = ('pass_att',)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @classmethod
    def _default_hyperpars(self, pos):
        pos = pos.upper()
        if pos == 'QB':
            return np.array((
                61.95, 47.90, # initial bayes parameters
                34.56, 19.07, # attractor bayes parameters
                0.425, # learn rate
                0.677, 0.976 # season,game memory
            ))
        else:
            logging.error('only QBs have passing models (not {}s)'.format(pos))
        pass

class PassYdsModel(YdsPerAttModel):
    """
    receiving yards per catch
    """
    name = 'pass_yds'
    pred_var = 'pass_yds'
    dep_vars = ('pass_cmp',)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def _default_hyperpars(self, pos):
        pos = pos.upper()
        if pos == 'QB':
            return np.array((
                1.16, 0.062, 1.34, 0.467, # initial bayes parameters
                2.55, # skew
                0.127, 0.0038, # learn rates
                0.711,0.867, # munu/nu, alpha/beta season memory
                0.992,1.0, # game memories
                ))
        else:
            logging.error('only QBs will have passing models (not {}s)'.format(pos))
        pass

class PassTdModel(TrialModel):
    name = 'pass_td'
    pred_var = 'pass_td'
    dep_vars = ('pass_cmp',)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @classmethod
    def _default_hyperpars(self, pos):
        pos = pos.upper()
        if pos == 'QB':
            return np.array((
                67.15, 962.76, # initial bayes parameters
                70.67, 962.48, # attractor bayes parameters
                1.0, # learn rate
                0.909, 0.986 # season,game memory
            ))
        else:
            logging.error('only QBs have passing models (not {}s)'.format(pos))
        pass

class PassIntModel(TrialModel):
    name = 'pass_int'
    pred_var = 'pass_int'
    dep_vars = ('pass_att',)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @classmethod
    def _default_hyperpars(self, pos):
        pos = pos.upper()
        if pos == 'QB':
            return np.array((
                6.99, 201.8, # initial bayes parameters
                3.95, 201.9, # attractor bayes parameters
                0.265, # learn rate
                0.861, 1.0 # season,game memory
            ))
        else:
            logging.error('only QBs have passing models (not {}s)'.format(pos))
        pass
