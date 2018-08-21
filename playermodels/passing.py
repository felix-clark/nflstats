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
                0.794, 0.0342, # bayes parameters
                0.066, # learn rate
                0.233,# season mem
                0.858, # 0.05 # game mem
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
                88.88, 62.28, # initial bayes parameters
                0.361, # learn rate
                0.853, 0.964 # season,game memory
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
                12.15/4, 1.079/4, 0.395/2, 1.08/2, # initial bayes parameters
                2.55, # skew
                0.00246, 1.46e-7, # learn rates
                0.706,0.863, # munu/nu, alpha/beta season memory
                0.999,1.0, # game memories
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
                71.15, 962.47, # initial bayes parameters
                1.0, # learn rate
                0.905, 1.0 # season,game memory
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
                18.75, 629.9, # initial bayes parameters
                0.400, # learn rate
                0.796, 1.0 # season,game memory
            ))
        else:
            logging.error('only QBs have passing models (not {}s)'.format(pos))
        pass
