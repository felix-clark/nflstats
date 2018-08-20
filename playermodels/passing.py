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
                0.817, 0.033, # bayes parameters
                0.080, # learn rate
                0.257, 0.857 # memories
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
            return np.array(( #TODO: run this
                60.16, 5.28, 7.30, 56.09, # initial bayes parameters
                0.0494, # skew
                0.0210, 0.00885, # learn rates
                0.678,0.923, # munu/nu, alpha/beta season memory
                0.982,0.994, # game memories
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
