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
                0.816, 0.033, # bayes parameters
                0.080, # learn rate
                0.257,# season mem
                0.857, # 0.05 # game mem
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
                64.38, 44.67, # initial bayes parameters
                0.332, # learn rate
                0.888, 0.964 # season,game memory
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
                1.144, 0.0987, 1.075, 0.536, # initial bayes parameters
                2.56, # skew
                2.99e-4, 2.72e-6, # learn rates
                0.709,0.865, # munu/nu, alpha/beta season memory
                0.982,1.0, # game memories
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
                6.02, 201.8, # initial bayes parameters
                0.235, # learn rate
                0.903, 0.995 # season,game memory
            ))
        else:
            logging.error('only QBs have passing models (not {}s)'.format(pos))
        pass
