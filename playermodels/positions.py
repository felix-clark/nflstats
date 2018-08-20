import logging
# from playermodels.qb import *
# from playermodels.rb import *
# from playermodels.wr import *
from playermodels.passing import *
from playermodels.rushing import *
from playermodels.receiving import *

def get_model_class(mname):
    """
    a convenience function to return a model type from its name.
    """
    models = {
        'rush_att': RushAttModel,
        'rush_yds': RushYdsModel,
        'rush_td': RushTdModel,
        'targets': RecTgtModel,
        'rec': RecModel,
        'rec_yds': RecYdsModel,
        'rec_td': RecTdModel,
        'pass_att': PassAttModel,
        'pass_cmp': PassCmpModel,
        'pass_yds': PassYdsModel,
        'pass_td': PassTdModel,
        'pass_int': PassIntModel,
    }
    if mname not in models:
        logging.error('could not provide model with name {}'.format(mname))
    else: return models[mname]
    
