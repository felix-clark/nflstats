import logging
from playermodels.rb import *
from playermodels.wr import *


def get_model_class(mname):
    """
    a convenience function to return a model type from its name.
    """
    if mname == 'rush_att': return RushAttModel
    if mname == 'rush_yds': return RushYdsModel
    if mname == 'rush_tds': return RushTdModel
    if mname == 'rec_rec': return RecRecModel
    if mname == 'rec_yds': return RecYdsModel
    if mname == 'rec_tds': return RecTdModel

    logging.error('could not provide model with name {}'.format(mname))
