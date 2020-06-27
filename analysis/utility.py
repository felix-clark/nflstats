from typing import Tuple

import numpy as np
import pandas as pd


def split_test_train(
    data: pd.DataFrame, test_frac: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split test and training data with an average fraction
    TODO: make the shuffling take a constant number
    """
    data["training"] = np.random.uniform(0, 1, len(data)) >= test_frac
    data_train = data.loc[data["training"]].drop(columns="training")
    data_test = data.loc[~data["training"]].drop(columns="training")
    return data_test, data_train
