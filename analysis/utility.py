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


def drop_noinfo(data: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns with exclusively null or redundant data
    """
    # This removes null and single-valued data
    noninfo_cols = [col for col in data.keys() if len(data[col].unique()) == 1]
    for col in noninfo_cols:
        print(col, data[col].unique())
    return data.drop(columns=noninfo_cols)
