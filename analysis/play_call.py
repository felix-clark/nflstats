#!/usr/bin/env python3

from typing import Any, Dict, List, Iterable, Optional

import numpy as np
import pandas as pd
from nptyping import NDArray
from sklearn.ensemble import RandomForestClassifier

from utility import split_test_train
from db import plays


def rfc_play_type(data: pd.DataFrame) -> RandomForestClassifier:
    """
    Test bed for Random Forest classification
    """
    clf = RandomForestClassifier(
        # The more trees, the fewer infinite logistic loss terms
        n_estimators=100,
        # n_estimators=20,  # temporarily turn this down for speed
        # n_estimators=200,  # temporarily turn this down for speed
        # consider changing to "entropy"; this reduces the logistic loss
        # criterion="gini",
        criterion="entropy",
        max_features="auto",
        n_jobs=4,
        # verbose=2,
        verbose=1,
        # regularization to prevent overfitting.
        # This slows down the calculation significantly
        # and increases the gini impurity, although it prevents infinities in the
        # entropy loss function.
        # A smaller value results in smaller loss usually, but increases the chance of
        # infinite loss.
        # ccp_alpha=0.0,
        ccp_alpha=5e-4,
    )
    data_test, data_train = split_test_train(data)
    y_data_train: pd.Series = data_train["play_type"]
    x_data_train: pd.DataFrame = data_train.drop(
        columns=["play_type", "desc"]
        # ).to_numpy()
    )
    fitted = clf.fit(x_data_train, y_data_train)
    print(fitted)

    y_data_test: pd.Series = data_test["play_type"]
    x_data_test: pd.DataFrame = data_test.drop(columns=["play_type", "desc"])
    # This is the mean accuracy of the most-likely prediction, not the probability or
    # info-loss score, so it's harsher than ideal.
    score: float = clf.score(x_data_test, y_data_test)
    print(f"score = {score}")
    print(x_data_train.columns)
    print(f"feature importance: {clf.feature_importances_}")
    play_types = clf.classes_
    pred_test_prob: NDArray[float] = clf.predict_proba(x_data_test)
    pred_test_log_prob: NDArray[float] = clf.predict_log_proba(x_data_test)
    print(play_types)
    print(pred_test_prob)
    # params = clf.get_params()
    # print(params)

    predictions_data: Dict[str, Any] = {"most_likely": clf.predict(x_data_test)}
    for play_type, play_probs in zip(play_types, pred_test_prob.T):
        predictions_data[play_type] = play_probs
    predictions: pd.DataFrame = pd.DataFrame(
        data=predictions_data, index=x_data_test.index
    )

    y_test_indices, y_factor_order = pd.factorize(y_data_test, sort=True)
    assert (y_factor_order == clf.classes_).all(), "Inconsistent factoring order"
    loss: float = 0.0
    gini_loss: float = 0.0
    for log_prob, prob, i_y in zip(pred_test_log_prob, pred_test_prob, y_test_indices):
        loss_term = -log_prob[i_y]
        if loss_term == np.inf:
            print(log_prob, i_y)

        loss += loss_term
        # try gini impurity to avoid INF for out-of-sample events
        gini_loss_term: float = 1.0 - prob[i_y]  # **2
        gini_loss += gini_loss_term
    loss /= len(y_test_indices)
    gini_loss /= len(y_test_indices)
    print(f"loss = {loss}")
    print(f"gini = {gini_loss}")

    for idx, test_play in x_data_test.sample(5).iterrows():
        print()
        print(data.loc[idx]["desc"])
        print(test_play)
        print(y_data_test.loc[idx])
        print(predictions.loc[idx])

    return fitted


def _get_data(columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    get and cache test data
    """
    cache_name = "play_cache.parquet"
    try:
        play_data: pd.DataFrame = pd.read_parquet(cache_name)
        # play_data.set_index(['game_id', 'play_id'], inplace=True)
        return play_data
    except Exception:
        print("Could not find cache")
    play_data = plays(
        range(2018, 2019+1), [("play_type", lambda s: ~s.isnull())], columns=columns
    )
    # discard kickoffs; these plays are mandatory. Kickoffs and onside kicks should be
    # handled with a separate model.
    # Also discard "no_play" which are penalties and timeouts, and extra points.
    play_data = play_data.loc[
        ~play_data["play_type"].isin(["no_play", "kickoff", "extra_point"])
    ]
    play_data.set_index(["game_id", "play_id"], inplace=True)
    play_data.to_parquet(cache_name)
    return play_data


def main():
    training_variables: List[str] = [
        "quarter_seconds_remaining",  # drop?
        "half_seconds_remaining",
        "game_seconds_remaining",
        # 'qtr', # redundant? Could use this along with quarter_seconds_remaining and drop the other two
        "down",
        # The yards to 1st down
        "ydstogo",
        # This appears to indicate the yards to the endzone
        "yardline_100",
        # equal to posteam_score - defteam_score
        "score_differential",
    ]
    columns = [
        # These two form a multi-index
        "game_id",
        "play_id",
        # a description for debugging
        "desc",
        # the indicator variable we want to predict
        "play_type",
        # These need to be discarded
        "extra_point_attempt",
        "two_point_attempt",
    ] + training_variables
    # To keep all fields, use this instead:
    # columns = None

    play_data: pd.DataFrame = _get_data(columns)
    # These are (mostly?) timeouts and penalties
    print(play_data["play_type"].unique())

    # Remove extra-point attempts; these should be handled with a separate classifier.
    # Will need to determine kick vs. run/pass and look for fakes.
    play_data = play_data.loc[
        (play_data["extra_point_attempt"] == 0) & (play_data["two_point_attempt"] == 0)
    ].drop(columns=["extra_point_attempt", "two_point_attempt"])

    # touchbacks and missed FGs that are received can have a NaN down. Just drop these
    # by the down for simplicity; a more complete disection would be good in the future.
    play_data = play_data.loc[~play_data["down"].isnull()]

    null_vals_df = play_data[play_data.isnull().any(axis=1)]
    # There are some weird plays, perhaps from missed field goals where the ball is received
    for _idx, row in null_vals_df.iterrows():
        print(row["desc"])
        print(row)

    # Random Forest #
    #################
    # rfc_results = rfc_play_type(play_data)
    rfc_play = rfc_play_type(play_data)


# Analysis TODO:
# Consider timeouts remaining
# Break pass plays up by pass location
# also run plays by gap? (or end/middle)


if __name__ == "__main__":
    main()
