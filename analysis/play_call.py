#!/usr/bin/env python3

from typing import Iterable, Optional
import pandas as pd
from db import plays
from sklearn.ensemble import RandomForestClassifier


def rfc(data: pd.DataFrame):
    """
    Test bed for Random Forest classification
    """
    clf = RandomForestClassifier(
        n_estimators=100,
        # consider changing to "entropy"
        criterion='gini',
        max_features='auto',
        n_jobs=2,
        verbose=2,
        # regularization to prevent overfitting.
        ccp_alpha=0.,
    )
    # TODO: split into test and training
    y_data = data['play_type']
    X_data = data.drop(columns=['play_type', 'desc']).to_numpy()
    print(y_data, X_data)
    fitted = clf.fit(X_data, y_data)
    print(fitted)
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
    play_data = plays([2018, 2019], [("play_type", lambda s: ~s.isnull())], columns=columns)
    # discard kickoffs; these plays are mandatory. Kickoffs and onside kicks should be
    # handled with a separate model.
    # Also discard "no_play" which are penalties and timeouts, and extra points.
    play_data = play_data.loc[~play_data["play_type"].isin(['no_play', 'kickoff', 'extra_point'])]
    play_data.set_index(['game_id', 'play_id'], inplace=True)
    play_data.to_parquet(cache_name)
    return play_data


def main():
    columns = [
        # These two form a multi-index
        'game_id',
        'play_id',
        # a description for debugging
        'desc',
        # the indicator variable we want to predict
        'play_type',
        # 'posteam', # could give a team-dependent description
        # 'posteam_type', # whether possession team is home or away. This is None for the SB
        # TODO: This appears to depend on the quarter so isn't a good way of determining
        # yards to goal. It would need to be combined with 'side_of_field', and others
        # to determine yards to goal.
        # 'yrdln', + 'posteam'/'posteam_type' could also probably do it.
        # 'yardline_100',
        'quarter_seconds_remaining',  # drop?
        'half_seconds_remaining',
        'game_seconds_remaining',
        # 'qtr', # redundant?
        'down',
        # This is only a flag of whether the down is "& GOAL"
        'goal_to_go',
        'ydstogo',
        # equal to posteam_score - defteam_score
        'score_differential',
        # These need to be discarded
        'extra_point_attempt',
        'two_point_attempt',
    ]
    # To keep all fields, use this instead:
    # columns = None

    play_data: pd.DataFrame = _get_data(columns)
    print(play_data)
    # These are (mostly?) timeouts and penalties
    print(play_data['play_type'].unique())

    play_data = play_data.loc[(play_data['extra_point_attempt'] == 0) & (play_data['two_point_attempt'] == 0)]

    null_vals_df = play_data[play_data.isnull().any(axis=1)]
    # There are some weird plays, perhaps from missed field goals where the ball is received
    for row in null_vals_df.iterrows():
        print(row)

    #################
    # Random Forest #
    #################
    rfc_results = rfc(play_data)

# Analysis TODO:
# Consider timeouts remaining
# Break pass plays up by pass location
# also run plays by gap? (or end/middle)


if __name__ == "__main__":
    main()
