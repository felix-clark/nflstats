#!/usr/bin/env python3

from typing import Iterable, Optional
import pandas as pd
from db import plays


def _get_data(columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    get and cache test data
    """
    cache_name = "play_cache.parquet"
    try:
        play_data: pd.DataFrame = pd.read_parquet(cache_name)
        return play_data
    except Exception:
        print("Could not find cache")
    play_data = plays([2018, 2019], [("play_type", lambda s: ~s.isnull())], columns=columns)
    # discard kickoffs; these plays are mandatory. Kickoffs and onside kicks should be
    # handled with a separate model.
    # Also discard "no_play which are penalties and timeouts"
    play_data = play_data.loc[~play_data["play_type"].isin(['no_play', 'kickoff'])]
    play_data.to_parquet(cache_name)
    return play_data


def main():
    # discard special play types like game beginning/end and comments.
    # These could be accessed with 'play_type_nfl'.
    play_data: pd.DataFrame = _get_data()
    print(play_data)
    print(play_data["play_type"].unique())
    print(play_data.columns)
    # These are (mostly?) timeouts and penalties
    print(play_data[play_data['play_type'] == 'no_play']['desc'])
    columns = [
        # the indicator variable we want to predict
        'play_type',
        # 'posteam', # could give a team-dependent description
        # 'posteam_type', # whether possession team is home or away. This is None for the SB
        # TODO: This appears to depend on the quarter so isn't a good way of determining
        # yards to goal. It would need to be combined with 'side_of_field', and others
        # to determine yards to goal.
        # 'yrdln', + 'posteam'/'posteam_type' could also probably do it.
        #'yardline_100',
        'quarter_seconds_remaining', # drop?
        'half_seconds_remaining',
        'game_seconds_remaining',
        # 'qtr', # redundant?
        'down',
        # This is only a flag of whether the down is "& GOAL"
        'goal_to_go',
        'ydstogo',
        # equal to posteam_score - defteam_score
        'score_differential',
    ]
    play_data = play_data[columns]
    print(play_data)

# Analysis TODO:
# Consider timeouts remaining
# Break pass plays up by pass location
# also run plays by gap? (or end/middle)

# posteam_score
# defteam_score
# score_differential
# posteam_score_post
# defteam_score_post
# score_differential_post

if __name__ == "__main__":
    main()
