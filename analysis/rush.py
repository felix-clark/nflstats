#!/usr/bin/env python3
from typing import Optional, Iterable
import pandas as pd

from db import plays
from utility import drop_noinfo


def _get_data(columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    get and cache test data
    """
    cache_name = "running_plays.parquet"
    try:
        play_data: pd.DataFrame = pd.read_parquet(cache_name)
        return play_data
    except Exception:
        print("Could not find cache")
    if columns and (
        "game_id" not in columns
        or "play_id" not in columns
        or "play_type" not in columns
    ):
        raise RuntimeError("Require game_id and play_id and play_type")
    play_data = plays(
        # The player IDs are a different format pre-2011
        # range(1999, 2019 + 1),
        range(2011, 2019 + 1),
        queries=[("play_type", "run")],
        columns=columns,
    )
    play_data.set_index(["game_id", "play_id"], inplace=True)
    play_data = drop_noinfo(play_data)
    play_data.sort_values(["game_id", "play_id"], inplace=True)
    play_data.to_parquet(cache_name)
    return play_data


def main():
    """
    Execute main analysis
    """
    # There may be more; this is just a first pass of what might be useful
    columns = [
        "posteam",
        "posteam_type",
        "defteam",
        "yardline_100",
        "sp",
        "down",
        "goal_to_go",
        "ydstogo",
        "ydsnet",  # unclear; net yards on drive?
        "desc",
        "yards_gained",
        "shotgun",
        "no_huddle",
        "qb_dropback",
        "qb_scramble",
        "run_location",
        "run_gap",
        "two_point_conv_result",
        "timeout",
        "ep",
        "epa",
        # There are several like these; can also use win probability [added] (wp/wpa)
        # 'total_home_epa',
        # 'total_away_epa',
        # 'total_home_rush_epa',
        # 'total_away_rush_epa',
        "wp",
        "def_wp",
        "wpa",
        "fumble_forced",
        "fumble_not_forced",
        "fumble_out_of_bounds",
        "safety",
        "penalty",
        "tackled_for_loss",
        "fumble_lost",
        # These are not quite identical; trick plays?
        "touchdown",
        "rush_touchdown",
        "two_point_attempt",
        "fumble",
        "drive_play_count",
        "drive_time_of_possession",
        "drive_first_downs",
        "passer",
        "rusher",
        "receiver",
        # about 5% of these plays have a pass?
        "pass",
        "rush",
        "first_down",
        "passer_id",
        "rusher_id",
        "receiver_id",
    ]
    columns = None

    rush_attempts = _get_data(columns)
    print(rush_attempts)
    for col in rush_attempts.columns:
        print(rush_attempts[col].describe())


if __name__ == "__main__":
    main()
