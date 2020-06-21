# For now define the data acquisition code in here. Split it up later.

import os
from collections import OrderedDict
from typing import Any, Callable, Iterable, Optional, Tuple, Union

import pandas as pd
import pyreadr

# TODO: save environment variable in .env to allow customization
# TODO: download from internet and cache if this doesn't exist
NFLFASTR_DATA_DIR: str = "/".join([os.getenv("HOME", "~"), "Data", "nflfastR-data"])


def _filter_years(
    data: pd.DataFrame, years: Optional[Union[int, Iterable[int]]], year_key: str
) -> pd.DataFrame:
    """
    Returns the subset of data with the season (defined by `year_key`) restricted by
    `years`, either a single value or a range.
    """
    if years is not None:
        if isinstance(years, Iterable):
            data = data.loc[data[year_key].isin(years)]
        else:
            data = data.loc[data[year_key] == years]
    return data


def roster(years: Optional[Union[int, Iterable[int]]] = None) -> pd.DataFrame:
    """
    Get roster data for a year or set of years
    """
    roster_data_file: str = "/".join(
        [NFLFASTR_DATA_DIR, "roster-data", "roster.csv.gz"]
    )
    data: pd.DataFrame = pd.read_csv(roster_data_file)
    data = _filter_years(data, years, "team.season")
    return data.reset_index(drop=True)


def schedule(year: int) -> pd.DataFrame:
    """
    Get the schedule for a given year
    """
    schedule_data_file: str = "/".join(
        [NFLFASTR_DATA_DIR, "schedules", f"sched_{year}.rds"]
    )
    r_data: OrderedDict = pyreadr.read_r(schedule_data_file)
    assert set(r_data.keys()) == set([None]), "Unexpected keys"
    data: pd.DataFrame = r_data[None]
    return data


def _year_plays(
    year: int, queries: Iterable[Tuple[str, Union[Any, Callable[[Any], bool]]]]
) -> pd.DataFrame:
    """
    Get the plays from a given year subject to a set of filters
    """
    # TODO: parquet instead of CSV?
    play_file: str = "/".join(
        # [NFLFASTR_DATA_DIR, "data", f"play_by_play_{year}.csv.gz"]
        [NFLFASTR_DATA_DIR, "data", f"play_by_play_{year}.parquet"]
    )
    # data: pd.DataFrame = pd.read_csv(play_file)
    data: pd.DataFrame = pd.read_parquet(play_file)
    for key, key_filter in queries:
        if callable(key_filter):
            data = data.loc[data[key].apply(key_filter)]
        else:
            data = data.loc[data[key] == key_filter]
    return data


def plays(
    years: Union[int, Iterable[int]],
    queries: Iterable[Tuple[str, Union[Any, Callable[[Any], bool]]]],
) -> pd.DataFrame:
    """
    Get play-by-play. The data can be filtered with (key, value) pairs where `value` is
    a value or a callable function that returns whether the value in the field should be
    kept.
    """
    data: pd.DataFrame = pd.DataFrame()
    if isinstance(years, Iterable):
        data = pd.concat([_year_plays(year, queries) for year in years])
    else:
        data = _year_plays(years, queries)
    return data.reset_index(drop=True)
