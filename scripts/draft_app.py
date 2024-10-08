#!/usr/bin/env python3
import argparse
import hashlib
import logging
import os
import pickle
import random
import sys
from cmd import Cmd
from datetime import datetime
from difflib import SequenceMatcher, get_close_matches
from itertools import chain, takewhile
from os import path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
from progressbar import progressbar
from metalogistic import MetaLogistic

from get_fantasy_points import get_points
from ruleset import bro_league, dude_league, nycfc_league, phys_league, ram_league
from tools import get_k_partition_boundaries, get_team_abbrev, rm_name_suffix

games_in_season = 17
# slap these bye/injury factors up here for now
# 13 games in regular FF season, but we're going to playoffs. we'll pretend they're
# independent. 18 weeks in season, 17 games played by each team, any reasonable league
# plays the first 17 so 16/17
bye_factor = (games_in_season - 1) / games_in_season
# this is the approximate fraction of the time that a player in
#  each position spends on the field uninjured.
# from sportinjurypredictor.net, based on average games missed assuming a 17 game season
# obviously rough, but captures trend and follows intuition
pos_injury_factor = {
    "QB": 0.94,
    "RB": 0.85,
    "WR": 0.89,
    "TE": 0.89,
    "DST": 1.0,
    "K": 1.0,
}
# From http://www.profootballlogic.com/articles/nfl-injury-rate-analysis/
# we can get an expected number of games played by position (out of 16).
# This was extracted from one year (2015) only.
# Kickers are not included.
pos_games_available = {
    "QB": 14.9 / bye_factor,
    "RB": 13.3 / bye_factor,
    "WR": 14.0 / bye_factor,
    "TE": 14.2 / bye_factor,
    "DST": 16.0 / bye_factor,
    # This kicker factor is made up.
    "K": 15.0 / bye_factor,
}
# TODO: use
# https://www.footballoutsiders.com/stat-analysis/2015/nfl-injuries-part-i-overall-view
# for a distribution from which to get a variance on these (use beta-binomial rather
# than binomial).


def single_team_sim(
    sim_ppg, sim_games, n_roster_per_team, replacement_baseline, flex_pos
):
    """
    returns the single-season points for a team given points per game and games played
    for every position. This function is not very efficient.
    """
    # We'll use an optimistic prediction to get it off the ground, and assume
    # that a manager can shuffle starters perfectly to fill all spots.
    main_positions = ["QB", "RB", "WR", "TE", "K", "DST"]
    flex_pos = flex_pos or ["RB", "WR", "TE"]
    games_pos: Dict[str, List[float]] = {pos: [] for pos in main_positions}
    for player, team, pos in sim_ppg.index:
        n_games = sim_games[(player, team, pos)]
        ppg = sim_ppg[(player, team, pos)]
        games_pos[pos].extend([ppg] * n_games)
    # extend the list of possible games with several at the baseline level
    # this is hacky and inelegant and should be changed
    for pos in games_pos:
        games_pos[pos].extend(
            [replacement_baseline[pos]] * games_in_season * n_roster_per_team[pos]
        )
        games_pos[pos].sort()
    total_points = 0
    # games_count = 0
    for pos in main_positions:
        n_games_needed = games_in_season * n_roster_per_team[pos]
        for _ in range(n_games_needed):
            total_points += (
                games_pos[pos].pop() if games_pos[pos] else replacement_baseline[pos]
            )
            # games_count += 1
    flex_games = sorted(chain(*[games_pos[pos] for pos in flex_pos]))
    if "FLEX" in n_roster_per_team:
        n_games_needed = games_in_season * n_roster_per_team["FLEX"]
        for _ in range(n_games_needed):
            total_points += (
                flex_games.pop()
                if flex_games
                else max(replacement_baseline["RB"], replacement_baseline["WR"])
            )
            # games_count += 1
    return total_points


def total_team_sims(
    sim_ppg, sim_games, n_roster_per_team, replacement_baseline, flex_pos
):
    """
    input: dataframes for starter and bench players with a number of simulated seasons
    position_baseline: assumed number of points that can be gotten for free
    returns a series of team scores taking into account positions
    """
    simulated_seasons = pd.Series(dtype=float)
    for sim_num in sim_ppg.columns:
        simulated_seasons.loc[sim_num] = single_team_sim(
            sim_ppg[sim_num],
            sim_games[sim_num],
            n_roster_per_team,
            replacement_baseline,
            flex_pos,
        )
    return simulated_seasons


def evaluate_roster(
    rosdf,
    n_roster_per_team,
    replacement_baselines,
    flex_pos=None,
    outfile=None,
    ppg_sims=None,
    games_sims=None,
):
    """
    applies projection for season points, with an approximation for bench value
    returns tuple of starter, bench value
    """
    numplayers = len(rosdf)
    numroster = sum([n_roster_per_team[pos] for pos in n_roster_per_team])
    if numplayers < numroster:
        print("This roster is not full.", file=outfile)
    if numplayers > numroster:
        print("This roster has too many players.", file=outfile)

    flex_pos = flex_pos or ["RB", "WR", "TE"]
    main_positions = ["QB", "RB", "WR", "TE", "K", "DST"]

    i_st = []  # the indices of the players we have counted so far
    for pos in main_positions:
        n_starters = n_roster_per_team[pos]
        rospos = rosdf[rosdf.index.get_level_values("pos") == pos].sort_values(
            "exp_proj", ascending=False
        )
        i_stpos = rospos.index[:n_starters]
        # val = (
        #     bye_factor
        #     * pos_injury_factor[pos]
        #     * rospos[rospos.index.isin(i_stpos)]["exp_proj"].sum()
        # )
        # starterval = starterval + val
        i_st.extend(i_stpos)

    n_flex = n_roster_per_team["FLEX"]
    rosflex = rosdf[
        (~rosdf.index.isin(i_st)) & (rosdf.index.get_level_values("pos").isin(flex_pos))
    ].sort_values("exp_proj", ascending=False)
    i_flex = rosflex.index[:n_flex]
    # starterval = starterval + rosflex[rosflex.index.isin(i_flex)]['exp_proj'].sum()
    i_st.extend(i_flex)

    drop_cols = ["adp", "ecp", "tier"]

    print("  starting lineup:", file=outfile)
    startdf = rosdf[rosdf.index.isin(i_st)].drop(drop_cols, axis=1, errors="ignore")
    print(startdf, file=outfile)

    benchdf = rosdf[~rosdf.index.isin(i_st)].drop(drop_cols, axis=1, errors="ignore")
    if not benchdf.empty:
        print("  bench:", file=outfile)
        print(benchdf, file=outfile)

    auctionval = rosdf["auction"].sum()

    simulated_seasons = None
    if ppg_sims is not None and games_sims is not None:
        # ros_idx = rosdf.drop(drop_cols, axis=1)[['player', 'team', 'pos']].values
        ros_idx = rosdf.index
        proj_ppg = ppg_sims.loc[ros_idx]
        proj_games = games_sims.loc[ros_idx]
        simulated_seasons = total_team_sims(
            proj_ppg,
            proj_games,
            n_roster_per_team,
            replacement_baseline=replacement_baselines,
            flex_pos=flex_pos,
        )

    # round values to whole numbers for josh, who doesn't like fractions :)
    print("approximate auction value:\t${:.2f}\n".format(auctionval), file=outfile)
    if simulated_seasons is not None:
        print("Simulation:\n", file=outfile)
        print(simulated_seasons.describe(), file=outfile)
        return simulated_seasons


def find_by_team(team, ap, pp):
    """
    prints players on the given team
    """
    available = ap[ap.index.get_level_values("team") == team.upper()]
    if len(available) > 0:
        print("Available players:")
        print(available)
    else:
        print("No available players found on team {}".format(team))
    picked = pp[pp.index.get_level_values("team") == team.upper()]
    if len(picked) > 0:
        print("Picked players:")
        print(picked)


def find_handcuff(index, ap, pp):
    """
    prints a list of players with the same team and position as the indexed player.
    ap: dataframe of available players
    pp: dataframe of picked players
    """
    # the "name" attribute is the index, so need dictionary syntax to grab actual name
    # name, pos, team = player['player'], player.pos, player.team
    name, team, pos = index
    print("Looking for handcuffs for {} ({}) - {}...\n".format(name, team, pos))
    ah = (
        ap[
            (ap.index.get_level_values("pos") == pos)
            & (ap.index.get_level_values("team") == team)
            & (ap.index.get_level_values("player") != name)
        ]
        if not ap.empty
        else ap
    )
    if not ah.empty:
        print("The following potential handcuffs are available:")
        # print(ah.drop(['volb'], axis=1))
        print(ah)
    ph = (
        pp[
            (pp.index.get_level_values("pos") == pos)
            & (pp.index.get_level_values("team") == team)
            & (pp.index.get_level_values("player") != name)
        ]
        if not pp.empty
        else pp
    )
    if not ph.empty:
        print("The following handcuffs have already been picked:")
        # print(ph.drop(['volb'], axis=1))
        print(ph)
    print()  # end on a newline


# adding features to search by team name/city/abbreviation might be nice,
#   but probably not worth the time for the additional usefulness.
#   It could also complicate the logic and create edge cases.
def find_player(search_str, ap, pp):
    """
    prints the players with one of the words in search_words in their name.
    useful for finding which index certain players are if they are not in the top when drafted.
    search_words: list of words to look for
    ap: dataframe of available players
    pp: dataframe of picked players
    """
    # clean periods, since they aren't consistent between sources
    search_str = search_str.replace(".", "")

    # check if any of the search words are in the full name
    def checkfunc(name: str) -> bool:
        exact_match = all(
            [
                sw in name.lower().replace(".", "")
                for sw in search_str.lower().split(" ")
            ]
        )
        if exact_match:
            return True
        approx_match = (
            SequenceMatcher(
                lambda c: c in "._ -", search_str.lower(), name.lower()
            ).ratio()
            > 0.6
        )
        return approx_match

    picked_players = pp.index.get_level_values("player")
    filt_mask = picked_players.map(checkfunc) if not pp.empty else None
    filtered_pp = pp[filt_mask] if not pp.empty else pp
    if not filtered_pp.empty:
        print("\n  Picked players:")
        print(filtered_pp)

    available_players = ap.index.get_level_values("player")
    checked_avail = available_players.map(checkfunc)

    filt_mask = checked_avail if not ap.empty else None
    filtered_ap = ap[filt_mask] if not ap.empty else ap
    if filtered_ap.empty:
        print("\n  Could not find any available players.")
    else:
        print("\n  Available players:")
        print(filtered_ap)


def get_player_values(
    ppg_df,
    games_df,
    n_roster_per_league,
    value_key="exp_proj",
    main_positions=None,
    flex_positions=None,
):
    """
    ppg_df: a dataframe with the expected points per game
    games_df: a dataframe with the games played
    n_roster_per_league: dictionary with number of positions required in each starting lineup
    returns the expected number of points for contributing starters
    """
    main_positions = main_positions or ["QB", "RB", "WR", "TE", "K", "DST"]
    flex_positions = flex_positions or ["RB", "WR", "TE"]

    assert (
        ppg_df.index == games_df.index
    ).all(), "PPG and games dataframes do not share indices"

    # gamesdf = df[['pos', value_key, 'g']].copy()
    # games = games_df[value_key]
    # ppg = ppg_df[value_key]
    gamesdf = pd.DataFrame(
        index=ppg_df.index, data={"ppg": ppg_df[value_key], "g": games_df[value_key]}
    )
    # gamesdf['ppg'] = gamesdf[value_key] / gamesdf['g']
    # The points per game must be in descending order
    gamesdf.sort_values("ppg", inplace=True, ascending=False)

    ppg_baseline = {}
    games_needed = {
        pos: (games_in_season * n_roster_per_league[pos]) for pos in main_positions
    }
    games_needed["FLEX"] = games_in_season * n_roster_per_league["FLEX"]

    for index, row in gamesdf.iterrows():
        _, _, pos = index
        games = row["g"]
        # pos, games = row[['pos', 'g']]
        gneeded = games_needed[pos]
        if gneeded > games:
            games_needed[pos] -= games
        elif 0 < gneeded <= games:
            games_needed[pos] = 0
            ppg_baseline[pos] = row["ppg"]
            if pos in flex_positions:
                gleft = games - gneeded
                gneeded = games_needed["FLEX"]
                if gneeded > gleft:
                    games_needed["FLEX"] -= gleft
                elif 0 < gneeded <= gleft:
                    games_needed["FLEX"] = 0
                    ppg_baseline["FLEX"] = row["ppg"]
        else:
            assert gneeded == 0
            if pos in flex_positions:
                gneeded = games_needed["FLEX"]
                if gneeded > games:
                    games_needed["FLEX"] -= games
                elif 0 < gneeded <= games:
                    games_needed["FLEX"] = 0
                    ppg_baseline["FLEX"] = row["ppg"]
                # if no games needed, we're done.
    del games_needed

    values = ppg_df[value_key].copy()
    for player in ppg_df.index:
        _, _, pos = player
        worst_starter_pg = ppg_baseline[pos]
        assert "FLEX" in ppg_baseline
        if pos in flex_positions:
            worst_starter_pg = min(worst_starter_pg, ppg_baseline["FLEX"])
        gs = games_df.loc[player, value_key]
        values.loc[player] = gs * (ppg_df.loc[player, value_key] - worst_starter_pg)
    return values


def get_auction_values(
    value_data, value_key, n_teams, n_roster_per_league, cap=200, min_bid=1
):
    """
    value_data: dataframe with columns of values, indexed by (player, team, position)
    value_key: column key to use as value
    """
    # these could be keyword arguments
    main_positions = ["QB", "RB", "WR", "TE", "K", "DST"]
    flex_positions = ["RB", "WR", "TE"]
    # manually devalue these due to their heavy dependence on weekly matchup
    crap_positions = ["K", "DST"]

    league_cap = n_teams * cap
    avail_cap = league_cap - min_bid * sum(n_roster_per_league.values())

    auction_values = value_data[[value_key]].copy()
    auction_values.loc[:, "auctionable"] = False
    auction_values.loc[:, "auction"] = 0

    # label positional (non-flex) starters as auctionable
    for pos in main_positions:
        # sort the players in each position so we can grab the top indices
        pos_start = auction_values.loc[
            auction_values.index.get_level_values("pos") == pos, value_key
        ].nlargest(n_roster_per_league[pos])
        auction_values.loc[pos_start.index, "auctionable"] = True
    flex_start = auction_values.loc[
        (auction_values.index.get_level_values("pos").isin(flex_positions))
        & (~auction_values["auctionable"]),
        value_key,
    ].nlargest(n_roster_per_league["FLEX"])

    auction_values.loc[flex_start.index, "auctionable"] = True

    # keep track of how many players are starter tier for each position
    n_starter_pos = {
        pos: len(
            auction_values.loc[
                auction_values["auctionable"]
                & (auction_values.index.get_level_values("pos") == pos)
            ].index
        )
        for pos in main_positions
    }

    # label next best set for the bench
    # NOTE: Should there be more positional dependence in here?
    bench_idx = (
        auction_values.loc[
            (~auction_values["auctionable"])
            & (~auction_values.index.get_level_values("pos").isin(crap_positions)),
            value_key,
        ]
        .nlargest(n_roster_per_league["BENCH"])
        .index
    )
    auction_values.loc[bench_idx, "auctionable"] = True

    auction_values.loc[auction_values["auctionable"], "auction"] = auction_values[
        value_key
    ].clip(lower=0)
    # manually de-value kickers and defense because of their matchup-dependence
    auction_values.loc[
        auction_values.index.get_level_values("pos").isin(crap_positions), "auction"
    ] = 0
    auction_values.loc[:, "auction"] *= avail_cap / auction_values["auction"].sum()

    auction_values.loc[auction_values["auctionable"], "auction"] += min_bid

    if not np.isclose(auction_values["auction"].sum(), league_cap):
        print(avail_cap)
        print(auction_values["auction"].sum())
        print(league_cap)
        logging.error("auction totals do not match league cap!")

    return auction_values["auction"]


def get_player_index(data, name, hide_stats=None):
    """
    returns the index (player, team, pos) of a player given their name and prompts if
    there are redundancies
    """
    hide_stats = hide_stats or []
    criterion = (
        data.index.get_level_values("player")
        .map(simplify_name)
        .str.contains(simplify_name(name))
    )
    filtered = data[criterion]
    if filtered.empty:
        logging.error("Could not find available player with name %s.", name)
        return
    if len(filtered) > 1:
        logging.info("Found multiple players:")
        print(filtered.drop(hide_stats, axis=1))
        filtered = prompt_for_unique(filtered)
        if filtered is None:
            return
    assert len(filtered) == 1, "Should only have one player filtered at this point."
    return filtered.index[0]


def load_player_list(outname):
    """loads the available and picked player data from the label \"outname\" """
    print("Loading with label {}.".format(outname))
    if path.isfile(outname + ".csv"):
        ap = pd.read_csv(outname + ".csv", index_col=["player", "team", "pos"])
    else:
        logging.error("Could not find file %s.csv!", outname)
    if path.isfile(outname + "_picked.csv"):
        pp = pd.read_csv(outname + "_picked.csv", index_col=["player", "team", "pos"])
    else:
        logging.error("Could not find file %s_picked.csv!", outname)
    return ap, pp


def _highlight(col):
    hl_max = ["vols", "volb", "vbsd", "auction", "vorp"]
    hl_min = ["adp", "ecp"]
    result = pd.Series("", index=col.index)
    if col.name == "g":
        result[col < col.max()] = "background-color: red"
    if col.name in hl_max:
        result[col == col.max()] = "background-color: yellow"
    if col.name in hl_min:
        result[col == col.min()] = "background-color: yellow"
    return result


def _textcol(row, stat="auction"):
    result = pd.Series("", index=row.index)
    return result


def pop_from_player_list(index, ap, pp=None, manager=None, pickno=None, price=None):
    """
    index: index of player to be removed from available
    """
    if index not in ap.index:
        raise IndexError(
            "The index ({}) does not indicate an available player!".format(index)
        )
    player = ap.loc[index]  # a dictionary of the entry
    # were using iloc, but the data may get re-organized so this should be safer
    if pp is not None:
        if index in pp.index:
            logging.error(
                "It seems like the index of the player is already in the picked player list."
            )
            logging.error("Someone needs to clean up the logic...")
            logging.debug("picked players w/index: %s", pp.loc[index])
            logging.debug("available players w/index: %s", ap.loc[index])
        pp.loc[index] = player
        if manager is not None:
            pp.loc[index, "manager"] = manager
            # this method of making the variable an integer is ugly and over time redundant.
            pp.manager = pp.manager.astype(int)
        if pickno is not None:
            pp.loc[index, "pick"] = pickno
            pp.pick = pp.pick.astype(int)
        # player = df.pop(index) # DataFrame.pop pops a column, not a row
        if price is not None:
            pp.loc[index, "price"] = price
    # name = player['player']
    # pos = player['pos']
    # team = player['team']
    # print('selecting {} ({}) - {}'.format(name, team, pos))
    print(f"selecting {index[0]} ({index[2]}) - {index[1]}")
    ap.drop(index, inplace=True)


def print_picked_players(pp, ap=None):
    """prints the players in dataframe df as if they have been selected"""
    npicked = pp.shape[0]
    if npicked == 0:
        print("No players have been picked yet.")
    else:
        with pd.option_context("display.max_rows", None):
            # TODO: we can probably still stand to improve this output:'
            drop_cols = ["manager", "pick", "volb", "tier"]
            print(pp.drop([col for col in drop_cols if col in pp], axis=1))
        if ap is not None:
            print("\nTypically picked at this point (by ADP):")
            adpsort = (
                pd.concat([pp, ap], sort=False)
                .sort_values("adp", ascending=True)
                .head(npicked)
            )
            for pos in ["QB", "RB", "WR", "TE", "K", "DST"]:
                print(
                    "{}:\t{}".format(
                        pos, len(adpsort[adpsort.index.get_level_values("pos") == pos])
                    )
                )
        print("\nPlayers picked by position:")
        # to_string() suppresses the last line w/ "name" and "dtype" output
        print(pp.index.get_level_values("pos").value_counts().to_string())


def print_teams(ap, pp):
    """
    prints a list of teams in both the available and picked player lists
    """
    teams = pd.concat([ap, pp], sort=False).index.unique(level="team")
    print(teams)


# this method will be our main output
def print_top_choices(
    df,
    ntop=10,
    npos=3,
    sort_key="value",
    sort_asc=False,
    drop_stats=None,
    hide_pos=None,
):
    if sort_key is not None:
        df.sort_values(sort_key, ascending=sort_asc, inplace=True)
    print("   DRAFT BOARD   ".center(pd.options.display.width, "*"))
    if drop_stats is None:
        drop_stats = []
    if hide_pos is None:
        hide_pos = []
    with pd.option_context("display.max_rows", None):
        print(
            df[~df.index.get_level_values("pos").isin(hide_pos)]
            .drop(drop_stats, inplace=False, axis=1)
            .head(ntop)
        )
    if npos > 0:
        positions = [
            pos for pos in ["QB", "RB", "WR", "TE", "K", "DST"] if pos not in hide_pos
        ]
        # can't figure out groupby right now -- might tidy up the output
        # print df[df.pos.isin(positions)].groupby('pos')# .agg({'exp_proj':sum}).nlargest(npos)
        for pos in positions:
            print(
                df[df.index.get_level_values("pos") == pos]
                .drop(drop_stats, inplace=False, axis=1)
                .head(npos)
            )


def print_top_position(
    df, pos, ntop=24, sort_key="value", sort_asc=False, drop_stats=None
):
    """prints the top `ntop` players in the position in dataframe df"""
    if sort_key is None:
        df.sort_index(ascending=sort_asc, inplace=True)
    else:
        df.sort_values(sort_key, ascending=sort_asc, inplace=True)
    if drop_stats is None:
        drop_stats = []
    # drop_cols = ['volb', 'tier']
    if pos.upper() == "FLEX":
        with pd.option_context("display.max_rows", None):
            print(
                df.loc[df.index.get_level_values("pos").isin(["RB", "WR", "TE"])]
                .drop(drop_stats, inplace=False, axis=1)
                .head(ntop)
            )
    else:
        with pd.option_context("display.max_rows", None):
            print(
                df[df.index.get_level_values("pos") == pos.upper()]
                .drop(drop_stats, inplace=False, axis=1)
                .head(ntop)
            )


def prompt_for_unique(players):
    """
    prompts the user to specify a selection.
    players should be indexed on (player, team, position)
    """
    indices = {str(num): index for num, (index, _) in enumerate(players.iterrows())}
    for num, index in indices.items():
        print(f"{num}: {index}")
    number = input("Select the intended player by index: ")
    if number not in indices:
        logging.error(f"{number} not found in the above list")
        return
    return players.loc[[indices[number]]]


def push_to_player_list(index, ap, pp):
    """
    index: index of player to be removed from available
    ap: dataframe of available players
    pp: dataframe of picked players
    """
    if index not in pp.index:
        raise IndexError(
            "The index ({}) does not indicate a picked player!".format(index)
        )
    player = pp.loc[index]
    if index in ap.index:
        print("The index of the picked player is already in the available player list.")
        print("Someone needs to clean up the logic...")
        print("DEBUG: picked players w/index:", pp.loc[index])
        print("DEBUG: available players w/index:", ap.loc[index])
    ap.loc[index] = player
    print(f"replacing {index[0]} ({index[2]}) - {index[1]}")
    pp.drop(index, inplace=True)


def save_player_list(outname, ap, pp=None):
    """saves the available and picked player sets with label "outname"."""
    print("Saving with label {}.".format(outname))
    ap.to_csv(outname + ".csv")
    if pp is not None:
        pp.to_csv(outname + "_picked.csv")


def simplify_name(name):
    """
    maps e.g. "A.J. Smith Jr." to "aj_smith_jr"
    """
    return name.strip().lower().replace(" ", "_").replace("'", "").replace(".", "")


def simulate_seasons(
    df: pd.DataFrame, n: int, hash: str, **kwargs: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate a number of seasons based on the expected, high, and low points.
    The number of games will be simulated as well as the points per game.
    Returns a tuple with a dataframe of simulated points per game and number of games.
    """
    index_cols = ["player", "team", "pos"]

    ppg_cache = kwargs.get("cache", f"simulation_cache_ppg_{hash}.csv")
    games_cache = kwargs.get("cache", f"simulation_cache_games_{hash}.csv")
    if path.isfile(ppg_cache) and path.isfile(games_cache):
        games_df = pd.read_csv(games_cache, index_col=index_cols)
        ppg_df = pd.read_csv(ppg_cache, index_col=index_cols)
        if len(games_df.columns) == len(ppg_df.columns) >= n:
            logging.info("Loading simulations from cache")
            return ppg_df, games_df
        else:
            logging.info("Simulation cache does not have sufficient iterations.")
    # seed based on randomness
    np.random.seed()

    # the "n" in the binomial drawing
    max_games = df["g"].to_numpy(dtype=int)
    # the "p" in the binomial drawing
    frac_games = np.vectorize(lambda pos: pos_games_available[pos] / games_in_season)(
        df.index.get_level_values("pos")
    )
    # compute the alpha and beta parameters for the beta-binomial distribution
    # Assume that alpha and beta are on the order of 1, which is close to the value
    # observed for most positions (1-2)
    alphas = frac_games * 1.5
    # we need an epsilon so that beta > 0
    betas = (1 - frac_games) * 1.5 + 1e-8

    logging.info("Simulating %s seasons...", n)
    sim_tags = [str(i_sim) for i_sim in range(n)]
    sim_games = pd.DataFrame(index=df.index, columns=sim_tags, dtype=int)
    sim_ppg = pd.DataFrame(index=df.index, columns=sim_tags, dtype=float)
    # TODO: instead of looping, can we use the size parameter in scipy?
    # This would generate a 2D array, which we'd have to put into columns
    # There are difficulties broadcasting over alpha/beta and size at the same time
    for n_sim in sim_tags:
        # the fraction is drawn from a beta distribution
        ps = st.beta.rvs(alphas, betas)
        games = st.binom.rvs(max_games, ps)
        sim_games[n_sim] = games
    x_fields = ["exp_proj_low", "exp_proj", "exp_proj_high"]
    # TODO: Reconsider this confidence interval. It's motivated by equally
    # distributing ~5 experts over the quantile, but this is a pretty arbitrary
    # choice.
    ps = [0.2, 0.5, 0.8]
    for idx, xlow, xmid, xhi in df[x_fields].itertuples(name=None):
        # Some of the data are incomplete on the edges. Add a little buffer to
        # make the distributions nice.
        xlow = min(xlow, xmid - 10.0)
        # xlow = min(xlow, max(xmid - 10., 0))
        xhi = max(xhi, xmid + 10.0)
        assert xlow <= xmid <= xhi

        points_dist = MetaLogistic(cdf_xs=[xlow, xmid, xhi], cdf_ps=ps)
        points = points_dist.rvs(size=n)
        # NOTE: This is terrible performance, do something better
        sim_ppg.loc[idx, sim_tags] = points
    # the imported projections assume that players will not miss time, so
    # divide by the max possible games.
    sim_ppg[sim_tags] /= np.expand_dims(max_games, axis=-1)
    sim_games.to_csv(games_cache)
    sim_ppg.to_csv(ppg_cache)
    return sim_ppg, sim_games


def verify_and_quit():
    user_verify = input("Are you sure you want to quit and lose all progress [y/N]? ")
    if user_verify.strip() == "y":
        print("Make sure you beat Russell.")
        exit(0)
    elif user_verify.lower().strip() == "n":
        print("OK then, will not quit after all.")
    else:
        print("Did not recognize confirmation. Will not quit.")


# note that this class can be used with tab-autocompletion...
# can we give it more words in its dictionary? (e.g. player names)
class MainPrompt(Cmd):
    """
    This is the main command loop for the program.
    The important data will mostly be copied into member variables so it has easy access.
    """

    # overriding default member variable
    prompt = " $$ "

    # These are the fields that should be saved and loaded
    save_fields = [
        "ap",
        "pp",
        "manager_names",
        "manager_picks",
        "draft_mode",
        "i_manager_turn",
        "user_manager",
        "n_teams",
        "n_roster_per_team",
    ]

    # member variables to have access to the player dataframes
    # TODO: consider combining into a single dataframe instead of moving players
    ap = pd.DataFrame()
    pp = pd.DataFrame()
    newsdf: Optional[pd.DataFrame] = None
    sim_ppg: Optional[pd.DataFrame] = None
    sim_games: Optional[pd.DataFrame] = None

    _sort_key = "value"
    _sort_asc = False

    flex_pos = ["RB", "WR", "TE"]

    hide_pos = ["K", "DST"]
    # hide_stats = ['tier', 'pos', 'volb']
    hide_stats = ["tier"]

    disabled_pos = ["K", "DST"]

    # _known_strategies = ['vols', 'vbsd', 'volb', 'vorp', 'adp', 'ecp']
    _known_strategies = ["value", "vorp", "adp", "ecp"]

    # member variables for DRAFT MODE !!!
    draft_mode = False
    i_manager_turn: Optional[int] = None
    # when initialized, looks like [1,2,3,...,11,12,12,11,10,...]
    manager_picks: List[int] = []
    user_manager: Optional[int] = None
    manager_names: Dict[int, str] = {}
    # managers can be set to automatically pick using a given strategy
    manager_auto_strats: Dict[int, str] = {}
    n_teams = 0
    n_roster_per_team: Dict[str, int] = {}

    # this is a member function we are overriding
    def emptyline(self):
        """
        do nothing when an empty line is entered.
        (without this definition, the last command is repeated)
        """
        pass

    def precmd(self, line):
        """
        this stub is run before every command is interpreted
        """
        self._update_vorp()
        # we need to return the line so that Cmd.onecmd() can process it
        # if we needed to, we would pre-process the input here
        return line

    def _advance_snake(self):
        """move up one step in the snake draft"""
        self.i_manager_turn = self.i_manager_turn + 1
        if self.i_manager_turn >= len(self.manager_picks):
            print("Draft is over!")
            conf = input("Are you done [y/N]? ")
            if conf != "y":
                print("Undoing last pick")
                push_to_player_list(self.pp.index[-1], self.ap, self.pp)
                # self._update_vorp()
                return self._regress_snake()
            # if we do this then we can't call "evaluate all". turning this off might
            # cause other bugs
            # self.draft_mode = False
            self.i_manager_turn = None
            self.manager_picks = []
            print("You're done! Type `evaluate all` to see a summary for each team.")
            self._set_prompt()
            return
        self._set_prompt()
        #####
        manager = self.manager_picks[self.i_manager_turn]
        if manager in self.manager_auto_strats:
            try:
                pickno = self.i_manager_turn + 1
                self._update_vorp()
                player_index = self._pick_rec(
                    manager, self.manager_auto_strats[manager]
                )
                pop_from_player_list(
                    player_index, self.ap, self.pp, manager=manager, pickno=pickno
                )
                self._advance_snake()
            except IndexError as err:
                logging.error(err)
                logging.error("could not pick player from list.")

    def _regress_snake(self):
        """move up one step in the snake draft"""
        self.i_manager_turn = self.i_manager_turn - 1
        self._set_prompt()

    def _get_current_manager(self):
        """returns number of current manager"""
        if not self.manager_picks or self.i_manager_turn is None:
            return None
        if self.i_manager_turn >= len(self.manager_picks):
            return None
        return self.manager_picks[self.i_manager_turn]

    def _get_manager_name(self, num=None):
        """
        returns name of manager number num
        if num is None then uses current manager
        """
        if num is None:
            num = self._get_current_manager()
        return (
            self.manager_names[num]
            if num in self.manager_names
            else "manager {}".format(num)
        )

    def _set_prompt(self):
        manno = self._get_current_manager()
        if manno is not None:
            managername = self._get_manager_name()
            if self.user_manager is not None and manno == self.user_manager:
                self.prompt = " s~({},{})s~  your pick! $$ ".format(
                    manno, self.i_manager_turn + 1
                )
            else:
                self.prompt = " s~({},{})s~  {}'s pick $$ ".format(
                    manno, self.i_manager_turn + 1, managername
                )
        else:
            self.prompt = " $$ "

    def _get_manager_roster(self, manager, pp=None):
        """returns dataframe of manager's roster"""
        # this will return a small copy w/ the manager index removed
        if pp is None:
            pp = self.pp
        if len(pp) == 0:
            return pp  # there isn't anything in here yet, and we haven't added the "manager" branch
        if "manager" in pp:
            return pp[pp.manager == manager].drop("manager", inplace=False, axis=1)
        return pp

    def _get_managers_til_next(self):
        """get list of managers before next turn"""
        # first we get the list of managers the will go before our next turn
        if not self.manager_picks or self.i_manager_turn is None:
            print('"managers til next" is only sensible in draft mode.')
            return None
        i_man: int = self.i_manager_turn
        current_team = self.manager_picks[i_man]
        comp_mans = []
        for man in self.manager_picks[i_man:]:
            if man not in comp_mans:
                comp_mans.append(man)
            else:
                break
        comp_mans.remove(current_team)  # don't include our own roster
        return comp_mans

    def _pick_rec(
        self,
        manager,
        strat="value",
        ap=None,
        pp=None,
        disabled_pos=None,
        vona_strat="adp",
    ):
        """
        picks the recommended player with the highest strat value
        returns the index of that player
        """
        # TODO: multiply by "need factor", based on how many of that position you have.
        # e.g. 0.8 once starters are full, 0.6 when already have 1 backup, 0.4 for 2, ...
        if ap is None:
            ap = self.ap
        if pp is None:
            pp = self.pp
        if disabled_pos is None:
            disabled_pos = []
        roster = self._get_manager_roster(manager, pp)
        # total_roster_spots = sum(
        #     [self.n_roster_per_team[pos] for pos in self.n_roster_per_team]
        # )
        n_roster_picked_pos = {
            pos: len(roster[roster.index.get_level_values("pos") == pos])
            for pos in self.n_roster_per_team
        }
        # if len(roster) >= total_roster_spots:
        #     manname = self._get_manager_name()
        #     print '{}\'s roster has no available spots left'.format(manname)

        starting_roster_spots = sum(
            [
                self.n_roster_per_team[pos]
                for pos in self.n_roster_per_team
                if pos.upper() != "BENCH"
            ]
        )
        crap_positions = ["K", "DST"]  # add DST when (or if) we bother
        # crap_starting_roster_spots = sum([self.n_roster_per_team[pos] for pos in crap_positions])
        needed_crap_starter_positions = [
            pos
            for pos in crap_positions
            if n_roster_picked_pos[pos] < self.n_roster_per_team[pos]
        ]
        # key_starting_roster_spots = starting_roster_spots - crap_starting_roster_spots

        key_positions = [
            "QB",
            "RB",
            "WR",
            "TE",
        ]  # this concept includes FLEX so don't count it
        # realistically "nonflex" will just be QBs but let's keep it flexible
        key_nonflex_positions = [
            pos for pos in key_positions if pos not in self.flex_pos
        ]
        needed_key_starter_positions = []
        needed_key_starter_positions.extend(
            [
                pos
                for pos in key_nonflex_positions
                if n_roster_picked_pos[pos] < self.n_roster_per_team[pos]
            ]
        )
        # print [len(roster[roster.pos == pos])
        #        > self.n_roster_per_team[pos] for pos in self.flex_pos]
        used_flex_spot = any(
            [
                n_roster_picked_pos[pos] > self.n_roster_per_team[pos]
                for pos in self.flex_pos
            ]
        )
        flex_mult = 0 if used_flex_spot else 1
        needed_key_starter_positions.extend(
            [
                pos
                for pos in self.flex_pos
                if n_roster_picked_pos[pos]
                < self.n_roster_per_team[pos]
                + flex_mult * self.n_roster_per_team["FLEX"]
            ]
        )
        # TODO: if picking for a flex spot, they should be evaluated by a
        # separate VOLS/VORP for FLEX (?) -- otherwise e.g. TEs get recommended
        # for flex too often

        current_roster_size = len(roster)
        acceptable_positions = []
        if needed_key_starter_positions:
            # if we still need key starters, make sure we grab these first
            acceptable_positions = needed_key_starter_positions
        elif (
            current_roster_size + len(needed_crap_starter_positions)
            >= starting_roster_spots
        ):
            # note: this logic will fail to fill crap positions if we're ever in
            # a situation where more than one of each is needed.
            # need to get a K/DST to fill the end of the lineup.
            acceptable_positions = needed_crap_starter_positions
        else:
            # once we have our starting lineup of important positions we can
            # pick for bench value and kickers.
            # vorp does a decent job of not picking kickers too quickly, but we
            # do need to keep it from taking more than one.
            acceptable_crap = [
                pos
                for pos in crap_positions
                if n_roster_picked_pos[pos] < self.n_roster_per_team[pos]
            ]
            # we allow backup players, but don't get more than half our bench with any one position
            acceptable_backup = [
                pos
                for pos in key_positions
                if n_roster_picked_pos[pos]
                < self.n_roster_per_team[pos] + self.n_roster_per_team["BENCH"] // 2
            ]
            acceptable_positions = acceptable_backup + acceptable_crap
        if strat == "vona":
            pos = self._get_max_vona_in(
                acceptable_positions, strat=vona_strat, disabled_pos=disabled_pos
            )
            if pos is None:
                # then the user probably has the next pick as well and we should just pick for value
                print(
                    "do you have the next pick? VONA is not well-defined. will return VOLS."
                )
                strat = "value"
            else:
                # vona_asc = vona_strat in ['adp', 'ecp']
                # topvonapos = ap[ap.pos == pos].sort_values(vona_strat, vona_asc)
                # take our projection over ADP/ECP.
                topvonapos = ap[ap.index.get_level_values("pos") == pos].sort_values(
                    "exp_proj", ascending=False
                )
                if len(topvonapos) <= 0:
                    print(
                        "error: could not get a list of availble position that maximizes VONA."
                    )
                    print("switch to regulat strat?")
                player_index = topvonapos.index[0]
                return player_index
        if strat == "vorp":
            self._update_vorp(
                ap, pp
            )  # just make sure we're using the right value, but probably too conservative
        # TODO: VONA-VORP is probably not updating VORP correctly for the propagated player lists
        acceptable_positions = [
            pos for pos in acceptable_positions if pos not in disabled_pos
        ]
        if len(acceptable_positions) <= 0:
            # if we've ruled out everything else, just pick one of the main positions
            acceptable_positions = key_positions
        asc = strat in ["adp", "ecp"]
        toppicks = ap[
            ap.index.get_level_values("pos").isin(acceptable_positions)
        ].sort_values(strat, ascending=asc)
        if len(toppicks) <= 0:
            print(
                "error: no available players in any position in {}".format(
                    acceptable_positions
                )
            )
        # player = topstart.iloc[0] # this is the player itself
        player_index = toppicks.index[0]
        return player_index

    def update_draft_html(self):
        # should make this separate function
        ntop = 32
        df = (
            self.ap[~self.ap.index.get_level_values("pos").isin(self.hide_pos)]
            .drop(self.hide_stats, inplace=False, axis=1)
            .sort_values("value", ascending=False)
            .head(ntop)
        )
        # some columns look better with custom formatting
        format_dict = {
            "exp_proj": "{:.1f}",
            "value": "{:.1f}",
            "vols": "{:.0f}",
            "vbsd": "{:.0f}",
            "vorp": "{:.0f}",
            "auction": "${:.0f}",
        }
        # right now, textcol() doesn't do anything
        sty = df.style.format(format_dict)
        # 'palegreen' is too light for a light palette, but it looks nice
        # get full list from matplotlib.colors.cnames
        cm = sns.light_palette("mediumseagreen", as_cmap=True)
        # TODO: diverging palettes for negative values ?
        hl_cols_rise = df.columns.isin(["exp_proj", "value", "auction", "vorp"])
        hl_cols_fall = df.columns.isin(["adp", "ecp"])
        sty = sty.background_gradient(cmap=cm, subset=hl_cols_rise, low=0.0)
        sty = sty.background_gradient(
            cmap=sns.light_palette("slateblue", as_cmap=True, reverse=True),
            subset=hl_cols_fall,
        )
        sty = sty.background_gradient(
            cmap=sns.light_palette("red", as_cmap=True, reverse=True),
            subset="g",
            high=0,
            low=1,
        )
        # sty = sty.apply(_highlight).apply(_textcol)
        if hl_cols_fall.any():
            sty = sty.highlight_min(subset=hl_cols_fall)
        sty = sty.highlight_max(subset=hl_cols_rise)
        # open and write the styled html
        with open("draft_board.html", "w") as f:
            sty.to_html(f)

    def _update_vorp(self, ap=None, pp=None):
        """
        updates the VORP values in the available players dataframe
        based on how many players in that position have been picked.
        a replacement for a 1-st round pick comes from the top of the bench,
        while a replacement for a bottom bench player comes from the waivers.
        """
        # should maybe cancel this.. it takes time to compute and we have lots
        # of thresholds now. It may be called more often than is necessary.
        if ap is None:
            ap = self.ap
        if pp is None:
            pp = self.pp

        positions = [
            pos
            for pos in list(self.n_roster_per_team.keys())
            if pos not in ["FLEX", "BENCH"]
        ]

        for pos in positions:
            # also implement: maximum players on roster
            # TODO: implement that max in the WAIV designation as well (elsewhere)
            # maximum probably won't matter for most drafts, so de-prioritize it
            # while your draft is in an hour :E

            pos_picked = (
                pp[pp.index.get_level_values("pos") == pos] if not pp.empty else pp
            )
            n_pos_picked = len(pos_picked.index)
            n_waiv_picked = len(pos_picked[pos_picked.tier == "FA"].index)
            # if any managers picked waiver-tier players, then we can shed
            #  the next-worst bench player from our calculations
            # we can still shed all WAIV players since this case raises the value of the threshold
            posdf = ap[ap.index.get_level_values("pos") == pos] if not ap.empty else ap
            # pos_draftable = self.ap[(self.ap.pos == pos) & (self.ap.tier != 'FA')]
            pos_draftable = posdf[posdf.tier != "FA"]
            n_pos_draftable = len(pos_draftable.index) - n_waiv_picked
            vorp_baseline = 0
            if n_pos_draftable <= 0:
                # no more "draftable" players -- vorp should be zero for top
                vorp_baseline = ap[ap.index.get_level_values("pos") == pos][
                    "value"
                ].max()
            else:
                frac_through_bench = (
                    n_pos_picked * 1.0 / (n_pos_picked + n_pos_draftable)
                )
                backup_mask = pos_draftable["tier"] == "BU"
                # we also need to include the worst starter in our list to make it agree
                # with VOLS before any picks are made
                worst_starters = pos_draftable[~backup_mask].sort_values(
                    "value", ascending=True
                )
                ls_index = None  # index of worst starter in position
                # fw_index = None # index of best wavier option in position (assuming ADP)
                if not worst_starters.empty:
                    ls_index = worst_starters.index[0]
                # if len(best_waivers) > 0:
                #     fw_index = best_waivers.index[0]
                pos_baseline = pos_draftable.loc[[ls_index]]
                n_pos_baseline = len(pos_baseline.index)
                if n_pos_baseline == 0:
                    # this can happen, e.g. with kickers who have no "backup" tier players
                    ap.loc[ap.index.get_level_values("pos") == pos, "vorp"] = ap[
                        "value"
                    ]
                    continue
                index = int(frac_through_bench * n_pos_baseline)
                if index >= len(pos_baseline):
                    print("warning: check index here later")
                    index = len(pos_baseline - 1)
                vorp_baseline = (
                    pos_baseline["value"].sort_values(ascending=False).iloc[index]
                )
            ap.loc[ap.index.get_level_values("pos") == pos, "vorp"] = (
                ap["value"] - vorp_baseline
            )

    def do_test(self, _):
        """
        A placeholder function to let us quickly test features
        """
        players_a, players_b = [], []
        main_positions = ["QB", "RB", "WR", "TE", "K", "DST"]
        n_draft = {pos: self.n_roster_per_team[pos] for pos in main_positions}
        n_draft["RB"] += (
            self.n_roster_per_team["FLEX"] + self.n_roster_per_team["BENCH"] // 2
        )
        n_draft["WR"] += (
            self.n_roster_per_team["BENCH"] - self.n_roster_per_team["BENCH"] // 2
        )
        for pos in main_positions:
            players_a.append(
                self.ap[self.ap.index.get_level_values("pos") == pos]
                .head(24)
                .sample(n=n_draft[pos])
            )
            players_b.append(
                self.ap[self.ap.index.get_level_values("pos") == pos]
                .head(24)
                .sample(n=n_draft[pos])
            )
        roster_a = pd.concat(players_a)
        _roster_b = pd.concat(players_b)
        # print(roster_a)
        # print(roster_b)
        baseline = {"QB": 20, "RB": 10, "WR": 10, "TE": 7, "K": 5, "DST": 6}
        evaluate_roster(
            roster_a,
            self.n_roster_per_team,
            replacement_baselines=baseline,
            flex_pos=self.flex_pos,
            ppg_sims=self.sim_ppg,
            games_sims=self.sim_games,
        )

    def do_auction(self, _):
        """
        right now, this just spits out some auction information
        """

        def print_gb(grouped):
            for pi in grouped.items():
                print("{}\t{:.2f}".format(*pi))
            print()

        print("  available value per team:")
        avail_pos_val = self.ap.groupby("pos")["auction"].sum()
        print_gb(avail_pos_val / self.n_teams)
        if self.pp.shape[0] > 0:
            print("\n  picked value per team:")
            print_gb(self.pp.groupby("pos")["auction"].sum() / self.n_teams)
            if "price" in self.pp:
                print("\n  average price per team:")
                picked_pos_price = self.pp.groupby("pos")["price"].sum()
                print_gb(picked_pos_price / self.n_teams)
                # TODO: make max budget configurable
                total_budget = self.n_teams * 200
                remaining_budget = total_budget - self.pp.price.sum()
                print(
                    "\n  inflation (total = {:.2f}):".format(
                        remaining_budget / self.ap.auction.sum()
                    )
                )
                allp = pd.concat((self.ap, self.pp), sort=False)
                infl_pos = (
                    allp.groupby("pos")["auction"]
                    .sum()
                    .subtract(picked_pos_price, fill_value=0.0)
                    / avail_pos_val
                )
                print_gb(infl_pos)

    def do_disable_pos(self, args):
        """
        disable positions from recommended picks
        """
        for pos in args.split(" "):
            upos = pos.upper()
            if upos in self.disabled_pos:
                print("{} is already disabled.".format(upos))
                continue
            if upos in ["QB", "RB", "WR", "TE", "K", "DST"]:
                print("Disabling {} in recommendations.".format(upos))
                self.disabled_pos.append(upos)

    def complete_disable_pos(self, text, line, begidk, endidx):
        """implements auto-complete for disable_pos function"""
        all_pos = ["QB", "RB", "WR", "TE", "K", "DST"]
        avail_disable = [pos for pos in all_pos if pos not in self.disabled_pos]
        if text:
            return [name for name in avail_disable if name.startswith(text.upper())]
        else:
            return avail_disable

    def do_enable_pos(self, args):
        """
        enable positions from recommended picks
        """
        for pos in args.split(" "):
            upos = pos.upper()
            if upos in self.disabled_pos:
                print("Enabling {} in recommendations.".format(upos))
                self.disabled_pos.remove(upos)

    def complete_enable_pos(self, text, line, begidk, endidx):
        """implements auto-complete for enable_pos function"""
        if text:
            return [name for name in self.disabled_pos if name.startswith(text.upper())]
        else:
            return self.disabled_pos

    def do_evaluate(self, args):
        """
        usage: evaluate [MAN]...
        evaluate one or more rosters
        if no argument is provided, the current manager's roster is shown
        type `evaluate all` to evaluate rosters of all managers
        """
        outfile = open("draft_evaluation.txt", "w")

        # calculate the replacement baselines
        main_positions = ["QB", "RB", "WR", "TE", "K", "DST"]
        # get approximate replacement baselines by taking the mean of the top few undrafted players
        replacement_baselines = {
            pos: self.ap[self.ap.index.get_level_values("pos") == pos]["exp_proj"]
            .sort_values(ascending=False)
            .head(self.n_teams)
            .mean()
            / games_in_season
            for pos in main_positions
        }
        if "manager" not in self.pp:
            # we aren't in a draft so just evaluate the full group of players
            print("roster from selected players:", file=outfile)
            evaluate_roster(
                self.pp,
                self.n_roster_per_team,
                replacement_baselines=replacement_baselines,
                flex_pos=self.flex_pos,
                outfile=outfile,
                ppg_sims=self.sim_ppg,
                games_sims=self.sim_games,
            )
            return
        indices = []
        if not args:
            indices = [self._get_current_manager()]
        elif args.lower() == "all":
            indices = list(range(1, self.n_teams + 1))
        else:
            try:
                indices = [int(a) for a in args.split(" ")]
            except ValueError as err:
                logging.error("Could not interpret managers to evaluate.")
                logging.error(err)
        # manager_vals = {}
        manager_sims = {}
        for i in indices:
            manager_name = self._get_manager_name(i)
            print(f"{manager_name}'s roster:", file=outfile)
            evaluation = evaluate_roster(
                self._get_manager_roster(i),
                self.n_roster_per_team,
                replacement_baselines=replacement_baselines,
                flex_pos=self.flex_pos,
                outfile=outfile,
                ppg_sims=self.sim_ppg,
                games_sims=self.sim_games,
            )
            if evaluation is not None:
                manager_sims[manager_name] = evaluation

        if manager_sims:
            simdf = pd.DataFrame(manager_sims)
            # for key, value in manager_sims.items():
            #     simdf.iloc[key] = value
            rankdf = simdf.rank(axis=1, ascending=False)
            # print(simdf)
            # print(rankdf)
            print("\nRankings:", file=outfile)
            print(rankdf.describe(), file=outfile)
            frac_top = (rankdf == 1).sum(axis=0) / len(rankdf)
            print("\nFraction best:", file=outfile)
            print(frac_top, file=outfile)
            n_top = min(6, len(manager_sims) // 2)
            frac_top_n = (rankdf <= n_top).sum(axis=0) / len(rankdf)
            print(f"\nFraction top {n_top}:", file=outfile)
            print(frac_top_n, file=outfile)
            frac_bot = (rankdf == rankdf.max()).sum(axis=0) / len(rankdf)
            print("\nFraction worst:", file=outfile)
            print(frac_bot, file=outfile)

            # assign tiers based on expected value
            if len(indices) > 3:
                # cluster into k tiers
                k = int(np.ceil(np.sqrt(1 + 2 * len(indices))))
                manager_vals = simdf.mean(axis=0)
                totvals = manager_vals.to_numpy()
                partitions = get_k_partition_boundaries(totvals, k - 1)[::-1]
                tier = 0
                sorted_manager_vals = sorted(
                    list(manager_vals.items()), key=lambda tup: tup[1], reverse=True
                )
                while len(sorted_manager_vals) > 0:
                    tier = tier + 1
                    print("Tier {}:".format(tier), file=outfile)
                    part_bound = partitions[0] if len(partitions) > 0 else -np.inf
                    tiermans = [
                        y
                        for y in takewhile(
                            lambda x: x[1] > part_bound, sorted_manager_vals
                        )
                    ]
                    for manager, manval in tiermans:
                        print(f"  {manager}: \t{int(manval)}", file=outfile)
                    print("\n", file=outfile)
                    sorted_manager_vals = sorted_manager_vals[len(tiermans):]
                    partitions = partitions[1:]
        if outfile is not None:
            print(f"evaltuation saved to {outfile.name}.")
            outfile.close()

    def do_exit(self, args):
        """alias for `quit`"""
        self.do_quit(args)

    def do_find(self, args):
        """
        usage: find NAME...
        finds and prints players with the string(s) NAME in their name.
        """
        # search_words = [word for word in args.replace('_', ' ').split(' ') if word]
        search_str = args.replace("_", " ")
        find_player(search_str, self.ap, self.pp)

    def complete_find(self, text, line, begidk, endidx):
        """implements auto-complete for player names"""
        avail_names = pd.concat([self.ap, self.pp], sort=False).index.unique(
            level="player"
        )
        mod_avail_names = [simplify_name(name) for name in avail_names]
        if text:
            return [name for name in mod_avail_names if name.startswith(text.lower())]
        return mod_avail_names

    def do_handcuff(self, args):
        """
        usage: handcuff I...
        find potential handcuffs for player with index(ces) I
        """
        index = None
        all_players = pd.concat([self.ap, self.pp], sort=False)
        index = get_player_index(all_players, args)
        find_handcuff(index, self.ap, self.pp)

    def complete_handcuff(self, text, line, begidk, endidx):
        """implements auto-complete for player names"""
        # avail_names = pd.concat([self.ap.index.get_level_values('player'),
        # self.pp.index.get_level_values('player')], sort=False).map(simplify_name)
        avail_names = [
            simplify_name(name) for name in self.ap.index.get_level_values("player")
        ]
        avail_names.extend(
            [simplify_name(name) for name in self.pp.index.get_level_values("player")]
        )
        if text:
            return [
                name for name in avail_names if name.startswith(simplify_name(text))
            ]
        return avail_names

    def do_hide(self, args):
        """
        usage: hide ITEM...
        hide a position or statistic from view
        """
        for arg in args.split(" "):
            if arg.lower() in self.ap:
                if arg.lower() not in self.hide_stats:
                    print("Hiding {}.".format(arg.lower()))
                    self.hide_stats.append(arg.lower())
                else:
                    print("{} is already hidden.".format(arg))
            elif arg.upper() in ["QB", "RB", "WR", "TE", "K", "DST"]:
                print("Hiding {}s.".format(arg.upper()))
                self.hide_pos.append(arg.upper())
            else:
                print("Could not interpret command to hide {}.".format(arg))
                print("Available options are in:")
                print("QB, RB, WR, TE, K, DST")
                print(self.ap.columns)

    def complete_hide(self, text, line, begidk, endidx):
        """implements auto-complete for hide function"""
        all_pos = ["QB", "RB", "WR", "TE", "K", "DST"]
        avail_hide_pos = [pos for pos in all_pos if pos not in self.hide_pos]
        avail_hide_stat = [
            stat for stat in self.ap.columns if stat not in self.hide_stats
        ]
        avail_hides = avail_hide_pos + avail_hide_stat
        if text:
            return [
                name.lower() for name in avail_hides if name.startswith(text.lower())
            ]
        return [name.lower() for name in avail_hides]

    def do_info(self, args):
        """print full data and news about player"""
        # This only checks
        player_index = get_player_index(self.ap, args)
        pl = None
        if player_index is not None:
            pl = self.ap.loc[player_index]
        else:
            logging.info("No players found in available pool. Checking picked.")
            player_index = get_player_index(self.pp, args)
            if player_index is not None:
                pl = self.pp.loc[player_index]
        if pl is None:
            logging.info("No players found in either pool.")
            return
        player, team, pos = player_index
        print()
        print(f"{player} ({pos}) - {team}:")
        for data in pl.items():
            out = (
                "{}:      \t{:.4f}" if type(data[1]) is np.float64 else "{}:      \t{}"
            )
            print(out.format(*data))
        if pl["n"] == "*" and self.newsdf is not None:
            # then there is a news story that we need the details of
            newsnames = self.newsdf.player.map(simplify_name)
            pix = newsnames.isin(
                get_close_matches(simplify_name(player), newsnames.values)
            )
            if newsnames[pix].shape[0] != 1:
                logging.warning("did not unambiguously identify news item:")
                print(newsnames[pix])
                pix = (
                    newsnames
                    == get_close_matches(simplify_name(pl.player), newsnames.values)[0]
                )
            for _, nrow in self.newsdf[pix].iterrows():
                print("\n  {}: {}".format(nrow.player, nrow.details))
        print()

    def complete_info(self, text, line, begidk, endidx):
        """implements auto-complete for player names"""
        names = [
            simplify_name(name) for name in self.ap.index.get_level_values("player")
        ]
        names.extend(
            [simplify_name(name) for name in self.pp.index.get_level_values("player")]
        )
        return (
            [name for name in names if name.startswith(text.lower())] if text else names
        )

    def do_list(self, args):
        """alias for `ls`"""
        self.do_ls(args)

    def do_load(self, args):
        """
        usage load [OUTPUT]
        loads player lists from OUTPUT.csv (default OUTPUT is draft_players)
        """
        outname = args if args else "draft_backup"
        # self.ap, self.pp = load_player_list(outname)
        picklename = f"{outname}.state.p"
        if path.isfile(picklename):
            logging.info("Loading state from %s", picklename)
        else:
            logging.error("File %s does not exist.", picklename)
            return
        with open(picklename, "rb") as pfile:
            save_data = pickle.load(pfile)
            for key in self.save_fields:
                setattr(self, key, save_data.pop(key))
            if save_data:
                logging.error("There is unused data in the loaded state:")
                print(save_data)
        self._set_prompt()

    # TODO: add completion for loads based on files matching '*.state.p'

    def do_ls(self, args):
        """
        usage: ls [N [M]]
        prints N top available players and M top players at each position
        """
        spl_args = [w for w in args.split(" ") if w]
        ntop, npos = 16, 3
        try:
            if spl_args:
                ntop = int(spl_args[0])
            if spl_args[1:]:
                npos = int(spl_args[1])
        except ValueError as err:
            logging.error("`ls` requires integer arguments.")
            logging.error(err)
        print_top_choices(
            self.ap,
            ntop,
            npos,
            self._sort_key,
            self._sort_asc,
            self.hide_stats,
            self.hide_pos,
        )

    def do_lspick(self, args):
        """prints summary of players that have already been picked"""
        # we already have the `roster` command to look at a roster of a manager;
        # TODO: let optional argument select by e.g. position?
        print_picked_players(self.pp, self.ap)

    def do_lspos(self, args):
        """
        usage: lspos POS [N]
        prints N top available players at POS where POS is one of (qb|rb|wr|te|flex|k)
        """
        spl_args = [w for w in args.split(" ") if w]
        if not spl_args:
            print("Provide a position (qb|rb|wr|te|flex|k) to the `lspos` command.")
            return
        pos = spl_args[0]
        ntop = 16
        if spl_args[1:]:
            try:
                ntop = int(spl_args[1])
            except ValueError:
                print("`lspos` requires an integer second argument.")
        print_top_position(
            self.ap, pos, ntop, self._sort_key, self._sort_asc, self.hide_stats
        )

    def do_mv(self, args):
        """
        Move a player to another team.
        usage: move <player> <manager name>
        """
        line_split = args.split(" ")
        if len(line_split) != 2:
            logging.error("usage: move <player> <manager name>")
            logging.info("moves the picked player to the given manager's team.")
            return
        player_name, manager_name = line_split
        target_managers = get_close_matches(
            manager_name, self.manager_names.values(), n=1
        )
        if not target_managers:
            logging.error("No manager matches %s", manager_name)
            return
        target_manager = target_managers[0]
        valid_managers = self.manager_names.values()
        if target_manager not in valid_managers:
            logging.error(f"Manager must be one of {valid_managers}")
            return
        player_index = get_player_index(self.pp, player_name)
        manager_index = list(self.manager_names)[
            list(self.manager_names.values()).index(manager_name)
        ]
        self.pp.loc[player_index, "manager"] = manager_index

    def complete_mv(self, text, line, begidk, endidx):
        """
        implements completion on picked players and manager names.
        """
        # begidx and endidx are supposed to be useful when the completion
        # depends on the position, but they don't appear to be defined.
        line_words = line.split(" ")[1:]
        completions = []
        if len(line_words) == 1:
            picked_names = self.pp.index.get_level_values("player").map(simplify_name)
            completions = [name for name in picked_names if name.startswith(text)]
        elif len(line_words) == 2:
            completions = [
                name
                for name in self.manager_names.values()
                if simplify_name(name).startswith(text.lower())
            ]
        return completions

    def do_name(self, args):
        """
        name [N] <manager name>
        names the current manager if first argument is not an integer
        """
        if not args:
            logging.error("usage: name [N] <manager name>")
            logging.info(
                "names the current manager if first argument is not an integer"
            )
            return
        mannum = self._get_current_manager()
        splitargs = args.split(" ")
        firstarg = splitargs[0]
        manname = args
        try:
            mannum = int(firstarg)
            manname = " ".join(splitargs[1:])
        except ValueError as e:
            if mannum is None:
                print(e)
                print("Could not figure out a valid manager to assign name to.")
                print("If not in draft mode, enter a number manually.")
        self.manager_names[mannum] = manname
        self._set_prompt()

    def do_next_managers(self, _):
        """
        usage: next_managers
        prints the remaining open starting lineup positions of the managers
        that will have (two) picks before this one's next turn.
        """
        if not self.draft_mode:
            print("this command is only available in draft mode.")
            return
        comp_mans = self._get_managers_til_next()

        # here we loop through the managers and see how many starting spots they have
        starting_pos = [
            pos
            for pos, numpos in list(self.n_roster_per_team.items())
            if numpos > 0 and pos not in ["FLEX", "BENCH"]
        ]
        pos_totals = {key: 0 for key in starting_pos}
        pos_totals["FLEX"] = 0
        for man in comp_mans:
            print("{} needs starters at:".format(self._get_manager_name(man)))
            roster = self._get_manager_roster(man)
            for pos in starting_pos:
                hasleft = self.n_roster_per_team[pos] - len(roster[roster.pos == pos])
                if hasleft > 0:
                    print("{}: {}".format(pos, hasleft))
                    pos_totals[pos] = pos_totals[pos] + hasleft
            flexused = sum(
                [
                    max(0, len(roster[roster.pos == pos]) - self.n_roster_per_team[pos])
                    for pos in self.flex_pos
                ]
            )
            flexleft = self.n_roster_per_team["FLEX"] - flexused
            if flexleft > 0:
                print("FLEX: {}".format(flexleft))
                pos_totals["FLEX"] = pos_totals["FLEX"] + flexleft
        if sum(pos_totals.values()) > 0:
            print(
                "\ntotal open starting roster spots ({} picks):".format(
                    2 * len(comp_mans)
                )
            )
            for pos, n in list(pos_totals.items()):
                if n > 0:
                    print("{}: {}".format(pos, n))

    def do_pick(self, args):
        """
        usage: pick <player>
               pick <strategy>  (in snake draft mode)
               pick <strategy> auto
               pick <player> price (to save auction price)
        remove player with index or name from available player list
        in snake draft mode, `pick vols` can be used to pick the VOLS recommended player.
        if "auto" is provided then this manager will automatically pick following this strategy
        """
        # try to strip the price off, if there are multiple arguments and the last one is a number
        price = None
        argl = args.lower().split(" ")
        if len(argl) > 1:
            try:
                price = int(argl[-1])
                argl.pop(-1)
            except ValueError:
                try:
                    price = float(argl[-1])
                    argl.pop(-1)
                except ValueError:
                    pass
        if price is None and "price" in self.pp:
            logging.warning("price field detected but none is provided.")
            logging.warning("usage:")
            logging.warning("pick <player> <price>")
            return

        manager = self._get_current_manager()
        index = None
        if manager is not None:
            if argl and argl[0] in self._known_strategies:
                index = self._pick_rec(manager, argl[0])
            if len(argl) > 1 and argl[1] == "auto":
                self.manager_auto_strats[manager] = argl[0]
        elif argl[0] in self._known_strategies:
            print("Must be in draft mode to set an automatic strategy.")

        args = " ".join(argl)  # remaining args after possibly popping off price
        # TODO: assign a player to a manager in auction
        if index is None:
            index = get_player_index(self.ap, args)
        try:
            pickno = self.i_manager_turn + 1 if self.draft_mode else None
            pop_from_player_list(
                index, self.ap, self.pp, manager=manager, pickno=pickno, price=price
            )
            self._update_vorp()
            if self.draft_mode:
                self._advance_snake()
        except IndexError as e:
            print(e)
            print("could not pick player from list.")
        self.update_draft_html()

    def complete_pick(self, text, line, begidk, endidx):
        """implements auto-complete for player names"""
        avail_names = self.ap.index.get_level_values("player")
        # TODO: make it look a bit prettier by allowing spaces instead of underscores.
        # see:
        # https://stackoverflow.com/questions/4001708/change-how-python-cmd-module-handles-autocompletion
        # clean up the list a bit, removing ' characters and replacing spaces with underscores
        mod_avail_names = [simplify_name(name) for name in avail_names]
        # TODO: allow another argument for manager names and complete based on available
        if text:
            return [name for name in mod_avail_names if name.startswith(text.lower())]
        else:
            return [name for name in mod_avail_names]

    def do_plot(self, args):
        """
        plot the dropoff of a quantity by position
        """
        # reverse the sort order for better colors on the important positions
        main_positions = sorted(["QB", "RB", "WR", "TE", "K", "DST"], reverse=True)
        yquant = args.strip().lower() if args else self._sort_key
        if yquant not in self.ap:
            logging.error(f"{yquant} is not a quantity that can be plotted.")
            return
        plot_cols = [yquant]
        for err_cols in set(["err_high", "err_low"]) & set(self.ap.columns):
            plot_cols.append(err_cols)
        plotdf = self.ap[self.ap.tier != "FA"][plot_cols]
        plotdf.sort_values("pos", inplace=True, ascending=False)
        plotdf["pos"] = plotdf.index.get_level_values("pos")
        for pos in main_positions:
            pos_idxs = (
                plotdf[plotdf.index.get_level_values("pos") == pos]
                .sort_values(yquant, ascending=False)
                .index
            )
            for rank, idx in enumerate(pos_idxs):
                plotdf.loc[idx, "posrank"] = rank
        g = sns.pointplot(
            data=plotdf,
            x="posrank",
            y=yquant,
            hue="pos",
            # this style looks decent, but I prefer lines connecting these points
            # because they're sorted
            # , kind='strip'
        )
        # plot error bands if they exist
        # TODO: re-scale for auction and any other variables
        # print(dir(g))
        if "err_high" in plotdf and "err_low" in plotdf.columns:
            for pos in main_positions:
                pos_data = plotdf[
                    plotdf.index.get_level_values("pos") == pos
                ].sort_values(yquant, ascending=False)
                g.fill_between(
                    x=pos_data["posrank"],
                    y1=pos_data[yquant] - pos_data["err_low"],
                    y2=pos_data[yquant] + pos_data["err_high"],
                    alpha=0.25,
                )
        # else:
        g.set_xlabel("Position rank")
        g.set_ylabel(yquant)
        g.set_xticklabels(
            []
        )  # by default the tick labels are drawn as floats, making them hard to read
        plt.show()

    def do_q(self, args):
        """alias for `quit`"""
        self.do_quit(args)

    def do_quit(self, _):
        """
        exit the program
        """
        verify_and_quit()

    def do_recommend(self, args):
        """print recommendations"""
        manager = self._get_current_manager()
        for strat in self._known_strategies:
            player, team, pos = self._pick_rec(
                manager, strat, disabled_pos=self.disabled_pos
            )
            print(f" {strat.upper()} recommended:\t  {player} ({pos}) - {team}")
        if self.manager_picks:
            # should we put auction in here?
            vona_strats = ["value", "adp", "ecp"]
            # vona-vorp takes too long
            for strat in vona_strats:
                player, team, pos = self._pick_rec(
                    manager,
                    strat="vona",
                    vona_strat=strat,
                    disabled_pos=self.disabled_pos,
                )
                # player = self.ap.loc[pick]
                print(
                    f" VONA-{strat.upper()} recommended:\t  {player} ({pos}) - {team}"
                )

    def do_roster(self, args):
        """
        usage: roster [N]...
               roster all
        prints the roster of the current manager so far
        can take a number or series of numbers to print only those manager's
        if "all" is passed then it will output all rosters
        """
        # this function is pretty redundant with the `evaluate` command, which give more
        # detailed information (e.g. breaking up starters and bench)
        if not self.draft_mode:
            print("The `roster` command is only available in draft mode.")
            return
        if args.lower() == "all":
            for i_man in range(1, 1 + self.n_teams):
                manname = self._get_manager_name(i_man)
                print("\n {}:".format(manname))
                theroster = self._get_manager_roster(i_man)
                if len(theroster) > 0:
                    print(theroster.drop(self.hide_stats, axis=1))
                else:
                    print("No players on this team yet.\n")
            print()
            return
        if not args:
            print("\n {}:".format(self._get_manager_name()))
            theroster = self._get_manager_roster(self._get_current_manager())
            if not theroster.empty:
                print(theroster.drop(self.hide_stats, axis=1))
                print()
            else:
                print("No players on this team yet.\n")
            return
        indices = []
        try:
            indices = [int(i) for i in args.split(" ")]
        except ValueError as e:
            print("`roster` requires integer arguments")
            print(e)
            return
        for i in indices:
            manname = self._get_manager_name(i)
            print("\n {}:".format(manname))
            theroster = self._get_manager_roster(i)
            if len(theroster) > 0:
                print(theroster)
            else:
                print("No players on this team yet.")
        print()  # newline

    def do_save(self, args):
        """
        usage: save [OUTPUT]
        saves player lists to OUTPUT.csv (default OUTPUT is draft_players)
        """
        outname = args if args else "draft_backup"
        # save_player_list(outname, self.ap, self.pp)
        save_data = {key: getattr(self, key) for key in self.save_fields}
        picklename = f"{outname}.state.p"
        logging.info("Saving state as %s", picklename)
        with open(picklename, "wb") as pfile:
            pickle.dump(save_data, pfile)

    def do_show(self, args):
        """
        usage: show ITEM...
        show a position or statistic that has been hidden
        """
        if args.strip().lower() == "all":
            print("Showing all.")
            self.hide_stats = []
            self.hide_pos = []
            return
        for arg in args.split(" "):
            if arg.lower() in self.hide_stats:
                print("Showing {}.".format(arg.lower()))
                self.hide_stats.remove(arg.lower())
            elif arg.upper() in self.hide_pos:
                print("Showing {}.".format(arg.upper()))
                self.hide_pos.remove(arg.upper())
            else:
                print("Could not interpret command to show {}.".format(arg))
                print("Available options are in:")
                print(self.hide_stats)
                print(self.hide_pos)

    def complete_show(self, text, line, begidk, endidx):
        """implements auto-complete for show function"""
        avail_shows = [name.lower() for name in self.hide_pos + self.hide_stats]
        if text:
            return [name for name in avail_shows if name.startswith(text.lower())]
        else:
            return [name for name in avail_shows]

    def do_snake(self, args):
        """
        usage: snake [N] [strat]
        initiate snake draft mode, with the user in draft position N and all other
        managers automatically draft with "strat" strategy
        """
        # self._update_vorp() # called in precmd() now
        if self.draft_mode:
            print("You are already in draft mode!")
            return
        if len(self.pp) > 0:
            print(
                "There are already picked players. This is not starting a draft from scratch."
            )
            print(
                "It is recommended that you quit and start fresh. Draft command will be canceled."
            )
            return
        numprompt = "Enter your position in the snake draft [1,...,{}]: ".format(
            self.n_teams
        )
        argstr = args.split(" ") if args else [input(numprompt)]
        numstr = argstr[0]
        # TODO: low priority: we could allow for multiple users
        try:
            self.user_manager = int(numstr)
            if self.user_manager not in list(range(1, self.n_teams + 1)):
                raise ValueError("Argument not in range.")
        except ValueError as err:
            logging.error(err)
            logging.error("Could not cast argument to draft.")
            logging.error("Use a single number from 1 to {}".format(self.n_teams))
            return

        if argstr[1:]:
            strats = [
                s.lower() for s in argstr[1:] if s.lower() in self._known_strategies
            ]
            for manager in [
                man for man in range(1, self.n_teams + 1) if man != self.user_manager
            ]:
                manstrat = random.choice(strats)
                print(
                    "Setting manager {} to use {} strategy.".format(manager, manstrat)
                )
                self.manager_auto_strats[manager] = manstrat

        # perhaps there is a proxy we can use for this to reduce the number of variables
        self.draft_mode = True
        n_rounds = sum([self.n_roster_per_team[pos] for pos in self.n_roster_per_team])
        print(f"{n_rounds} rounds of drafting commencing.")
        self.manager_picks = []
        for i in range(n_rounds):
            if i % 2 == 0:
                self.manager_picks.extend(list(range(1, self.n_teams + 1)))
            else:
                self.manager_picks.extend(list(range(self.n_teams, 0, -1)))
        self.i_manager_turn = -1
        self._advance_snake()

    def do_sort(self, args):
        """
        usage: sort [QUANT]
        choose a stat to sort by
        if QUANT is not provided, sorts by index (label)
        """
        argl = args.lower().strip()
        if not argl:
            argl = None
        if argl is not None and argl not in self.ap:
            print("argl is not a valid sortable quantity.")
            return
        self._sort_key = argl
        self._sort_asc = argl in [None, "adp", "ecp"]
        # TODO: reset the index here? <- will maybe cause problems after players are drafted

    def complete_sort(self, text, line, begidk, endidx):
        """implements auto-complete for sort function"""
        avail_sorts = [name.lower() for name in self.ap.columns]
        if text:
            return [name for name in avail_sorts if name.startswith(text.lower())]
        else:
            return [name for name in avail_sorts]

    def do_team(self, args):
        """
        usage: team [TEAM]
        lists players by TEAM abbreviation.
        if no argument is provided, lists the available teams.
        """
        if args:
            find_by_team(args, self.ap, self.pp)
        else:
            print_teams(self.ap, self.pp)

    def do_unpick(self, args):
        """
        usage: unpick
        moves player(s) with index(ces) I from picked list to available.
        if no index is provided, then the last player picked will be returned.
        """
        # we used to allow indices, and multiple indices, but this gets too complicated
        # w/ draft mode.
        if not self.pp.empty:
            lasti = self.pp.index[-1]
            try:
                push_to_player_list(lasti, self.ap, self.pp)
                self._update_vorp()
                if self.draft_mode:
                    self._regress_snake()
            except IndexError as err:
                print(err)
                print(f"could not put player ({lasti}) back in available list.")
        else:
            print("No players have been picked.")
        self.update_draft_html()

    # define this here for ease and move it later
    def _step_vona(self, ap, pp, managers_til_next, strat="adp"):
        if not managers_til_next:
            return (ap, pp)
        manager = managers_til_next[0]
        pickidx = self._pick_rec(manager, strat=strat, ap=ap, pp=pp, disabled_pos=None)
        newap = ap.drop(pickidx)
        newpp = ap.loc[ap.index]  # .copy()
        newpp["manager"] = manager
        newpp = pd.concat([pp, newpp], sort=False)
        return self._step_vona(newap, newpp, managers_til_next[1:], strat)

    def do_print_vona(self, args):
        """
        usage: print_vona strat
        print out VONA for each position, assuming `strat` strategy for others
        """
        if not self.manager_picks:
            print("command only available in snake draft mode.")
            return
        # strat = args.strip().lower() if args else None
        for strat in self._known_strategies:
            print(f"Assuming {strat.upper()} strategy:")
            positions = [
                pos
                for (pos, numpos) in list(self.n_roster_per_team.items())
                if pos not in ["FLEX", "BENCH"] and numpos > 0
            ]
            for pos in positions:
                topval = self.ap[self.ap.index.get_level_values("pos") == pos][
                    "exp_proj"
                ].max()
                # get "next available" assuming other managers use strategy "strat" to pick
                managers = self._get_managers_til_next()
                managers.extend(managers[::-1])
                na_ap, na_pp = self._step_vona(self.ap, self.pp, managers, strat)
                naval = na_ap[na_ap.index.get_level_values("pos") == pos][
                    "exp_proj"
                ].max()
                print("{}: {}".format(pos, topval - naval))

    def _get_max_vona_in(self, positions, strat, disabled_pos=None):
        # vona_dict = {pos:0 for pos in positions)
        if disabled_pos is None:
            disabled_pos = []
        max_vona = -1000.0
        max_vona_pos = None
        for pos in positions:
            topval = self.ap[self.ap.index.get_level_values("pos") == pos][
                "exp_proj"
            ].max()
            # get "next available" assuming other managers use strategy "strat" to pick
            managers = self._get_managers_til_next()
            managers.extend(managers[::-1])
            na_ap, _ = self._step_vona(self.ap, self.pp, managers, strat)
            naval = na_ap[na_ap.index.get_level_values("pos") == pos]["exp_proj"].max()
            vona = topval - naval
            if vona > max_vona and pos not in disabled_pos:
                max_vona, max_vona_pos = vona, pos
        return max_vona_pos


def get_hash(df: pd.DataFrame, *objs, n_bits: int = 16) -> str:
    """
    Digest a dataframe into a single hash string
    """
    hash = hashlib.sha256()
    hash.update(pd.util.hash_pandas_object(df).values)
    for obj in objs:
        hash.update(obj)
    return hash.hexdigest()[-n_bits:]


def main():
    """main function that runs upon execution"""

    # default log level is warning
    logging.getLogger().setLevel(logging.INFO)

    # use argument parser
    parser = argparse.ArgumentParser(
        description="Script to aid in real-time fantasy draft"
    )
    parser.add_argument(
        "--ruleset",
        type=str,
        choices=["phys", "dude", "bro", "nycfc", "ram"],
        default="bro",
        help="which ruleset to use of the leagues I am in",
    )
    parser.add_argument(
        "--n-teams", type=int, default=14, help="number of teams in the league"
    )
    parser.add_argument(
        "--n-qb", type=int, default=1, help="number of starting QBs per team"
    )
    parser.add_argument(
        "--n-rb", type=int, default=2, help="number of starting RBs per team"
    )
    parser.add_argument(
        "--n-wr", type=int, default=2, help="number of starting WRs per team"
    )
    parser.add_argument(
        "--n-te", type=int, default=1, help="number of starting TEs per team"
    )
    parser.add_argument(
        "--n-flex", type=int, default=1, help="number of FLEX spots per team"
    )
    parser.add_argument(
        "--n-dst", type=int, default=1, help="number of D/ST spots per team"
    )
    parser.add_argument(
        "--n-k", type=int, default=1, help="number of starting Ks per team"
    )
    parser.add_argument(
        "--n-bench", type=int, default=6, help="number of bench spots per team"
    )
    parser.add_argument(
        "--ci",
        type=float,
        default=0.75,
        help="confidence interval to assume for high/low",
    )
    parser.add_argument(
        "--simulations", type=int, default=4096, help="number of simulations to run"
    )
    parser.add_argument(
        "--auction-cap", type=int, default=200, help="auction budget per manager"
    )
    parser.add_argument(
        "--year", type=int, default=datetime.now().year, help="year of season"
    )

    args = parser.parse_args()
    n_teams = args.n_teams
    n_roster_per_team = {
        "QB": args.n_qb,
        "RB": args.n_rb,
        "WR": args.n_wr,
        "TE": args.n_te,
        "FLEX": args.n_flex,
        "DST": args.n_dst,
        "K": args.n_k,
        "BENCH": args.n_bench,
    }
    n_roster_per_league = {
        key: (n_teams * val) for key, val in n_roster_per_team.items()
    }

    # in principle FLEX can be defined in a different way,
    # so we'll leave this definition local so that we might change it later.
    flex_pos = ["RB", "WR", "TE"]

    # TODO: set this up in its own function in the ruleset file
    if args.ruleset == "phys":
        rules = phys_league
    if args.ruleset == "dude":
        rules = dude_league
    if args.ruleset == "bro":
        rules = bro_league
    if args.ruleset == "nycfc":
        rules = nycfc_league
    if args.ruleset == "ram":
        rules = ram_league

    logging.info("Initializing with ruleset:")
    # print some output to verify the ruleset we are working with
    rulestr = "  {} team, {} PPR".format(n_teams, rules.ppREC)
    if rules.ppPC != 0 or rules.ppINC != 0:
        rulestr += ", {}/{} PPC/I".format(rules.ppPC, rules.ppINC)
    logging.info(rulestr)
    rosterstr = " "
    for pos in [
        "QB",
        "RB",
        "WR",
        "TE",
        "FLEX",
    ]:  # there's always just 1 DST and K, right?
        nper = n_roster_per_team[pos]
        rosterstr += " {}{} /".format(nper, pos)
    logging.info(rosterstr[:-2])

    main_positions = ["QB", "RB", "WR", "TE", "K", "DST"]

    year = args.year
    posdfs = []
    # also collect "floor" and "ceiling" data if it exists
    posdfs_high = []
    posdfs_low = []
    for pos in main_positions:
        filename = "preseason_rankings/project_fp_{}_pre{}.csv".format(
            pos.lower(), year
        )
        posdf = pd.read_csv(filename)
        # TODO (low priority): try using a multi-indexed dataframe instead of
        # decorating every entry with the position?
        posdf["pos"] = pos
        posdfs.append(posdf)

        filename_high = filename.replace(".csv", "_high.csv")
        filename_low = filename.replace(".csv", "_low.csv")
        if path.isfile(filename_high):
            posdf_high = pd.read_csv(filename_high)
            posdf_high["pos"] = pos
            posdfs_high.append(posdf_high)
        if path.isfile(filename_low):
            posdf_low = pd.read_csv(filename_low)
            posdf_low["pos"] = pos
            posdfs_low.append(posdf_low)
    # create dataframe of all available players
    availdf = pd.concat(posdfs, ignore_index=True, sort=False)
    availdf_high = (
        pd.concat(posdfs_high, ignore_index=True, sort=False) if posdfs_high else None
    )
    availdf_low = (
        pd.concat(posdfs_low, ignore_index=True, sort=False) if posdfs_low else None
    )

    alldfs = [availdf]
    if availdf_high is not None:
        alldfs.append(availdf_high)
    if availdf_low is not None:
        alldfs.append(availdf_low)

    # add the team acronym to the DST entries for consistency/elegance
    teamlist = availdf[~availdf.team.isnull()]["team"].sort_values().unique()
    for df in alldfs:
        df.loc[df.pos == "DST", "team"] = df.loc[df.pos == "DST", "player"].map(
            lambda n: get_team_abbrev(n, teamlist)
        )

        # if they have no stats listed (NaN) we can treat that as a zero
        # this should be called before ADP is added, since it has some missing
        # values that we want to keep as NaN for clarity
        df.fillna(0, inplace=True)

    for df in alldfs:
        # decorate the dataframe with projections for our ruleset
        df.loc[df.pos != "DST", "exp_proj"] = get_points(rules, df)
        # for DST, just take the FP projection.
        df.loc[df.pos == "DST", "exp_proj"] = df["fp_projection"]
        keep_cols = ["player", "team", "pos", "exp_proj"]
        drop_cols = [col for col in df.columns if col not in keep_cols]
        # can go ahead and filter out stats once we have projections
        df.drop(drop_cols, axis=1, inplace=True)

    # merge into a single dataframe
    if availdf_high is not None:
        availdf_high.rename(columns={"exp_proj": "exp_proj_high"}, inplace=True)
        availdf = availdf.merge(
            availdf_high[["player", "team", "pos", "exp_proj_high"]],
            how="left",
            on=["player", "team", "pos"],
        )
        del availdf_high
    if availdf_low is not None:
        availdf_low.rename(columns={"exp_proj": "exp_proj_low"}, inplace=True)
        availdf = availdf.merge(
            availdf_low[["player", "team", "pos", "exp_proj_low"]],
            how="left",
            on=["player", "team", "pos"],
        )
        del availdf_low

    adpfname = f"preseason_rankings/fp_adp_pre{year}.csv"
    if os.path.exists(adpfname):
        adpdf = pd.read_csv(adpfname)
        # add team acronym on ADP data too, so that we can use "team" as an additional merge key
        # adpdf = adpdf[~adpdf.pos.str.contains("TOL")]
        # only merge with the columns we are interested in for now.
        # combine on both name and team because there are sometimes multiple players w/ same name
        availdf = availdf.merge(
            adpdf[["player", "team", "adp"]], how="left", on=["player", "team"]
        )
    else:
        logging.warning("Could not find ADP file")
    ecpfname = f"preseason_rankings/fp_ecp_pre{year}.csv"
    if os.path.exists(ecpfname):
        ecpdf = pd.read_csv(ecpfname)
        # add team acronym on ADP data too, so that we can use "team" as an additional merge key
        ecpdf = ecpdf[~ecpdf.pos.str.contains("TOL")]
        # TODO: This should probably just be removed
        ecpdf.loc[ecpdf.team.isnull(), "team"] = ecpdf.loc[
            ecpdf.team.isnull(), "player"
        ].map(lambda n: get_team_abbrev(n, teamlist))
        # only merge with the columns we are interested in for now.
        # combine on both name and team because there are sometimes multiple players w/ same name
        availdf = availdf.merge(
            ecpdf[["player", "team", "ecp"]], how="left", on=["player", "team"]
        )
    else:
        logging.warning("Could not find ECP file")
    availdf.loc[:, "n"] = ""
    availdf.loc[:, "rank"] = ""
    availdf.loc[:, "g"] = ""

    col_order = [
        "player",
        "n",
        "team",
        "pos",
        "rank",
        "g",
        "adp",
        "ecp",
        "exp_proj",
        "exp_proj_high",
        "exp_proj_low",
    ]
    # re-order the columns
    availdf = availdf[[c for c in col_order if c in availdf]]

    # flag players with news items
    newsfile = "data/news.csv"
    if os.path.isfile(newsfile):
        newsdf = pd.read_csv(newsfile)
        newsdf = newsdf[newsdf.pos.isin(main_positions)]
        for _, pnews in newsdf.iterrows():
            pnamenews, pteamnews, posnews = pnews[["player", "team", "pos"]]
            # pnamenews, pteamnews, posnews = index
            # we should be able to just find the intersection of the indices,
            # but the team names are inconsistent.
            # pix = (availdf.index.get_level_values('pos') == posnews)
            # pix &= (availdf.index.get_level_values('player') == pnamenews)
            pix = availdf.pos == posnews
            pix &= availdf.player == pnamenews
            # the team abbreviations are not always uniform #TODO: make it well-defined
            # pix &= (availdf.team == pteamnews)
            if availdf[pix].shape[0] > 1:
                logging.warning(
                    "multiple matches found for news item about {}!".format(pnamenews)
                )
                print(availdf[pix])
            if availdf[pix].shape[0] == 0:
                pix = availdf.pos == posnews
                cutoff = 0.75  # default is 0.6, but this seems too loose
                rmsuff = availdf.player.map(rm_name_suffix)
                pix &= rmsuff.isin(
                    get_close_matches(
                        rm_name_suffix(pnamenews), rmsuff.values, cutoff=cutoff
                    )
                )
                if availdf[pix].shape[0] > 1:
                    logging.warning(
                        "multiple matches found for news item about {}!".format(pnamenews)
                    )
                    print(availdf[pix])
                if availdf[pix].shape[0] == 0:
                    logging.warning(
                        "there is news about %s (%s) %s, but this player could not be found!",
                        pnamenews, pteamnews, posnews
                    )
            availdf.loc[pix, "n"] = "*"  # flag this column
    else:
        newsdf = None
        logging.warning("News file does not exist")

    # default is 17 games; we'll check for suspensions.
    availdf.loc[:, "g"] = games_in_season
    susfile = "data/suspensions.csv"
    if os.path.exists(susfile):
        sussdf = pd.read_csv(susfile)
        rmsuff = availdf.player.map(rm_name_suffix).map(simplify_name).copy()
        for _, psus in sussdf.iterrows():
            pnamesus, pteamsus, possus, gsus = psus[
                ["player", "team", "pos", "games_suspended"]
            ]
            pnamesimp = simplify_name(rm_name_suffix(pnamesus))
            pix = (rmsuff == pnamesimp) & (availdf.pos == possus)
            # pix = (availdf.player == pnamesus) & (availdf.pos == possus)
            # the team abbreviations are not always uniform #TODO: make it well-defined
            # pix &= (availdf.team == pteamsus)
            if len(availdf[pix]) > 1:
                logging.warning("multiple matches found for suspension!")
                print(availdf[pix])
            if len(availdf[pix]) == 0:
                pix = availdf.pos == posnews
                cutoff = 0.75  # default is 0.6, but this seems too loose
                pix &= rmsuff.isin(
                    get_close_matches(pnamesimp, rmsuff.values, cutoff=cutoff)[:1]
                )
                if availdf[pix].shape[0] > 1:
                    logging.warning(
                        "multiple matches found for suspension of {}!".format(pnamenews)
                    )
                    print(availdf[pix])
                if availdf[pix].shape[0] == 0:
                    logging.error(
                        "Could not find {} ({}) {}, suspended for {} games!".format(
                            pnamesus, pteamsus, possus, gsus
                        )
                    )
            if np.isnan(gsus):
                logging.warning("unknown suspension time for {}".format(pnamesus))
            else:
                availdf.loc[pix, "g"] = availdf[pix]["g"] - gsus
    else:
        logging.warning("No suspensions file")

    # re-index on player, team, pos
    index_cols = ["player", "team", "pos"]
    availdf.set_index(index_cols, inplace=True)
    availdf.sort_index(inplace=True)

    hash_vals = []
    for v in vars(args).values():
        hash_vals.append(str(v))
    # NOTE: This might be redundant with the args so long as the rulesets
    # remain constant, but it shouldn't hurt
    for v in rules._asdict().values():
        hash_vals.append(str(v))
    ruleset_hash = ''.join(hash_vals).encode("utf-8")

    hash = get_hash(availdf, ruleset_hash)

    ci = args.ci

    n_sims = args.simulations
    sim_ppg, sim_games = simulate_seasons(availdf, n=n_sims, hash=hash)

    # we can drop the high and low fields here
    availdf.drop(
        ["exp_proj_high", "exp_proj_low"], axis=1, inplace=True
    )

    sim_value = None
    value_cache_name = f"sim_value_cache_{hash}.csv"
    if path.isfile(value_cache_name):
        value_df = pd.read_csv(value_cache_name, index_col=["player", "team", "pos"])
        # Is the games column still in here?
        if len(value_df.columns) >= n_sims:
            logging.info("Loading simulated value from cache")
            sim_value = value_df
    if sim_value is None:
        logging.info("Calculating value from simulations")
        # initialize the dataframe
        sim_value = sim_ppg.copy()
        sim_value[:] = 0
        # The index is now set to player,team,pos
        for col in progressbar(sim_ppg.columns):
            if "Unnamed" in col:
                # there's some extraneous data that is carried along; drop these columns
                logging.warning(f"There is a strange column in the simulations: {col}")
                continue
            sim_value.loc[:, col] = get_player_values(
                sim_ppg, sim_games, n_roster_per_league, value_key=col
            )
        sim_value.to_csv(value_cache_name)

    # define confidence intervals for value
    values_cis = 0.5 * np.array([1, 1 + ci, 1 - ci])
    values_quantiles = sim_value.quantile(values_cis, axis=1)
    medians, highs, lows = (values_quantiles.loc[ci] for ci in values_cis)
    availdf["value"] = medians
    availdf.loc[:, "err_high"] = highs - medians
    availdf.loc[:, "err_low"] = medians - lows

    # sort by index so the next operation has O(1) lookup
    sim_value.sort_index(inplace=True)

    # Do the exact same thing with auction values
    sim_auction = None
    auction_cache_name = f"sim_auction_cache_{hash}.csv"
    if path.isfile(auction_cache_name):
        auction_df = pd.read_csv(
            auction_cache_name, index_col=["player", "team", "pos"]
        )
        # Is the games column still in here?
        if len(auction_df.columns) >= n_sims:
            logging.info("Loading simulated auction from cache")
            sim_auction = auction_df
    if sim_auction is None:
        logging.info("Calculating auction from simulations")
        # initialize the dataframe
        sim_auction = sim_value.copy()
        sim_auction[:] = 0
        for col in progressbar(sim_auction.columns):
            if "Unnamed" in col:
                # there's some extraneous data that is carried along; drop these columns
                logging.warning(
                    f"There is a strange column in the simulation values: {col}"
                )
                continue
            sim_auction.loc[:, col] = get_auction_values(
                sim_value,
                col,
                n_teams,
                n_roster_per_league,
                cap=args.auction_cap,
                min_bid=1,
            )
            # sim_value.loc[:, col] =
            #   get_player_values(sim_ppg, sim_games, n_roster_per_league, value_key=col)
        sim_auction.to_csv(auction_cache_name)

    # define confidence intervals for value
    # values_cis = 0.5*np.array([1, 1+ci, 1-ci])
    auction_quantiles = sim_auction.quantile(values_cis, axis=1)
    medians, highs, lows = (auction_quantiles.loc[ci] for ci in values_cis)
    availdf["auction"] = medians
    # availdf.loc[:, 'auction_high'] = highs
    availdf["auction_high"] = highs
    availdf["auction_low"] = lows

    # Everything added for the variance should happen before this point

    # label nominal (non-flex) starters by their class
    for pos in main_positions:
        # sort the players in each position so we can grab the top indices
        availpos = availdf.loc[
            availdf.index.get_level_values("pos") == pos, :
        ].sort_values("value", ascending=False)
        for i_class in range(n_roster_per_team[pos]):
            ia, ib = i_class * n_teams, (i_class + 1) * n_teams
            itoppos = availpos.index[ia:ib]
            icls = availdf.index.isin(itoppos)
            availdf.loc[icls, "tier"] = "{}{}".format(pos, i_class + 1)
    availflex = availdf.loc[
        (availdf.index.get_level_values("pos").isin(flex_pos))
        & (availdf["tier"].isnull()),
        :,
    ].sort_values("value", ascending=False)
    for i_class in range(n_roster_per_team["FLEX"]):
        ia, ib = i_class * n_teams, (i_class + 1) * n_teams
        itoppos = availflex.index[ia:ib]
        icls = availdf.index.isin(itoppos)
        availdf.loc[icls, "tier"] = "FLEX{}".format(i_class + 1)

    # label backup tier. this undervalues WR and RB
    total_bench_positions = n_roster_per_league["BENCH"]
    total_start_positions = len(availdf[availdf.tier.notnull()])
    crap_positions = ["K", "DST"]
    # there will be some extra spots since the integer division is not exact.
    # fill these with more flex spots.
    n_more_backups = (
        total_start_positions + total_bench_positions - availdf.tier.count()
    )  # count excludes nans
    add_bu_ix = (
        availdf.loc[
            availdf.tier.isnull()
            & (~availdf.index.get_level_values("pos").isin(crap_positions))
        ]
        .sort_values("value", ascending=False)
        .head(n_more_backups)
        .index
    )
    availdf.loc[add_bu_ix, "tier"] = "BU"
    # now label remaining players as waiver wire material
    availdf.loc[availdf.tier.isnull(), "tier"] = "FA"

    for pos in main_positions:
        posdf = availdf[(availdf.index.get_level_values("pos") == pos)].sort_values(
            "value", ascending=False
        )
        for idx in range(posdf.shape[0]):
            label = posdf.index[idx]
            availdf.loc[label, "rank"] = "{}{}".format(pos, idx + 1)

    # Make an empty dataframe with these reduces columns to store the picked
    # players. This might be better as another level of index in the dataframe,
    # or simply as an additional variable in the dataframe. In the latter case
    # we'd need to explicitly exclude it from print statements.
    pickdf = pd.DataFrame(
        columns=availdf.columns,
        index=pd.MultiIndex(levels=[[], [], []], codes=[[], [], []], names=index_cols),
    )

    # set some pandas display options
    pd.options.display.precision = 2  # default is 6
    pd.options.display.width = 108  # default is 80

    # set seaborn style to nice default
    sns.set()

    prompt = MainPrompt()
    prompt.ap = availdf
    prompt.pp = pickdf
    prompt.newsdf = newsdf
    prompt.sim_games = sim_games
    prompt.sim_ppg = sim_ppg
    prompt.n_teams = n_teams
    prompt.n_roster_per_team = n_roster_per_team
    prompt.update_draft_html()
    while True:
        try:
            prompt.cmdloop()
        except (SystemExit, KeyboardInterrupt, EOFError):
            # a system exit, Ctrl-C, or Ctrl-D can be treated as a clean exit.
            #  will not create an emergency backup.
            print("Goodbye!")
            break
        except Exception as err:
            logging.error(err)
            backup_fname = "draft_backup"
            logging.error(sys.exc_info())
            logging.error(f'Backup save with label "{backup_fname}".')
            prompt.do_save(backup_fname)
            # save_player_list(backup_fname, prompt.ap, prompt.pp)
            # raise err


if __name__ == "__main__":
    main()
