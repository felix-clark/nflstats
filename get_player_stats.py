#!/usr/bin/env python3
from urllib.request import urlopen
from bs4 import BeautifulSoup
from sys import argv
import pandas as pd
import logging
import os

def get_player_stats(pfrid, years):
    """
    get the dataframe of the player's weekly stats
    pfrid: pro-football-reference id (e.g. GurlTo01)
    """
    f = 'data/players/{id}.csv'.format(id=pfrid)
    df = None
    if os.path.isfile(f):
        try:
            df = pd.read_csv(f)
        except e:
            logging.error('could not read {}: {}'.format(f, e))
    if df is None:
        logging.info('making cache for {}'.format(pfrid))
        df = _make_cache(pfrid, years)
        
    return df


def _make_dirs():
    if not os.path.isdir('data'):
        logging.info('creating data/')
        os.mkdir('data')
    if not os.path.isdir('data/players'):
        logging.info('creating data/players/')
        os.mkdir('data/players')

def _make_cache(pfrid, years):
    _make_dirs()

    # don't save some useless or redundant data
    ignore_cols = ['game_date', 'age',
                   'pass_cmp_perc',
                   'all_td',
                   'scoring',
                   ]
    
    df = pd.DataFrame()
    
    for year in years:
        url = 'https://www.pro-football-reference.com/players/{}/{}/gamelog/{year}/'.format(pfrid[0], pfrid, year=year)
        html = urlopen(url)
        soup = BeautifulSoup(html, 'lxml')
        table_rows = soup.select('#stats tr')
        stats = _get_stats(table_rows, ignore_cols)
        stats['year'] = year
        # sort=False because we've already ordered the columns how we want
        df = df.append(stats, ignore_index=True, sort=False)
    f = 'data/players/{id}.csv'.format(id=pfrid)
    df.to_csv(f, index=False)
    return df

def _get_stats(table_rows, ignore_cols=None):
    """
    get the player data from the table data (td) elements in the table rows (tr)
    """
    # logging.info('in get_stats')
    if ignore_cols is None:
        ignore_cols = []

    # set the first few columns to an organized order
    df = pd.DataFrame(columns=['year', 'game_num', 'team', 'game_location', 'opp', 'game_result'])
    
    for row in table_rows:
        # skip some empty rows, like the label rows
        if(len(row.find_all('td')) == 0):
            continue
        player_dict = {}
        for thing in row:
            key = thing.get('data-stat')
            value = thing.get_text()
            if key not in ignore_cols and '_pct' not in key and 'yds_per' not in key:
                player_dict[key] = value
        rker = player_dict.pop('ranker')
        if rker:
            df = df.append(player_dict, ignore_index=True)
    return df

def _undrafted_players():
    """
    there are a few relevant players that were not drafted.
    this will just provide manually a few drafted since 1992
    """
    # TODO: would be nice to automate this a bit
    columns = ['player', 'pfr_id', 'pos', 'year', 'year_max']
    players = [
        ['Kurt Warner', 'WarnKu00', 'QB', 1998, 2009],
        ['Tony Romo', 'RomoTo00', 'QB', 2004, 2016],
        ['Antonio Gates', 'GateAn00', 'TE', 2003, 2017], # might need to update the final year?
        ['Rod Smith', 'SmitRo01', 'WR', 1995, 2006],
        ['James Harrison', 'HarrJa23', 'OLB', 2002, 2017],
        ['Priest Holmes', 'HolmPr00', 'RB', 1997, 2007],
        ['Adam Vinatieri', 'vinatada01', 'K', 1996, 2017],
        ['Jason Peters', 'PeteJa21', 'T', 2004, 2017], # started as a tight end
        ['Wes Welker', 'WelkWe00', 'WR', 2004, 2015],
        ['Fred Jackson', 'JackFr02', 'RB', 2007, 2015],
        ['Danny Woodhead', 'WoodDa02', 'RB', 2009, 2017],
        ['Mike Tolbert', 'TolbMi00', 'RB', 2008, 2017],
        ['John Kuhn', 'KuhnJo00', 'RB', 2006, 2017], # fullback
        ['Allen Hurns', 'HurnAl01', 'WR', 2014, 2017],
        ['Vontaze Burfict', 'BurfVo00', 'LB', 2012, 2017], # ILB, i think
        ['Malcom Butler', 'ButlMa01', 'CB', 2014, 2017],
        ['Michael Bennett', 'BennMi99', 'DE', 2009, 2017],
        ['Victor Cruz', 'CruzVi00', 'WR', 2010, 2016],
        ['C.J. Anderson', 'AndeC.00', 'RB', 2013, 2017],
        ['Arian Foster', 'FostAr00', 'RB', 2009, 2016],
    ]
    return pd.DataFrame(columns=columns, data=players)

# assuming all the draft data is there, make a file that lists names, ids, positions, and years active
def get_fantasy_player_dict(startyear=1992):
    logging.info('in get_fantasy_player_dict')
    lastyear = 2018
    fname = 'data/players/index.csv'
    if os.path.isfile(fname):
        return pd.read_csv(fname)
    logging.info('generating relevant player index file')
    _make_dirs()
    positions = ['QB', 'RB', 'WR', 'TE', 'K']
    df = _undrafted_players()
    df = df[df['pos'].isin(positions)]
    # df = pd.DataFrame(columns=keepcols)
    # keepcols = ['player', 'pfr_id', 'pos', 'year', 'year_max']
    keepcols = df.columns
    for year in range(startyear, lastyear+1):
        draftdf = pd.read_csv('data/draft/class_{}.csv'.format(year))
        keepix = (draftdf['pos'].isin(positions)) & (draftdf['years_as_primary_starter'] > 0)
        draftdf = draftdf[keepix]
        # filter out players who have been around but usually just as backups
        # we could consider a more refined cutoff
        years_in_league = draftdf['year_max']+1 - draftdf['year']
        keepix = draftdf['years_as_primary_starter'] >= years_in_league // 2
        draftdf = draftdf[keepix]
        df = df.append(draftdf[keepcols])
    df.to_csv(fname, index=False)
    return df
    
def get_pos_players(pos, startyear=1992):
    logging.info('in get_pos_players')
    plls = pd.read_csv('data/players/index.csv')
    return plls[plls.pos == pos]

