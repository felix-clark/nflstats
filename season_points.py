#!/usr/bin/env python

import nflgame
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import norm, gamma

import os.path

# import dist_fit
from ruleset import bro_league, phys_league, dude_league
from getPoints import *
import dist_fit

# this will create a data file with summary data for all positions
def generateSummaryDataFile( fname, years, league_rules ):
    # this list will be used as a constructor to the DataFrame
    # perhaps there is a slicker way to construct a DataFrame?
    datalist = []

    for year in years:
        print 'reading {} season...'.format( year )

        games = nflgame.games_gen( year )
        # get game-level stats
        try:
            all_players = nflgame.combine_game_stats( games )
        except TypeError:
            print 'Could not open {} data.'.format( year )
            continue
        # for some reason "position" does allow any through and we must use "guess_position"
        top_players = {}
        top_players['QB'] = all_players.passing().filter( guess_position='QB' ).filter( passing_yds = lambda y: y >= 500 )
        top_players['RB'] = all_players.rushing().filter( guess_position='RB' ).filter( rushing_yds = lambda y: y>=200 )
        top_players['WR'] = all_players.receiving().filter( guess_position='WR' ).filter( receiving_yds = lambda y: y>=200 )
        top_players['TE'] = all_players.receiving().filter( guess_position='TE' ).filter( receiving_yds = lambda y: y>=200 )
        top_players['K'] = all_players.kicking().filter( guess_position='K' ).filter( kicking_fgyds= lambda y: y>=200 )

        for (pos,players) in top_players.items():
            for pstat in players:
                # print qbstat.rushing_lng, qbstat.rushing_lngtd # longest rush, longest TD rush
                # print qbstat.stats # dict of the stats
                # print qbstat.formatted_stats() # relatively nicely printed out version of the stats dict
                pfullname = pstat.player.full_name.strip()
                base_pts = getBasePoints( league_rules, pstat )
                # make a list of dicts to create a pandas data frame
                datalist.append({'position':pos,
                                 'year':year,
                                 'name':pfullname,
                                 'fantasy_points':base_pts
                                 # 'playerid':pstat.playerid
                })
                df = pd.DataFrame(datalist)
                df.to_csv( fname )
            


datafilename = 'season_points.csv'
if not os.path.isfile( datafilename ):
    # at the moment we only have preseason ranking data for 2015 and 2016
    generateSummaryDataFile( datafilename, range(2009,2017), league_rules = bro_league )


                
df = pd.DataFrame.from_csv( datafilename )

# everything looks better with a default set() call.
# can add arguments to tweak from default
sns.set()

for pos in ['QB', 'RB', 'WR', 'TE', 'K']:
    point_data = df[df.position==pos]['fantasy_points']
    g = sns.distplot(point_data, kde=True, rug=True, fit=gamma)
    # g.set_axis_labels('yearly fantasy points (w/out bonus)')
    g.set_title( '{} yearly fantasy points'.format(pos) )
    # print dir(g)
    g.get_figure().savefig('{}_score.png'.format(pos.lower()))

    plt.show()
    # plt.savefig('qb_score.png')

    # dist_fit.plot_counts( df[df.position==pos]['fantasy_points'], label='{} fantasy points'.format(pos), fits=['neg_binomial'] )
    # dist_fit.plot_counts_poly( df[df.position==pos]['fantasy_points'], label='{} fantasy points'.format(pos))
