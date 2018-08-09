#!/usr/bin/python

import nflgame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sys import argv
import os.path

from ruleset import bro_league, phys_league
from get_fantasy_points import *

def playerFilename( pname ):
    return pname.lower().replace(' ', '_')

# this will create a data file with summary data for all positions
def generatePlayerDataFile( player_name='tom brady', years=range(2009, 2017) ):

    poss_players = [p for p in nflgame.find( player_name ) if p.position == 'QB']
    if not poss_players:
        print 'could not find ', player_name, ' in nfl database.'
        exit(1)
    if len(poss_players) > 1:
        print 'found multiple ', player_name, '\'s!'
        exit(2)
    player = poss_players[0]
    playerid = player.playerid
    
    # this list will be used as a constructor to the DataFrame
    # perhaps there is a slicker way to construct a DataFrame?
    datalist = []
    for year in years:
        print 'reading {} season...'.format( year )
        for week in range(1,18):
            # print 'looking at week ', week
            weekly_games = nflgame.games_gen( year, week )
            try:
                # use combine_play_stats() to get play-level information
                weekly_player_stats = nflgame.combine_game_stats( weekly_games )
            except TypeError:
                print str(year), ' not in database.'
                break
            for pstat in weekly_player_stats.passing().filter( playerid=playerid ):
                base_pts = getBasePoints( bro_league, pstat )
                datalist.append({'year':year,
                                 'week':week,
                                 'fantasy_points':base_pts,
                                 'pass_attempts':pstat.passing_att,
                                 'completions':pstat.passing_cmp,
                                 'pass_yards':pstat.passing_yds,
                                 'pass_td':pstat.passing_td,
                                 'interceptions':pstat.passing_int,
                                 'passer_rating':pstat.passer_rating(),
                                 'rushing_yds':pstat.rushing_yds,
                                 'rushing_td':pstat.rushing_td,
                                 # '2pc':pstat.passing_twoptm+pstat.rushing_twoptm+pstat.receiving_twoptm
                                 # 'playerid':pstat.playerid
                })

                df = pd.DataFrame(datalist)
                df.to_csv( playerFilename( player_name ) + '.csv' )


player_name = ' '.join(argv[1:]).strip()
datafilename = playerFilename( player_name ) + '.csv'
if not os.path.isfile( datafilename ):
    generatePlayerDataFile( player_name )

                
df = pd.DataFrame.from_csv( datafilename )

# everything looks better with a default set() call.
# can add arguments to tweak from default
sns.set()

# can edit 'vars' to only show some stats
plot_vars = ['week', 'fantasy_points',
             # 'pass_attempts',
             'completions', 'pass_yards', 'pass_td', 'interceptions', 'passer_rating']
g = sns.pairplot(data=df, hue='year', vars=plot_vars)
plt.show()
g.savefig( playerFilename( player_name ) + '.png')

