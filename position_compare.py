#!/usr/bin/env python

import nflgame
import numpy as np
import pandas as pd
import seaborn as sns

import os.path

# import dist_fit
from ruleset import bro_league, phys_league
from getPoints import *

def getPreseasonRank( position, year, pname ):
    filename = 'preseason_rankings/top_' + position.lower() + 's_pre' + str(year) + '.txt'
    f = open(filename, 'r')
    rank = None
    for l in f:
        ldata = l.split('. ')
        if pname.lower() == '. '.join(ldata[1:]).rstrip('\n').lower():
            rank = int( ldata[0] )
            break
    f.close()
    return rank
    

# this will create a data file with summary data for all positions
def generateSummaryDataFile( fname, years ):
    # this list will be used as a constructor to the DataFrame
    # perhaps there is a slicker way to construct a DataFrame?
    datalist = []

    for year in years:
        print 'reading {} season...'.format( year )

        games = nflgame.games_gen( year )
        # get game-level stats
        all_players = nflgame.combine_game_stats( games )
        # top_qbs = all_players.passing().filter( playerid=lambda x: x in qb_ids ).sort('passing_yds').limit( 32 )
        # for some reason "position" does allow any through and we must use "guess_position"
        top_players = {}
        top_players['QB'] = all_players.passing().filter( guess_position='QB' ).sort('passing_yds').limit( 32 )
        top_players['RB'] = all_players.rushing().filter( guess_position='RB' ).sort('rushing_yds').limit( 64 )
        top_players['WR'] = all_players.receiving().filter( guess_position='WR' ).sort('receiving_yds').limit( 64 )
        top_players['TE'] = all_players.receiving().filter( guess_position='TE' ).sort('receiving_yds').limit( 24 )
        # checked that kickers have a very flat response vs. preseason rank.
        # check again once we add bonuses for long FGs?
        # top_players['K'] = all_players.kicking().filter( guess_position='K' ).sort('kicking_fgyds').limit( 24 )

        for (pos,players) in top_players.items():
            for pstat in players:
                # print qbstat.rushing_lng, qbstat.rushing_lngtd # longest rush, longest TD rush
                # print qbstat.stats # dict of the stats
                # print qbstat.formatted_stats() # relatively nicely printed out version of the stats dict
                pfullname = pstat.player.full_name.strip()
                base_pts = getBasePoints( bro_league, pstat )
                # make a list of dicts to create a pandas data frame
                ps_rank = getPreseasonRank( pos, year, pfullname )
                datalist.append({'position':pos,
                                 'year':year,
                                 'name':pfullname,
                                 'preseason_rank':ps_rank,
                                 'fantasy_points':base_pts
                                 # 'playerid':pstat.playerid
                })

                # can and should factor the reading of data from nflgame
                # and the plotting of it.
                # use DataFrame.read_csv(<filename>) (and to_csv(<filename>) ?)
                df = pd.DataFrame(datalist)
                df.to_csv( fname )
            

                # at some point it would be nice to run through by week/play to attribute bonus points
                # for week in range(1,18):
                #     # print 'looking at week ', week
                #     weekly_games = nflgame.games_gen( year, week )
                #     weekly_player_stats = nflgame.combine_game_stats( weekly_games )
                #     for pstat in weekly_player_stats.passing().filter( playerid=lambda x: x in top_playerids ):
                #         # msg = '{} ({}): {} carries for {} yards and {} TDs'.format( pstat, pstat.player.full_name,
                #         #                                                             pstat.rushing_att, pstat.rushing_yds,
                #         #                                                             pstat.rushing_tds )
                #         # print msg
                #         # print dir(pstat) # to check methods
                #         rshyd = pstat.rushing_yds
                #         rshtd = pstat.rushing_tds


datafilename = 'playerdata.csv'
if not os.path.isfile( datafilename ):
    # at the moment we only have preseason ranking data for 2015 and 2016
    generateSummaryDataFile( datafilename, range(2015,2017) )


                
df = pd.DataFrame.from_csv( datafilename )

# everything looks better with a default set() call.
# can add arguments to tweak from default
sns.set()

g = sns.lmplot(x='preseason_rank',y='fantasy_points',hue='year',
               # can also use "items" to select specific branches in the filter
               data=df[(df.position=='QB') & (df.preseason_rank<=28)]
               )
g.set_axis_labels('ESPN staff preseason ranking', 'yearly fantasy points (w/out bonus)')
g.savefig('qbs.png')

g = sns.lmplot(x='preseason_rank',y='fantasy_points',hue='position',
               data=df[df.preseason_rank<=16]
               )
g.set_axis_labels('ESPN staff preseason ranking', 'yearly fantasy points (w/out bonus)')
g.savefig('compare_pos.png')

