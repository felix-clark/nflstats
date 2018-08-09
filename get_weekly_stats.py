#!/usr/bin/env python
from __future__ import division
import logging
import os.path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nflgame

# list of the positions we care about in fantasy
off_pos = ['QB', 'RB', 'WR', 'TE', 'K']

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    
    if not os.path.isdir('weekly_stats'):
        logging.info('creating directory weekly_stats')
        os.mkdir('weekly_stats')
        
    firstyear,lastyear = 2009,2017 # 2009 # full range of available data is from 2009
    for year in range(firstyear, lastyear+1):
        print 'processing {} season'.format( year )
        games = nflgame.games_gen( year, kind='REG' )
        
        # get game-level stats
        # all_players = nflgame.combine_game_stats( games )
        # n_top_players = 32
        # top_player_names = [ p.player.full_name for p in all_players.rushing().sort('rushing_yds').limit( n_top_players ) ]
        # top_playerids = [ p.playerid for p in all_players.rushing().sort('rushing_yds').limit( n_top_players ) ]
        
        # print top_player_names
        df = pd.DataFrame()
        # nflgame.combine(nflgame.games(year)).csv('weekly_stats/fantasy_weekly_stats_{}.csv'.format(year)) # this is for the entire year
        
        for week in range(1,18):
            # print 'looking at week ', week
            weekly_games = nflgame.games_gen( year, week )
            weekly_player_stats = nflgame.combine_game_stats( weekly_games )
            # this sort of works (but not filtering?):
            # weekly_player_stats.csv('weekly_stats/fantasy_stats_year_{}_week_{}.csv'.format(year, week))
            # weekly_player_stats.filter(position=lambda p: p.player in off_pos).csv('weekly_stats/fantasy_stats_year_{}_week_{}.csv'.format(year, week))
            # for pstat in weekly_player_stats.rushing().filter( playerid=lambda x: x in top_playerids ):
            
            dict_list = []
            for pstat in weekly_player_stats:
                if pstat.player is None:
                    logging.error('could not get player object for:')
                    print(pstat)
                    print(pstat.stats)
                    continue
                pos = pstat.player.position
                if pos not in off_pos: continue
                # if pos == 'K':
                #     print(pstat.stats) # lets see which stats are relevant
                statdict = pstat.stats
                statdict['name'] = pstat.player.full_name
                statdict['playerid'] = pstat.playerid
                statdict['pos'] = pos
                statdict['team'] = pstat.team
                statdict['year'] = year
                statdict['week'] = week
                try:
                    statdict['passing_int'] = statdict.pop('passing_ints') # rename this for consistency
                except:
                    pass
                statdict['twoptm'] = pstat.rushing_twoptm + pstat.receiving_twoptm
                # unfortunately receiving targets are not recorded in the weekly stats... TODO: scrape play-by-play?
                # it does seem to be in nfldb, under receiving_tar
                dict_list.append(statdict)

            df = df.append(dict_list)

        # print('columns scraped (not all will be saved)')
        # print(df.columns)
        df = df[['name','team','pos','playerid','week',
                 'passing_att','passing_cmp','passing_yds','passing_tds','passing_int',
                 'rushing_att','rushing_yds','rushing_tds',
                 'receiving_rec','receiving_yds','receiving_tds', # add 'receiving_tgt' in here once we get it
                 'kicking_xpa', 'kicking_xpmade', 'kicking_fga', 'kicking_fgm', 'kicking_fgyds',
                 'twoptm', 'passing_twoptm', 'fumbles_lost']]
         # for kickers, just keep attempts and successes?
         # there are also misses recorded directly, but we don't need them.
         # blocks (fgb/xpb) just count as misses.
        df.to_csv('weekly_stats/fantasy_stats_year_{}.csv'.format(year))
                
