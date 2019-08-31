#!/usr/bin/env python3
import logging
import os.path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import nflgame

# from get_player_stats import get_player_stats, get_fantasy_player_dict
from nflstats.get_player_stats import *
import nflstats.get_player_stats

# list of the positions we care about in fantasy
off_pos = ['QB', 'RB', 'WR', 'TE', 'K']

def main():
    logging.getLogger().setLevel(logging.DEBUG)
    players = get_fantasy_player_dict()
    if len(argv) > 1:
        pos = argv[1]
        if pos.upper() in off_pos:
            players = get_pos_players(pos.upper())
            for _,player in players.iterrows():
                logging.info('scraping for player {}'.format(player['player']))
                # print(player)
                # years = int(player['year']), int(player['year_max'])+1
                # get_player_stats(player['pfr_id'], range(*years))
                get_player_stats(player['pfr_id'])
        else:
            pfr_id = argv[1]
            get_player_stats(pfr_id)
    else:
        thisfile = os.path.basename(argv[0])
        logging.info(f'usage: {thisfile} <position or id>')

    # the rest of this code relied on nflgame, which we don't really need now that we've got weekly scraping
        
    # if not os.path.isdir('weekly_stats'):
    #     logging.info('creating directory weekly_stats')
    #     os.mkdir('weekly_stats')
        
    # # Fred Jackson, Danny Woodhead, and Marcel Reece are listed as WRs in yearly data
    # # they've been fixed locally at the time of writing, but re-scraping would put the same error back
    # dict_id_pos = {'00-0024204':'RB', '00-0026019':'RB', '00-0026393':'RB'}
    
    # firstyear,lastyear = 2009,2017 # 2009 # full range of available data is from 2009, though 2009 is not complete
    # for year in range(firstyear, lastyear+1):
    #     print 'processing {} season'.format( year )
    #     games = nflgame.games_gen( year, kind='REG' )
        
    #     # get game-level stats
    #     # all_players = nflgame.combine_game_stats( games )
    #     # n_top_players = 32
    #     # top_player_names = [ p.player.full_name for p in all_players.rushing().sort('rushing_yds').limit( n_top_players ) ]
    #     # top_playerids = [ p.playerid for p in all_players.rushing().sort('rushing_yds').limit( n_top_players ) ]
        
    #     # print top_player_names
    #     df = pd.DataFrame()
    #     # nflgame.combine(nflgame.games(year)).csv('weekly_stats/fantasy_weekly_stats_{}.csv'.format(year)) # this is for the entire year
        
    #     for week in range(1,18):
    #         # print 'looking at week ', week
    #         weekly_games = nflgame.games_gen( year, week )
    #         weekly_player_stats = nflgame.combine_game_stats( weekly_games )
    #         # this sort of works (but not filtering?):
    #         # weekly_player_stats.csv('weekly_stats/fantasy_stats_year_{}_week_{}.csv'.format(year, week))
    #         # weekly_player_stats.filter(position=lambda p: p.player in off_pos).csv('weekly_stats/fantasy_stats_year_{}_week_{}.csv'.format(year, week))
    #         # for pstat in weekly_player_stats.rushing().filter( playerid=lambda x: x in top_playerids ):
            
    #         dict_list = []
    #         for pstat in weekly_player_stats:
    #             if pstat.player is None:
    #                 logging.error('could not get player object for:')
    #                 print(pstat)
    #                 print(pstat.stats)
    #                 continue
    #             pos = pstat.player.position 
    #             if not pos:
    #                 pos = get_position(pstat, dict_id_pos, year)
    #             if pos not in off_pos:
    #                 continue
    #             # if pos == 'K':
    #             #     print(pstat.stats) # lets see which stats are relevant
    #             statdict = pstat.stats
    #             statdict['name'] = pstat.player.full_name
    #             statdict['playerid'] = pstat.playerid
    #             statdict['pos'] = pos
    #             statdict['team'] = pstat.team
    #             statdict['year'] = year
    #             statdict['week'] = week
                
    #             try:
    #                 statdict['passing_int'] = statdict.pop('passing_ints') # rename this for consistency
    #             except:
    #                 pass
    #             statdict['twoptm'] = pstat.rushing_twoptm + pstat.rec_twoptm
    #             # unfortunately receiving targets are not recorded in the weekly stats... TODO: scrape play-by-play?
    #             # it does seem to be in nfldb, under rec_tar
    #             dict_list.append(statdict)

    #         df = df.append(dict_list)

    #     # print('columns scraped (not all will be saved)')
    #     # print(df.columns)
    #     df = df[['name','team','pos','playerid','week',
    #              'passing_att','passing_cmp','passing_yds','passing_tds','passing_int',
    #              'rushing_att','rushing_yds','rushing_tds',
    #              'rec','rec_yds','rec_tds', # add 'rec_tgt' in here once we get it
    #              'kicking_xpa', 'kicking_xpmade', 'kicking_fga', 'kicking_fgm', 'kicking_fgyds',
    #              'twoptm', 'passing_twoptm', 'fumbles_lost']]
    #      # for kickers, just keep attempts and successes?
    #      # there are also misses recorded directly, but we don't need them.
    #      # blocks (fgb/xpb) just count as misses.
    #     df.to_csv('weekly_stats/fantasy_stats_year_{}.csv'.format(year))
        
# def get_position(pstat, dict_id_pos, year):
#     pid = pstat.playerid
#     if pid not in dict_id_pos:    
#         # if pstat.passing_att == 0 and pstat.rushing_att == 0 and pstat.rec ==0:
#         #     # probably defensive
#             # dict_id_pos[pid] = 'def'
#         checkdf = pd.read_csv('yearly_stats/fantasy_{}.csv'.format(year))
#         checkdf = checkdf[checkdf['name'] == pstat.player.full_name]
#         poses = checkdf['pos']
#         if poses.size == 0:
#             # logging.warning('no player with name {} found. will skip.'.format(pstat)) # could be defense
#             dict_id_pos[pid] = None
#         elif (poses == poses.iloc[0]).all(): # all elements are the same
#             dict_id_pos[pid] = poses.iloc[0]
#         else:
#             logging.warning('multiple results found for {} ({})'.format(pstat, poses.tolist()))
#             if 'QB' in poses.tolist():
#                 if pstat.passing_att > 2:
#                     dict_id_pos[pid] = 'QB'
#                     return dict_id_pos[pid]
#                 else: poses = poses[poses != 'QB']
#             if 'RB' in poses.tolist():
#                 if pstat.rushing_att > 4:
#                     dict_id_pos[pid] = 'RB'
#                     return dict_id_pos[pid]
#                 else: poses = poses[poses != 'RB']
#             assert(poses.size > 0)
#             if poses.size == 1:
#                 dict_id_pos[pid] = poses.iloc[0]
#             else:
#                 logging.error('still can\'t resolve descrepancy')
#                 print(checkdf)
#     return dict_id_pos[pid]

if __name__ == '__main__':
    main()
