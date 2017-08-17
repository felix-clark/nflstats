#!/usr/bin/env python

import nflgame

import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt

import dist_fit
from ruleset import bro_league, phys_league

# doesn't cover every rule, just the main ones.
# some stats are not as easily extracted
def getBasePoints( rs, plyr ):
    # missing bonus 40 and 50 yd tds here. other method getBonusPoints, which looks at weekly stats??
    # also missing extra points for longer field goals
    # note: for some reason a line like
    # + some_val
    # is valid python, but will not get added
    fpts = \
    + rs.ppPY * plyr.passing_yds \
    + rs.ppPY25 * (plyr.passing_yds / 25) \
    + rs.ppPC * plyr.passing_cmp \
    + rs.ppINC * (plyr.passing_att - plyr.passing_cmp) \
    + rs.ppPTD * plyr.passing_tds \
    + rs.ppINT * plyr.passing_int \
    + rs.pp2PC * plyr.passing_twopc \
    + rs.ppRY * plyr.rushing_yds \
    + rs.ppRY10 * (plyr.rushing_yds / 10) \
    + rs.ppRTD * plyr.rushing_tds \
    + rs.pp2PR * plyr.rushing_twoptm \
    + rs.ppREY * plyr.receiving_yds \
    + rs.ppREY10 * (plyr.receiving_yds / 10) \
    + rs.ppREC * plyr.receiving_rec \
    + rs.ppRETD * plyr.receiving_tds \
    + rs.pp2PRE * plyr.receiving_twoptm \
    + rs.ppFUML * plyr.fumbles_lost \
    + rs.ppPAT * plyr.kicking_xpmade \
    + rs.ppFGM * (plyr.kicking_fga - plyr.kicking_fgm) \
    + rs.ppFG0 * plyr.kicking_fgm
    return fpts


qb_fpts = []
rb_fpts = []
wr_fpts = []
te_fpts = []
k_fpts = []


# for year in range(2009, 2017):
for year in range(2016, 2017):
    # year = 2009 # options range from 2009 to present
    print 'processing {} season'.format( year )
    games = nflgame.games_gen( year )
    # get game-level stats
    all_players = nflgame.combine_game_stats( games )
    # top_qbs = all_players.passing().filter( playerid=lambda x: x in qb_ids ).sort('passing_yds').limit( 32 )
    # for some reason "position" does allow any through and we must use "guess_position"
    top_qbs = all_players.passing().filter( guess_position='QB' ).sort('passing_yds').limit( 32 )
    top_rbs = all_players.rushing().filter( guess_position='RB' ).sort('rushing_yds').limit( 64 )
    top_wrs = all_players.receiving().filter( guess_position='WR' ).sort('receiving_yds').limit( 80 )
    top_tes = all_players.receiving().filter( guess_position='TE' ).sort('receiving_yds').limit( 32 )
    top_ks = all_players.kicking().filter( guess_position='K' ).sort('kicking_fgyds').limit( 32 )

    # print [rb.name for rb in top_rbs]  # uses up list; would need copy to do additional operations
    
    # top_player_names = [ p.player.full_name for p in all_players.passing().sort('passing_yds').limit( n_top_players ) ]
    # top_playerids = [ p.playerid for p in all_players.passing().sort('passing_yds').limit( n_top_players ) ]

    for qbstat in top_qbs:
        # print qbstat.rushing_lng, qbstat.rushing_lngtd # longest rush, longest TD rush
        # print qbstat.stats # dict of the stats
        # print qbstat.formatted_stats() # relatively nicely printed out version of the stats dict
        base_pts = getBasePoints( bro_league, qbstat )
        qb_fpts.append( base_pts )
    for rbstat in top_rbs:
        base_pts = getBasePoints( bro_league, rbstat )
        rb_fpts.append( base_pts )
    for wrstat in top_wrs:
        base_pts = getBasePoints( bro_league, wrstat )
        wr_fpts.append( base_pts )
    for testat in top_tes:
        base_pts = getBasePoints( bro_league, testat )
        te_fpts.append( base_pts )
    for kstat in top_ks:
        # print dir(kstat)
        base_pts = getBasePoints( bro_league, kstat )
        k_fpts.append( base_pts )
        
        
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
    #         rec = pstat.receiving_rec
    #         recyd = pstat.receiving_yds
    #         rectd = pstat.receiving_tds
    #         tds = pstat.tds
    #         # print rshtd,rectd,tds
    #         rush_att.append( pstat.rushing_att )
    #         # rush_tds.append( rshtd )
    #         # rush_yds.append( rshyd )
    #         all_yds.append( rshyd + recyd )
    #         rec_rec.append( rec )
    #         all_tds.append( tds )


print 'season fantasy points:'
print 'QB:'
print qb_fpts
print 'RB:'
print rb_fpts
print 'WR:'
print wr_fpts
print 'TE:'
print te_fpts
print 'K:'
print k_fpts

# in a single game
# dist_fit.plot_counts( fantasy_points, label='points', fits=['neg_binomial'] )
# dist_fit.plot_counts_poly( fantasy_points, bounds=(-50,400), label='total yards' )

# there are games with negative yards so counts are not appropriate
# print 'rushing yards:'
# dist_fit.plot_counts( rush_yds, label='rushing yards' )


# qb_names = ['Tom Brady', 'Drew Brees', 'Russell Wilson']
# qbs = []
# for qbn in qb_names:
#     qbs += nflgame.find( qbn )
# # player = None
# # players = nflgame.find( player_name )
# # if not players:
# #     print 'Could not find player ', player_name
# #     exit(1)
# # if len( players ) > 1:
# #     print 'Found multiple ', player_name, 's!'
# #     exit(1)
# print [qb.name for qb in qbs]
# print [qb.position for qb in qbs]
# qb_ids = [qb.playerid for qb in qbs]


# def get_rush_att( player_name = 'todd gurley', year = 2013 ):
#     '''
#     player: 
#     year: season ( >= 2009 )
#     '''

# player = None
# players = nflgame.find( player_name )
# if not players:
#     print 'Could not find player ', player_name
#     exit(1)
# if len( players ) > 1:
#     print 'Found multiple ', player_name, 's!'
#     exit(1)
