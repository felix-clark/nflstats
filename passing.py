#!/usr/bin/env python

import nflgame

import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt

import dist_fit

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

rush_att = []
# rush_tds = []
all_tds = []
rush_yds = []
all_yds = []
rec_rec = []
all_twopta, all_twoptm = [],[] # can be split into receiving/rushing
#  there are other stats: fumbles lost, etc...

fantasy_points = []
# points per
ppPY25 = 1 # points per 25 passing yards
ppPC = 1 # points per completion
ppINC = -1 # points per incompletion
ppPTD = 4 # points per touchdown pass
ppPTD40 = 2 # points per 40+ yard TD pass
ppPTD50 = 3 # points per 50+ yard TD pass
ppINT = -2 # points per interception
pp2PC = 2 # points per 2pt passing conversion

ppRY10 = 1 # points per 10 rushing yards
ppRTD = 6 # points per rushing touchdown
ppRTD40 = 2 # points per 40+ yard rushing TD
ppRTD50 = 3 # points per 50+ yard rushing TD
pp2PR = 2 # points per 2pt rushing conversion

ppREY10 = 1 # points per 10 receiving yards
ppREC = 1 # points per reception
ppRETD = 6 # points per receiving touchdown
ppRETD40 = 2 # points per 40+ yard receivinv TD
ppRETD50 = 3 # points per 50+ yard receivinv TD
pp2PRE = 2 # points per 2pt receiving conversion


# for year in range(2009, 2017):
for year in range(2016, 2017):
    # year = 2009 # options range from 2009 to present
    print 'processing {} season'.format( year )
    games = nflgame.games_gen( year )
    # get game-level stats
    all_players = nflgame.combine_game_stats( games )

    n_top_players = 32
    top_player_names = [ p.player.full_name for p in all_players.passing().sort('passing_yds').limit( n_top_players ) ]
    top_playerids = [ p.playerid for p in all_players.passing().sort('passing_yds').limit( n_top_players ) ]
    # top_players = [ p.playerid for p in all_players.rushing().sort('rushing_yds').limit( n_top_players ) ]

    # print top_player_names

    for week in range(1,18):
        # print 'looking at week ', week
        weekly_games = nflgame.games_gen( year, week )
        weekly_player_stats = nflgame.combine_game_stats( weekly_games )
        for pstat in weekly_player_stats.passing().filter( playerid=lambda x: x in top_playerids ):
            # msg = '{} ({}): {} carries for {} yards and {} TDs'.format( pstat, pstat.player.full_name,
            #                                                             pstat.rushing_att, pstat.rushing_yds,
            #                                                             pstat.rushing_tds )
            # print msg
            # print dir(pstat) # to check methods
            rshyd = pstat.rushing_yds
            rshtd = pstat.rushing_tds
            rec = pstat.receiving_rec
            recyd = pstat.receiving_yds
            rectd = pstat.receiving_tds
            tds = pstat.tds
            # print rshtd,rectd,tds
            rush_att.append( pstat.rushing_att )
            # rush_tds.append( rshtd )
            # rush_yds.append( rshyd )
            all_yds.append( rshyd + recyd )
            rec_rec.append( rec )
            all_tds.append( tds )


# print 'rushing attempts:'
# # negative binomial does quite well here for single year.
# # note p~0.5 ... more like 0.7 w/ all seasons
# # poisson is underdispersed.
# # neg bin doesn't do as well w/ all years, but still better than poisson
# dist_fit.plot_counts( rush_att, label='rushing attempts', fits=['neg_binomial'] )

# print 'total TDs:'
# # negative binomial is redundant w/ poisson here. TDs are rare, and relatively independent.
# # geometric does OK, but is clearly inferior w/ high stats
# # poisson does quite well even when all years are combined : -logL/N ~ 1
# dist_fit.plot_counts( all_tds, label='touchdowns', fits=['poisson'] )

# print 'receptions:'
# # poisson is too narrow, geometric has too heavy of tail
# # neg binomial is not perfect, -logL/N ~ 2. doesn't quite capture shape
# # get p~0.5... coincidence?
# dist_fit.plot_counts( rec_rec, label='receptions', fits=['neg_binomial'] )

print 'total yards:'
# in a single game
dist_fit.plot_counts_poly( all_yds, bounds=(-50,400), label='total yards' )

# there are games with negative yards so counts are not appropriate
# print 'rushing yards:'
# dist_fit.plot_counts( rush_yds, label='rushing yards' )
