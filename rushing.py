#!/usr/bin/env python

import nflgame

import numpy as np
import matplotlib.pyplot as plt

import dist_fit

rush_att = []
# rush_td = []
all_td = []
rush_yds = []
all_yds = []
rec_rec = []
all_twopta, all_twoptm = [],[] # can be split into receiving/rushing
#  there are other stats: fumbles lost, etc...

fantasy_points = []
# points per
pp_yard = 0.1
pp_reception = 1
pp_td = 6
pp_twopm = 2


# for year in range(2009, 2017):
for year in range(2016, 2017):
    # year = 2009 # options range from 2009 to present
    print 'processing {} season'.format( year )
    games = nflgame.games_gen( year )
    # get game-level stats
    all_players = nflgame.combine_game_stats( games )

    n_top_players = 32
    top_player_names = [ p.player.full_name for p in all_players.rushing().sort('rushing_yds').limit( n_top_players ) ]
    top_playerids = [ p.playerid for p in all_players.rushing().sort('rushing_yds').limit( n_top_players ) ]
    # top_players = [ p.playerid for p in all_players.rushing().sort('rushing_yds').limit( n_top_players ) ]

    # print top_player_names

    for week in range(1,18):
        # print 'looking at week ', week
        weekly_games = nflgame.games_gen( year, week )
        weekly_player_stats = nflgame.combine_game_stats( weekly_games )
        for pstat in weekly_player_stats.rushing().filter( playerid=lambda x: x in top_playerids ):
            # msg = '{} ({}): {} carries for {} yards and {} TDs'.format( pstat, pstat.player.full_name,
            #                                                             pstat.rushing_att, pstat.rushing_yds,
            #                                                             pstat.rushing_td )
            # print msg
            # print dir(pstat) # to check methods
            rshyd = pstat.rushing_yds
            rshtd = pstat.rushing_td
            rec = pstat.receiving_rec
            recyd = pstat.receiving_yds
            rectd = pstat.receiving_td
            tds = pstat.tds
            # print rshtd,rectd,tds
            rush_att.append( pstat.rushing_att )
            # rush_td.append( rshtd )
            # rush_yds.append( rshyd )
            all_yds.append( rshyd + recyd )
            rec_rec.append( rec )
            all_td.append( tds )


# print 'rushing attempts:'
# # negative binomial does quite well here for single year.
# # note p~0.5 ... more like 0.7 w/ all seasons
# # poisson is underdispersed.
# # neg bin doesn't do as well w/ all years, but still better than poisson
# dist_fit.plot_counts( rush_att, label='rushing attempts', fits=['neg_binomial'] )

print 'total TDs:'
# negative binomial is redundant w/ poisson here. TDs are rare, and relatively independent.
# geometric does OK, but is clearly inferior w/ high stats
# poisson does quite well even when all years are combined : -logL/N ~ 1
# for a collection of rushers, we should use NB which gets updated to converge to the rusher's poisson w/ infinite data
dist_fit.plot_counts( all_td, label='touchdowns', fits=['poisson', 'neg_binomial'] )

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
