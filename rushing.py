#!/usr/bin/env python
from __future__ import division
import dist_fit
import nflgame
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.getLogger().setLevel(logging.DEBUG)
            
rush_att = []
rush_yds_pa = []
rush_tds = []
rush_tds_pa = []
all_td = []
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
            rshatt = pstat.rushing_att
            rshyd_pa = pstat.rushing_yds / rshatt
            rshtd = pstat.rushing_tds
            rshtd_pa = rshtd / rshatt
            # logging.info('rush att, yds, tds: {}, {}, {}'.format(rshatt, rshyd_pa, rshtd_pa))

            rec_tgt = pstat.receiving_tgt
            rec = pstat.receiving_rec
            recyd = pstat.receiving_yds
            rectd = pstat.receiving_td
            tds = pstat.tds
            # print rshtd,rectd,tds
            rush_att.append( rshatt )
            rush_tds.append( rshtd )
            rush_tds_pa.append( rshtd_pa )
            rush_yds_pa.append( rshyd_pa )
            # all_yds.append( rshyd + recyd )
            rec_rec.append( rec )
            all_td.append( tds )


# print 'rushing attempts:'
# # negative binomial does quite well here for single year.
# # note p~0.5 ... more like 0.7 w/ all seasons
# # poisson is under-dispersed.
# # neg bin doesn't do as well w/ all years, but still better than poisson
# # beta-negative binomial should have the extra dispersion to capture this
# dist_fit.plot_counts( rush_att, label='rushing attempts per game' )

# logging.info('exiting early')
# exit(0)

# print 'rushing yards per attempt:'
# # in a single game
# dist_fit.plot_counts_poly( rush_yds_pa, label='rush yds per attempt', bounds=(-10,50))

print 'rush TDs per attempt:'
# negative binomial is redundant w/ poisson here. TDs are rare, and relatively independent.
# geometric does OK, but is clearly inferior w/ high stats
# poisson does quite well even when all years are combined : -logL/N ~ 1
# for a collection of rushers, we should use NB which gets updated to converge to the rusher's poisson w/ infinite data
# dist_fit.plot_counts( all_td, label='touchdowns', fits=['poisson', 'neg_binomial'] )
dist_fit.plot_fraction( rush_tds, rush_att, label='touchdowns per attempt', fits=['beta_binomial'] )

# print 'receptions:'
# # poisson is too narrow, geometric has too heavy of tail
# # neg binomial is not perfect, -logL/N ~ 2. doesn't quite capture shape
# # get p~0.5... coincidence?
# dist_fit.plot_counts( rec_rec, label='receptions', fits=['neg_binomial'] )

# there are games with negative yards so counts are not appropriate
# print 'rushing yards:'
# dist_fit.plot_counts( rush_yds, label='rushing yards' )
