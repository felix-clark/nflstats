#!/usr/bin/env python3
import dist_fit
import numpy as np
import pandas as pd
import logging

logging.getLogger().setLevel(logging.DEBUG)

rbdf = pd.DataFrame()

firstyear,lastyear = 2009,2017
for year in range(firstyear,lastyear+1):
    yrdf = pd.read_csv('weekly_stats/fantasy_stats_year_{}.csv'.format(year))
    yrdf = yrdf[yrdf['pos'] == 'RB']
    good_col = lambda col: 'passing' not in col and 'kicking' not in col
    columns = [col for col in yrdf.columns if good_col(col)]
    rbdf = rbdf.append(yrdf[columns].fillna(0))


rush_att = rbdf['rushing_att']
rush_yds = rbdf['rushing_yds']
rush_tds = rbdf['rushing_tds']
good_rushers = (rush_att > 4) # need a better way to select players of interest

# print(rbdf[~good_rushers])
# print(rush_att[~good_rushers])

print('rushing attempts:')
# negative binomial does quite well here for single year, but only for top players.
# note p~0.5 ... more like 0.7 w/ all seasons
# poisson is under-dispersed.
# neg bin doesn't do as well w/ all years, but still better than poisson
# beta-negative binomial should have the extra dispersion to capture this
dist_fit.plot_counts( rush_att[good_rushers], label='rushing attempts per game' )

# logging.info('exiting early')
# exit(0)

# print 'rushing yards per attempt:'
# # in a single game
# dist_fit.plot_counts_poly( rush_yds_pa, label='rush yds per attempt', bounds=(-10,50))

print('rush TDs per attempt:')
# negative binomial is redundant w/ poisson here. TDs are rare, and relatively independent.
# geometric does OK, but is clearly inferior w/ high stats
# poisson does quite well even when all years are combined : -logL/N ~ 1
# for a collection of rushers, we should use NB which gets updated to converge to the rusher's poisson w/ infinite data
# dist_fit.plot_counts( all_td, label='touchdowns', fits=['poisson', 'neg_binomial'] )

# this ratio fit doesn't do so well. TDs are farely rare overall, and the alpha/beta parameters tend to blow up in the fit.
# perhaps just a poisson or simple rate would suffice.
dist_fit.plot_fraction( rush_tds[good_rushers], rush_att[good_rushers], label='touchdowns per attempt' )

# print('receptions:')
# # poisson is too narrow, geometric has too heavy of tail
# # neg binomial is not perfect, -logL/N ~ 2. doesn't quite capture shape
# # get p~0.5... coincidence?
# dist_fit.plot_counts( rec_rec, label='receptions', fits=['neg_binomial'] )

# there are games with negative yards so counts are not appropriate
# print 'rushing yards:'
# dist_fit.plot_counts( rush_yds, label='rushing yards' )
