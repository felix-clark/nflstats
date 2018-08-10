#!/usr/bin/env python3
import dist_fit
import numpy as np
import pandas as pd
import logging


def main():
    logging.getLogger().setLevel(logging.DEBUG)

    rbdf = pd.DataFrame()

    good_rb_ids = set()

    firstyear,lastyear = 2009,2017 # 2009 seems to have very limited stats
    for year in range(firstyear,lastyear+1):
        yrdf = pd.read_csv('weekly_stats/fantasy_stats_year_{}.csv'.format(year))
        yrdf = yrdf[yrdf['pos'] == 'RB']
        good_col = lambda col: 'passing' not in col and 'kicking' not in col
        columns = [col for col in yrdf.columns if good_col(col)]
        yrdf = yrdf[columns].fillna(0)
        
        # these are defined pretty much just for debugging
        display_cols = ['name']
        display_cols += [col for col in columns if 'rushing' in col]

        good_rb_names = top_rb_names(year, 16)
        # good_rb_names = good_rb_names.tail(8)
        good_rbs = yrdf['name'].isin(good_rb_names)
        yrdf = yrdf[good_rbs]
        
        # lets select good RBs by those who have more than some # of touches in a season
        # we can't just get IDs though, since some players are only relevant for a season or two
        # season_tot = yrdf.groupby(['name','playerid'], as_index=False).sum() # really just need playerid
        # season_best = yrdf.groupby(['name','playerid'], as_index=False).max()
        # baseline_tot = season_tot['rushing_att'] > 60
        # baseline_best = season_best['rushing_att'] > 20
        # or had at least some games with lots of attempts, to account for injured stars    
        # good_rb_ids |= set(season_tot[baseline_tot]['playerid'].tolist()).union(set(season_best[baseline_best]['playerid'].tolist()))
        
        rbdf = rbdf.append(yrdf)

    # print(good_rb_ids)
    
    rush_att = rbdf['rushing_att']
    rush_yds = rbdf['rushing_yds']
    rush_tds = rbdf['rushing_tds']
    # good_rbs = (rush_att > 4) # need a better way to select players of interest
    good_rbs = rush_att > 0
    # print(rbdf[good_rbs][display_cols])
    
    # print(rbdf[~good_rushers])
    # print(rush_att[~good_rushers])
    
    print('rushing attempts:')
    # negative binomial does quite well here for single year, but only for top players.
    # note p~0.5 ... more like 0.7 w/ all seasons
    # poisson is under-dispersed.
    # neg bin doesn't do as well w/ all years, but still better than poisson
    # beta-negative binomial should have the extra dispersion to capture this
    rush_att_fits = ['neg_binomial'
                     , 'beta_neg_binomial' # beta-negative is not really an improvement - we don't need more variance
    ]
    dist_fit.plot_counts( rush_att[good_rbs], label='rushing attempts per game' ,fits=rush_att_fits)
    
    # logging.info('exiting early')
    # exit(0)
    
    # print 'rushing yards per attempt:'
    # # in a single game
    # dist_fit.plot_counts_poly( rush_yds_pa, label='rush yds per attempt', bounds=(-10,50))
    
    # negative binomial is redundant w/ poisson here. TDs are rare, and relatively independent.
    # geometric does OK, but is clearly inferior w/ high stats
    # poisson does quite well even when all years are combined : -logL/N ~ 1
    # for a collection of rushers, we should use NB which gets updated to converge to the rusher's poisson w/ infinite data
    # dist_fit.plot_counts( all_td, label='touchdowns', fits=['poisson', 'neg_binomial'] )
    
    # this ratio fit doesn't do so well. TDs are farely rare overall, and the alpha/beta parameters tend to blow up in the fit.
    # perhaps just a poisson or simple rate would suffice.
    print('rush TDs per attempt:')
    dist_fit.plot_fraction( rush_tds[good_rbs], rush_att[good_rbs], label='touchdowns per attempt' )
    
    # print('receptions:')
    # # poisson is too narrow, geometric has too heavy of tail
    # # neg binomial is not perfect, -logL/N ~ 2. doesn't quite capture shape
    # # get p~0.5... coincidence?
    # dist_fit.plot_counts( rec_rec, label='receptions', fits=['neg_binomial'] )
    
    # there are games with negative yards so counts are not appropriate
    # print 'rushing yards:'
    # dist_fit.plot_counts( rush_yds, label='rushing yards' )

# returns a list of the top N rbs by ADP in a given year
def top_rb_names(year, n=32):
    topdf = pd.read_csv('adp_historical/adp_rb_{}.csv'.format(year))
    # this should already be sorted
    topdf = topdf.head(n)
    return topdf['name']

if __name__ == '__main__':
    main()
