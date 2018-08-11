#!/usr/bin/env python3
import dist_fit
import numpy as np
import pandas as pd
import logging

from rb_model import *

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
        yrdf['year'] = year
        
        # # these are defined pretty much just for debugging
        # display_cols = ['name']
        # display_cols += [col for col in columns if 'rushing' in col]

        # TODO: select RB1 and RB2/3 each week for each team. use this designation to set bayesian priors.
        # sould be more precise than this random ADP site, which has a few strange results
        
        # could also provide a weight based on ADP?
        good_rb_names = top_rb_names(year)
        # good_rb_names = good_rb_names.tail(8) # we got the defaults from the top 16. we'll refine them later.
        # to be included each week, they need to have been a good RB and also have actually played:
        good_rbs = yrdf['name'].isin(good_rb_names) & (yrdf['rushing_att'] > 0)
        yrdf = yrdf[good_rbs]
        
        rbdf = rbdf.append(yrdf)

    # they should already be sorted properly, but let's check.
    rbdf = rbdf.sort_values(['year', 'week'])
    playerids = rbdf['playerid'].unique()

    years = rbdf['year'].unique()
    
    tot_kld = 0.
    # tot_kld_stoch = 0.
    tot_week = 0
    kld_dict,week_dict = {},{}

    # basing rush attempts on the past is actually pretty bad.
    # ideally we use a team-based touch model.
    # we need to look into the discrepancies more to figure out the problems
    # i suspect injuries, trades, then matchups are the big ones.
    ra_mod_args = (
        0.132, # lr
        0.651, # mem
        0.812, # gmem
        (6.391, 0.4695) # ab0
    )
    rtd_mod_args = (
        1.0, # lr
        0.77, # mem
        1.0, # gmem
        (42., 1400.) # ab0
    )
    for pid in playerids:
        pdf = rbdf[rbdf['playerid'] == pid]
        pname = pdf['name'].unique()[0]
        
        ra_mod  = RushAttModel(*ra_mod_args)
        # stochastic models are not trivial to get working since the parameters can easily jump to invalid values
        # ra_stoch_mod = RushAttStochModel(*ra_stoch_mod_args)
        rtd_mod = RushTdModel(*rtd_mod_args)
        models = [ra_mod,
                  # ra_stoch_mod,
                  rtd_mod]
        
        years = pdf['year'].unique()
        # we could skip single-year seasons
        for icareer,year in enumerate(years):
            # can we use "group by" or something to reduce the depth of these loops?
            ypdf = pdf[pdf['year'] == year]
            for index, week in ypdf.iterrows():
                ra = week['rushing_att']
                rtd = week['rushing_tds']
                # kld = ra_mod.kld(ra)
                kld = rtd_mod.kld(rtd, ra)
                tot_kld += kld
                tot_week += 1
                # if kld > 5:
                #     logging.info('bad prediction ({}) in year {} week {}: for {}: {}'.format(ra_mod, year, week['week'], pname, ra))
                ra_mod.update_game(ra)
                # ra_stoch_mod.update_game(ra)
                rtd_mod.update_game(rtd, ra)
            for model in models:
                model.new_season()
                
        # logging.info('after {} year career, {} is modeled by:'.format(len(years), pname))
        # logging.info('  {}'.format(ra_mod))
        # logging.info('  {}'.format(rtd_mod))

    logging.info('rush att model arguments: {}'.format(ra_mod_args))
    # logging.info('rush td model arguments: {}'.format(rtd_mod_args))
    logging.info('rush att kld = {} out of {} weeks (avg {:.6f})'.format(tot_kld, tot_week, tot_kld/tot_week))
    # logging.info('rush td kld = {} out of {} weeks (avg {:.7f})'.format(tot_kld, tot_week, tot_kld/tot_week))
    # logging.info('rush att stochastic model arguments: {}'.format(ra_stoch_mod_args))
    # logging.info('rush att stochastic kld = {} (avg {:.6f})'.format(tot_kld_stoch, tot_kld_stoch/tot_week))
            
    
    # logging.warning('exiting early')
    # exit(0)
    
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

    print('rushing yards per attempt:')
    # in a single game
    dist_fit.plot_avg_per( rush_yds[good_rbs]/rush_att[good_rbs], weights=rush_att[good_rbs],
                           label='rush yds per attempt', bounds=(-10,50))
    
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
