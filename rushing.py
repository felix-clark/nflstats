#!/usr/bin/env python3
import dist_fit
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.optimize as opt
import itertools
import logging
import os.path
import argparse

from rb_model import *

def main():
    logging.getLogger().setLevel(logging.DEBUG)
    sns.set()

    all_models = ['rush_att', 'rush_yds', 'rush_tds']
    
    parser = argparse.ArgumentParser(description='optimize and analyze bayesian models')
    parser.add_argument('--opt-hyper',nargs='?',type=str,choices=all_models,help='try to improve hyperparameters for this model')

    args = parser.parse_args()
    
    if args.opt_hyper:
        hps = minimize_model_hyperparameters(args.opt_hyper)
    
    rbdf = get_model_df()
    rbdf = rbdf[rbdf['week'] < 17].dropna() # somehow dropna=True doesn't remove these
    
    models = ['rush_att', 'rush_yds', 'rush_tds'] # edit this to suppress info we've already looked at
    # models = ['rush_yds'] # edit this to suppress info we've already looked at
    for model in models:
        klds = rbdf[model+'_kld']
        chisqs = rbdf[model+'_chisq']
        logging.info('{} kld = {:.6f}, chisq = {:.4f} out of {} weeks (avg {:.6f}, {:.3f})'
                     .format(model, klds.sum(), chisqs.sum(),
                             klds.size, klds.mean(), chisqs.mean()))
        # print(rbdf['{}_chisq'.format(model)].mean()) # yes, this gives the same result
        plot_vars = ['kld', 'res']
        for pname in plot_vars:
            plt.figure()
            year_plt = sns.boxenplot(data=rbdf, x='career_year', y=model+'_'+pname) # hue = model # when we compare models (baseline would be nice)
        plt.show(block=True)

        
    for model in models:
        residuals = rbdf['{}_res'.format(model)]
        residuals = residuals[residuals.notnull()]
        plt_res = sns.distplot(residuals,
                               hist_kws={'log':False, 'align':'left'})
        # plt_rar.figure.savefig('rush_att_res')
        plt_res.figure.show()

    plt.show(block=True)
    exit(0)

    ra_bins = np.arange(0,45,10)
    for low,up in zip(ra_bins[:-1], ra_bins[1:]):
        res = rbdf[(low <= rbdf['rushing_att']) & (rbdf['rushing_att'] < up)]['rush_yds_res']
        # res = rbdf[(low <= rbdf['rushing_att']) & (rbdf['rushing_att'] < up)]['rush_tds_kld']
        # res = rbdf[(low <= rbdf['rushing_att']) & (rbdf['rushing_att'] < up)]['rush_att_kld']
        res = res[res.notnull()]
        plt_res = sns.distplot(res,
                               hist_kws={'log':False, 'align':'left'})
        # plt_rar.figure.savefig('rush_att_res')
        plt_res.figure.show()
    plt.show(block=True)
    
    
    # print (rbdf.columns)
    # plt.figure()
    
    # plt_corr = sns.pairplot(resdf, #height = 4,
    #                         vars=['rushing_att', 'rush_att_res', 'rush_att_chisq', 'rush_att_kld'],
    #                         dropna=True,
    #                         kind='reg' # do linear regression to look for correlations
    # )
    # plt.show(block=True)
    # plt_corr = sns.pairplot(resdf, #height = 4,
    #                         vars=['rushing_att', 'rushing_yds', 'rush_yds_res', 'rush_yds_kld', 'rushing_tds'],
    #                         dropna=True,
    #                         kind='reg' # do linear regression to look for correlations
    # )
    # plt.show(block=True)

    # resnames = ['{}_res'.format(m) for m in models]
    # plt_corr = sns.pairplot(resdf, height = 4,
    #                         vars=resnames,
    #                         kind='reg' # do linear regression to look for correlations
    # )        
    # plt.show(block=True)
        
    # pd.options.display.max_rows=10    
    # ra_chisqs = rbdf.groupby('playerid')['rush_att_chisq'].mean()
    # print(ra_chisqs[ra_chisqs > 1.5].index)
    # print( rbdf[rbdf['playerid'].isin(ra_chisqs[ra_chisqs > 1.5].index)][['name','year','week','rushing_att','rush_att_ev','rush_att_chisq']])
    
    # print(rbdf.groupby('year')['rush_att_chisq'].mean())    
            
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
    
    # print('rushing attempts:')
    # # negative binomial does quite well here for single year, but only for top players.
    # # note p~0.5 ... more like 0.7 w/ all seasons
    # # poisson is under-dispersed.
    # # neg bin doesn't do as well w/ all years, but still better than poisson
    # # beta-negative binomial should have the extra dispersion to capture this
    # rush_att_fits = ['neg_binomial'
    #                  , 'beta_neg_binomial' # beta-negative is not really an improvement - we don't need more variance
    # ]
    # dist_fit.plot_counts( rush_att[good_rbs], label='rushing attempts per game' ,fits=rush_att_fits)

    # print('rushing yards per attempt:')
    # # in a single game
    # dist_fit.plot_avg_per( rush_yds[good_rbs]/rush_att[good_rbs], weights=rush_att[good_rbs],
    #                        label='rush yds per attempt', bounds=(-10,50))
    
    # negative binomial is redundant w/ poisson here. TDs are rare, and relatively independent.
    # geometric does OK, but is clearly inferior w/ high stats
    # poisson does quite well even when all years are combined : -logL/N ~ 1
    # for a collection of rushers, we should use NB which gets updated to converge to the rusher's poisson w/ infinite data
    # dist_fit.plot_counts( all_td, label='touchdowns', fits=['poisson', 'neg_binomial'] )
    
    # # the direct ratio fit doesn't do so well. TDs are farely rare overall, and the alpha/beta parameters tend to blow up in the fit.
    # # perhaps just a poisson or simple rate would suffice.
    # print('rush TDs per attempt:')
    # dist_fit.plot_fraction( rush_tds[good_rbs], rush_att[good_rbs], label='touchdowns per attempt' )
    
    # print('receptions:')
    # # poisson is too narrow, geometric has too heavy of tail
    # # neg binomial is not perfect, -logL/N ~ 2. doesn't quite capture shape
    # # get p~0.5... coincidence?
    # dist_fit.plot_counts( rec_rec, label='receptions', fits=['neg_binomial'] )
    
    # there are games with negative yards so counts are not appropriate
    # print 'rushing yards:'
    # dist_fit.plot_counts( rush_yds, label='rushing yards' )

def get_rb_df(fname = None):
    if fname is None:
        fname = 'data_rb_cache.csv'
    if os.path.isfile(fname):
        return pd.read_csv(fname, index_col=0)
    logging.info('will compile and cache RB data')
    rbdf = pd.DataFrame()

    firstyear,lastyear = 2009,2017 # 2009 seems to have very limited stats
    for year in range(firstyear,lastyear+1):
        yrdf = pd.read_csv('weekly_stats/fantasy_stats_year_{}.csv'.format(year), index_col=0)
        # filtering by position alone rules out e.g. Fred Jackson in 2009 because of an error in the data
        yrdf = yrdf[(yrdf['pos'] == 'RB') | (yrdf['rushing_att'] > 100)]
        good_col = lambda col: 'passing' not in col and 'kicking' not in col
        columns = [col for col in yrdf.columns if good_col(col)]
        yrdf = yrdf[columns].fillna(0)
        yrdf['year'] = year
        # could also provide a weight based on ADP?
        good_rb_names = top_rb_names(year)
        # good_rb_names = good_rb_names.tail(8) # we got the defaults from the top 16. we'll refine them later.
        # to be included each week, they need to have been a good RB and also have actually played:
        good_rbs = yrdf['name'].isin(good_rb_names) & (yrdf['rushing_att'] > 0)
        yrdf = yrdf[good_rbs]        
        rbdf = rbdf.append(yrdf)
    
    # they should already be sorted properly, but let's check.
    rbdf = rbdf.sort_values(['year', 'week']).reset_index(drop=True)
    logging.info('saving relevant RB data to {}'.format(fname))
    rbdf.to_csv(fname)
    return rbdf

    
def get_model_df( pos='RB', fname = None):
    if fname is None:
        fname = 'model_cache_{}.csv'.format(pos.lower())
    if os.path.isfile(fname):
        return pd.read_csv(fname)

    rbdf = get_rb_df()    
    playerids = rbdf['playerid'].unique()
    years = rbdf['year'].unique()    

    # basing rush attempts on the past is not so great.
    # ideally we use a team-based touch model.
    # we need to look into the discrepancies more to figure out the problems
    # i suspect injuries, trades, then matchups are the big ones.
    # many mistakes are in week 17, too, where the starters are often different

    model_defs = {
        'rush_att':RushAttModel,
        'rush_yds':RushYdsModel,
        'rush_tds':RushTdModel,
    }
    tot_week = 0
    
    for pid in playerids:
        pdf = rbdf[rbdf['playerid'] == pid]
        pname = pdf['name'].unique()[0]

        plmodels = {mname:mod.for_position(pos)
                    for mname,mod in model_defs.items()}
        
        years = pdf['year'].unique()
        # we could skip single-year seasons
        for icareer,year in enumerate(years):
            # can we use "group by" or something to reduce the depth of these loops?
            ypdf = pdf[(pdf['year'] == year) & (pdf['week'] < 17)] # week 17 is often funky
            for index, row in ypdf.iterrows():
                # these do point to the same one:
                # assert((row[['name', 'year', 'week']] == rbdf.loc[index][['name', 'year', 'week']]).all())
                tot_week += 1
                rbdf.loc[index,'career_year'] = icareer+1
                for mname,model in plmodels.items():
                    mvars = [row[v] for v in model.var_names]
                    kld = model.kld(*mvars)
                    chisq = model.chi_sq(*mvars)
                    data = row[model.pred_var]
                    # saving to the dataframe slows the process down significantly
                    depvars = [row[v] for v in model.dep_vars]
                    ev = model.ev(*depvars)
                    var = model.var(*depvars)
                    res = (data-ev)/np.sqrt(var)
                    # res = (data-ev)/model.scale(*depvars)
                    # if mname == 'rush_tds':
                    #     print('rush_att, ev, var, res = {}, {}, {}, {}'.format(row['rushing_att'], ev, var, res))
                    rbdf.loc[index,'{}_ev'.format(mname)] = ev
                    rbdf.loc[index,'{}_res'.format(mname)] = res
                    rbdf.loc[index,'{}_kld'.format(mname)] = kld
                    rbdf.loc[index,'{}_chisq'.format(mname)] = chisq
                    # if kld > 15:
                    #     logging.warning('large KL divergence: {}'.format(kld))
                    #     logging.warning('{} = {}'.format(model.pred_var,data))
                    #     logging.warning(model)
                    #     logging.warning(rbdf.loc[index])
                    model.update_game(*mvars) # it's important that this is done last, after computing KLD and chi^2
            for _,mod in plmodels.items():
                mod.new_season()
                
        # logging.info('after {} year career, {} is modeled by:'.format(len(years), pname))
        # logging.info('  {}'.format(plmodels['rush_yds']))
        
    rbdf.to_csv(fname)
    return rbdf

def minimize_model_hyperparameters(model_name='rush_att'):
    logging.info('will search for good hyperparameters for {}'.format(model_name))
    rbdf = get_rb_df()    
    playerids = rbdf['playerid'].unique()
    years = rbdf['year'].unique()    

    model_defs = {
        'rush_att':{
            'model':RushAttModel,
            'start_pars':(
                2.807, # alpha0
                0.244, # beta0
                0.121, # lr
                0.677, # mem
                0.782, # gmem
            ),
            'par_bounds':[
                (0,None),
                (0,None),
                (0.,1.0),
                (0.1,1.0),
                (0.5,1.0),
            ],
            
        },
        'rush_yds':{
            'model':RushYdsModel,
            'start_pars':( # now w/ a version that scales the variance for df
                122.26, # mu*nu
                36.39, # nu
                8.87, # alpha
                40.09, # beta 
                0.00237, # mean learn rate
                0.0239, # learn rate for variance # small values w/ full memory 
                0.81, # skew # with a lower mu, a higher skew can be used
                # 1.0, # seasonal memory decay for munu/nu
                0.613 # seasonal memory decay, for alpha/beta
                # 1.0,1.0, # game mem
            ),
            # 'start_pars':( # now w/ a version that does not scale alpha and beta learning by rush_att
            #     116.30, # mu*nu
            #     43.77, # nu
            #     5.54, # alpha
            #     12.80, # beta (stddev = 1.78)
            #     0.003187, # mean learn rate
            #     8.87e-5, # learn rate for variance # small values w/ full memory 
            #     2.026, # skew # with a lower mu, a higher skew can be used
            #     # 1.0, # seasonal memory decay for munu/nu
            #     0.867 # seasonal memory decay, for alpha/beta
            #     # 1.0,1.0, # game mem
            # ),
            'par_bounds':[
                (0,None),(0,None),(0,None),(0,None),
                (1e-6,1.0),
                (0.0,1.0),
                (0.1,10.0), # skew
                (0.5,1.0),
                (0.5,1.0),
            ],
        },
        'rush_tds':{ # this default one has very poor residuals, or there is some other problem
            'model':RushTdModel,
            'start_pars':(
                19.07,
                684.7, # ab0
                1.84, # lr
                0.775, # mem
                1.0, # gmem
            ),
            'par_bounds':[
                (0,None),(0,None),
                (0.1,10.0),# let's uncap learn rate
                (0.2,1.0),
                (0.5,1.0)
                ]
        }
    }
    mdef = model_defs[model_name]
    mdtype = mdef['model']
    pars0 = mdef['start_pars']
    logging.info('starting with parameters {}'.format(pars0))

    def tot_kld(hparams):
        tot_week = 0
        tot_kld = 0
        for pid in playerids:
            pdf = rbdf[rbdf['playerid'] == pid]
            pname = pdf['name'].unique()[0]
            plmodel = mdtype(*hparams)
            years = pdf['year'].unique()
            # we could skip single-year seasons
            for icareer,year in enumerate(years):
                # can we use "group by" or something to reduce the depth of these loops?
                ypdf = pdf[(pdf['year'] == year) & (pdf['week'] < 17)] # week 17 is often funky
                for index, row in ypdf.iterrows():
                    tot_week += 1
                    mvars = [row[v] for v in plmodel.var_names]
                    kld = plmodel.kld(*mvars)
                    tot_kld += kld
                    # it's important that updating is done after computing KLD
                    plmodel.update_game(*mvars)
                plmodel.new_season()
        print('kld = {}'.format(tot_kld))
        return tot_kld

    minned = opt.minimize(tot_kld, x0=pars0,
                          # method='Nelder-Mead', # N-M can't deal w/ bounds
                          bounds=mdef['par_bounds'],
                          # it'll take several iterations to start going in the right direction w/ the default algorithm
                          # tho there are several function calls per "iteration" w/ default
                          options={'maxiter':32,
                                   # 'ftol':1e-12 # seems like a waste
                          })
    print(minned)
    minpars = minned.x
    print(mdtype(*minpars))
    return minpars

# returns a list of the top N rbs by ADP in a given year
def top_rb_names(year, nadp=32, pos='RB', n1=4, n2=12):
    """
    nadp: top number to choose from ADP
    n1: number of times as a POS1 *on their team* to trigger inclusion
    n2: number of times as a POS2 on their team to be included
    """
    
    adpfname = 'adp_historical/adp_{}_{}.csv'.format(pos.lower(), year)
    if not os.path.isfile(adpfname):
        logging.error('cannot find ADP file. try running get_historical_adp.py')
    topdf = pd.read_csv(adpfname)
    # this should already be sorted
    topdf = topdf.head(nadp)
    topnames = topdf['name']
    
    if pos != 'RB': # we might just want a different function for each position
        raise NotImplementedError
    
    # note that some of the fantasy data is incorrect, e.g. 2009 Fred Jackson is listed as a WR
    # toppldf = pd.DataFrame(['name', 'playerid', 'year', pos+'1', pos+'2']) # will hold how many times each player was an RB1
    # print (toppldf)

    yrdf = pd.read_csv('weekly_stats/fantasy_stats_year_{}.csv'.format(year))
    # yrdf = yrdf[yrdf['pos'] == pos] # some of the position data is incorrect, so just rank by rush attempts
    # manually remove QBs from here
    yrdf = yrdf[(yrdf['rushing_att'] > 4) & (yrdf['pos'] != 'QB')]
    teams = yrdf['team'].unique()
    weeks = yrdf['week'].unique()
    top1,top2 = {},{}
    for week,team in itertools.product(weeks, teams):
        relpos = yrdf[(yrdf['week'] == week) & (yrdf['team'] == team)].sort_values('rushing_att', ascending=False)
        if len(relpos) == 0: continue # then it's probably a bye week
        toprsh = relpos.iloc[0]
        if toprsh['rushing_yds'] >= 10:
            t1name = toprsh['name']
            top1[t1name] = top1.get(t1name, 0) + 1
            top2[t1name] = top2.get(t1name, 0) + 1
        if len(relpos) < 2: continue
        toprsh = relpos.iloc[1]
        if toprsh['rushing_yds'] >= 10:
            t2name = toprsh['name']
            top2[t2name] = top2.get(t2name, 0) + 1

    pretop = topnames.copy()
    topnames = topnames.append(pd.Series([name for name,ntop in top1.items() if ntop >= n1]))
    topnames = topnames.append(pd.Series([name for name,ntop in top2.items() if ntop >= n2]))

    topnames = topnames.unique()
    return topnames

if __name__ == '__main__':
    main()
