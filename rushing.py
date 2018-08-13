#!/usr/bin/env python3
import dist_fit
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import os.path

from rb_model import *

def main():
    logging.getLogger().setLevel(logging.DEBUG)
    sns.set()
    
    rbdf = get_model_df()
    # models = ['rush_att', 'rush_yds', 'rush_tds'] # edit this to suppress info we've already looked at
    models = ['rush_att', 'rush_yds'] # edit this to suppress info we've already looked at
    for model in models:
        klds = rbdf[model+'_kld']
        chisqs = rbdf[model+'_chisq']
        logging.info('{} kld = {:.6f}, chisq = {:.4f} out of {} weeks (avg {:.6f}, {:.3f})'
                     .format(model, klds.sum(), chisqs.sum(),
                             klds.size, klds.mean(), chisqs.mean()))
        weights = rbdf['rushing_att'] / rbdf['rushing_att'].mean()
        wklds = weights*klds
        logging.info('weighted kld = {:6f}, (avg {})'.format(wklds.sum(), wklds.mean()))
        # print(rbdf['{}_chisq'.format(model)].mean()) # yes, this gives the same result

    # for model in models:
    #     residuals = rbdf['{}_res'.format(model)]
    #     residuals = residuals[residuals.notnull()]
    #     plt_res = sns.distplot(residuals,
    #                            hist_kws={'log':False, 'align':'left'})
    #     # plt_rar.figure.savefig('rush_att_res')
    #     plt_res.figure.show()

    # exit(0)

    # ra_bins = np.arange(0,45,10)
    # for low,up in zip(ra_bins[:-1], ra_bins[1:]):
    #     res = rbdf[(low <= rbdf['rushing_att']) & (rbdf['rushing_att'] < up)]['rush_yds_res']
    #     # res = rbdf[(low <= rbdf['rushing_att']) & (rbdf['rushing_att'] < up)]['rush_tds_kld']
    #     # res = rbdf[(low <= rbdf['rushing_att']) & (rbdf['rushing_att'] < up)]['rush_att_kld']
    #     res = res[res.notnull()]
    #     plt_res = sns.distplot(res,
    #                            hist_kws={'log':False, 'align':'left'})
    #     # plt_rar.figure.savefig('rush_att_res')
    #     plt_res.figure.show()
    # plt.show(block=True)
    
    
    # print (rbdf.columns)
    # plt.figure()
    
    resdf = rbdf[rbdf['week'] < 17].dropna() # somehow dropna=True doesn't remove these
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

    print('rushing yards per attempt:')
    # in a single game
    dist_fit.plot_avg_per( rush_yds[good_rbs]/rush_att[good_rbs], weights=rush_att[good_rbs],
                           label='rush yds per attempt', bounds=(-10,50))
    
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

def get_model_df(fname = 'rush_model_cache.csv', savemodels=True):
    if os.path.isfile(fname):
        return pd.read_csv(fname)
    
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
    rbdf = rbdf.sort_values(['year', 'week']).reset_index(drop=True)
    playerids = rbdf['playerid'].unique()

    years = rbdf['year'].unique()    

    # basing rush attempts on the past is not so great.
    # ideally we use a team-based touch model.
    # we need to look into the discrepancies more to figure out the problems
    # i suspect injuries, trades, then matchups are the big ones.
    # many mistakes are in week 17, too, where the starters are often different

    mu0,nu0,a0 = (4.15,12.,2.)
    b0 = 2.9*nu0*a0/(nu0+1)
    model_defs = {
        'rush_att':{
            'model':RushAttModel,
            'args':(
                0.135, # lr
                0.66, # mem
                0.8, # gmem
                (6.392, 0.4694) # ab0
            )
        },
        'rush_yds':{
            'model':RushYdsModel,
            'args':(
                (0.04,0.002), # learn rate
                0.83, # mem (season)
                0.99, # game mem
                1.15*0, # skew # possibly not implemented correctly - any nonzero value worsens KLD
                (mu0*nu0, nu0, a0, b0) # beta (stddev = 1.78)
            )
        },
        # 'rush_tds':{ # this default one has very poor residuals, or there is some other problem
        #     'model':RushTdModel,
        #     'args':(
        #         1.0, # lr
        #         0.77, # mem
        #         1.0, # gmem
        #         (42., 1400.) # ab0
        #     )
        # }
    }
    tot_week = 0
    
    for pid in playerids:
        pdf = rbdf[rbdf['playerid'] == pid]
        pname = pdf['name'].unique()[0]

        plmodels = {mname:mdef['model'](*mdef['args'])
                    for mname,mdef in model_defs.items()}
        
        years = pdf['year'].unique()
        # we could skip single-year seasons
        for icareer,year in enumerate(years):
            # can we use "group by" or something to reduce the depth of these loops?
            ypdf = pdf[(pdf['year'] == year) & (pdf['week'] < 17)] # week 17 is often funky
            for index, row in ypdf.iterrows():
                # these do point to the same one:
                # assert((row[['name', 'year', 'week']] == rbdf.loc[index][['name', 'year', 'week']]).all())
                tot_week += 1
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
                    if savemodels:
                        rbdf.loc[index,'{}_ev'.format(mname)] = ev
                        rbdf.loc[index,'{}_res'.format(mname)] = res
                        rbdf.loc[index,'{}_kld'.format(mname)] = kld
                        rbdf.loc[index,'{}_chisq'.format(mname)] = chisq
                    if kld > 14:
                        logging.warning('large KL divergence: {}'.format(kld))
                        logging.warning('{} = {}'.format(model.pred_var,data))
                        logging.warning(model)
                        logging.warning(rbdf.loc[index])
                        if pname == 'Ray Rice':
                            exit(1)
                    model.update_game(*mvars) # it's important that this is done last, after computing KLD and chi^2
            for _,mod in plmodels.items():
                mod.new_season()
                
        # logging.info('after {} year career, {} is modeled by:'.format(len(years), pname))
        # logging.info('  {}'.format(plmodels['rush_yds']))
        
    for mname,mdict in model_defs.items():
        logging.info('{} model arguments: {}'.format(mname, mdict['args']))
    rbdf.to_csv(fname)
    return rbdf


# returns a list of the top N rbs by ADP in a given year
def top_rb_names(year, n=32):
    topdf = pd.read_csv('adp_historical/adp_rb_{}.csv'.format(year))
    # this should already be sorted
    topdf = topdf.head(n)
    return topdf['name']

if __name__ == '__main__':
    main()
