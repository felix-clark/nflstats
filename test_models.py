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

from playermodels.rb import *

def main():
    logging.getLogger().setLevel(logging.DEBUG)
    sns.set()

    all_models = ['rush_att', 'rush_yds', 'rush_tds']
    
    parser = argparse.ArgumentParser(description='optimize and analyze bayesian models')
    parser.add_argument('position',type=str,choices=['RB', 'QB', 'WR', 'TE'],help='which position to analyze')
    parser.add_argument('--opt-hyper',nargs='?',type=str,choices=all_models,help='try to improve hyperparameters for this model')

    args = parser.parse_args()

    position = args.position
    logging.info('working with {}s'.format(position))
    
    if args.opt_hyper:
        hps = minimize_model_hyperparameters(position, args.opt_hyper)
    
    posdf = get_model_df(position)
    posdf = posdf[posdf['week'] < 17]# .dropna() # don't necessarily remove nans; we need these for QBs

    # models = ['rush_att', 'rush_yds', 'rush_tds'] # edit this to suppress info we've already looked at
    # models = ['rush_yds'] # edit this to suppress info we've already looked at
    models = []
    for model in models:
        klds = posdf[model+'_kld']
        chisqs = posdf[model+'_chisq']
        logging.info('{} kld = {:.6f}, chisq = {:.4f} out of {} weeks (avg {:.6f}, {:.3f})'
                     .format(model, klds.sum(), chisqs.sum(),
                             klds.size, klds.mean(), chisqs.mean()))
        # print(posdf['{}_chisq'.format(model)].mean()) # yes, this gives the same result
        plot_vars = ['kld', 'cdf']
        for pname in plot_vars:
            pass
            # plt.figure()
            # year_plt = sns.boxenplot(data=posdf, x='career_year', y=model+'_'+pname) # hue = model # when we compare models (baseline would be nice)
            # year_plt = sns.lvplot(data=posdf, x='career_year', y=model+'_'+pname) # hue = model # when we compare models (baseline would be nice)
        # plt.show(block=True)

    # print(posdf[posdf['rushing_att'] == 0])
    # TODO: we should be able to split the dataset in half randomly and see flat CDFs in both samples
    # pltvar = sns.distplot(posdf['rush_yds_cdf'])
    # pltvar.figure.show()
    # plt.show(block=True)
    # cdf_plt = sns.pairplot(posdf, #height = 4,
    #                        vars=['rush_yds_cdf', 'rushing_att', 'rushing_yds'],
    #                        hue='career_year',
    # )
    # plt.show(block=True)
    
    # the pearson correlation of the CDFs should be the spearman correlation of the data. (though it's not really)
    # note that our models for yards are *given* the attempt, so when parameterizing them
    # we should use the correlations in these cdfs.
    # we should probably use the raw spearman correlation from the data
    corrdf = posdf[['rushing_att', 'rushing_yds', 'rushing_tds', 'rush_att_cdf', 'rush_yds_cdf', 'rush_tds_cdf']].copy()
    corrdf['rushing_ypa'] = corrdf['rushing_yds'] / corrdf['rushing_att']
    corrdf['rushing_tdpa'] = corrdf['rushing_tds'] / corrdf['rushing_att']
    rush_corr = corrdf[['rushing_att', 'rushing_ypa', 'rushing_tdpa']].corr(method='spearman')
    print(rush_corr)
    # # spearman and pearson are quite similar for the cdf, at least when the models are working decently
    # rush_corr = corrdf[['rush_att_cdf', 'rush_yds_cdf', 'rush_tds_cdf']].corr(method='spearman')
    # print(rush_corr)
    
    plt_corr = sns.pairplot(corrdf, #height = 4,
                            vars=['rush_att_cdf', 'rush_yds_cdf', 'rush_tds_cdf'],
                            dropna=True,
                            # kind='reg', # do linear regression to look for correlations
                            # hue='career_year'
    )
    plt.show(block=True)

    exit(1)
        
    for model in models:
        cdf = posdf['{}_cdf'.format(model)]
        # cdf = cdf[cdf.notna()] # drops zeros?
        plt_cdf = sns.distplot(cdf,
                               hist_kws={'log':False, 'align':'left'})
        # plt_cdf.figure.savefig('rush_att_res')
        plt_cdf.figure.show()
    plt.show(block=True)

    # resnames = ['{}_cdf'.format(m) for m in models] # we could take the ppf of this to look at standardized residuals
    # plt_corr = sns.pairplot(resdf, height = 4,
    #                         vars=resnames,
    #                         kind='reg' # do linear regression to look for correlations
    # )        
    # plt.show(block=True)
        
    # logging.warning('exiting early')
    # exit(0)
    
    good_pos = True
    if position == 'RB': good_pos = posdf['rushing_att'] > 0
    if position == 'QB': good_pos = posdf['passing_cmp'] > 0
    if position in ['WR', 'TE']:
         # not clear we should filter these. include zero for now
        good_pos = posdf['receiving_rec'] >= 0
    rec_rec = posdf[good_pos]['receiving_rec']
    
    # print(posdf[~good_rushers])
    # print(rush_att[~good_rushers])
    
    # print('rushing attempts:')
    print('receptions')
    # negative binomial does quite well here for single year, but only for top players.
    # note p~0.5 ... more like 0.7 w/ all seasons
    # poisson is under-dispersed.
    # neg bin doesn't do as well w/ all years, but still better than poisson
    # beta-negative binomial should have the extra dispersion to capture this
    rush_att_fits = [
        'geometric',
        'poisson',
        'neg_binomial',
        'beta_neg_binomial' # beta-negative does well overall for QBs (accounting for extra variance)
    ]
    # dist_fit.plot_counts( rush_att[good_pos], label='rushing attempts per game' ,fits=rush_att_fits)
    dist_fit.plot_counts( rec_rec, label='receptions per game' ,fits=rush_att_fits)
    print((rec_rec == 0).any())

    # print('rushing yards per attempt:')
    # # in a single game
    # dist_fit.plot_avg_per( rush_yds[good_rbs]/rush_att[good_rbs], weights=rush_att[good_rbs],
    #                        label='rush yds per attempt', bounds=(-10,50))
    
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

    pass
    
def get_pos_df(pos, fname = None):
    pos = pos.upper()
    if fname is None:
        fname = 'data_{}_cache.csv'.format(pos.lower())
    if os.path.isfile(fname):
        return pd.read_csv(fname, index_col=0)
    logging.info('will compile and cache {} data'.format(pos))
    rbdf = pd.DataFrame()

    
    adpfunc = None
    if pos == 'QB': adpfunc = top_qb_names
    if pos == 'RB': adpfunc = top_rb_names
    if pos == 'WR': adpfunc = top_wr_names
    if pos == 'TE': adpfunc = top_te_names
    # ...
    
    firstyear,lastyear = 2009,2017 # 2009 seems to have very limited stats
    for year in range(firstyear,lastyear+1):
        yrdf = pd.read_csv('weekly_stats/fantasy_stats_year_{}.csv'.format(year), index_col=0)
        
        mask = None
        # filtering by position alone rules out e.g. Fred Jackson in 2009 because of an error in the data
        # ... but this data is weekly. it'd be more messiness to correct that here.
        # if pos == 'RB': mask = (yrdf['pos'] == 'RB') | (yrdf['rushing_att'] > 100) .. else
        mask = (yrdf['pos'] == pos) # we may have to use more custom workarounds
        # to be included each week, they need to have been a good [RB] and also have actually played:
        if pos == 'QB': mask &= (yrdf['passing_cmp'] > 0)
        if pos == 'RB': mask &= (yrdf['rushing_att'] > 0)
        if pos in ['WR', 'TE']: mask &= ((yrdf['receiving_rec'] > 0) | (yrdf['rushing_att'] > 0))
        if pos == 'K': mask &= ((yrdf['kicking_xpa'] > 0) | (yrdf['kicking_fga'] > 0))
        yrdf = yrdf[mask]
        
        # we won't try to model passing for non-QBs; it's too rare to be meaningful
        # similarly we won't track QB receptions
        good_col = None
        if pos == 'QB': good_col = lambda col: 'receiving' not in col and 'kicking' not in col
        if pos == 'K': good_col = lambda col: 'passing' not in col and 'rushing' not in col and 'receiving' not in col
        if pos in ['RB', 'WR', 'TE']: good_col = lambda col: 'passing' not in col and 'kicking' not in col
        columns = [col for col in yrdf.columns if good_col(col)]
        yrdf = yrdf[columns].fillna(0)
        
        yrdf['year'] = year
        # could also provide a weight based on ADP or production?
        good_pos_names = adpfunc(year)
        good_pos = yrdf['name'].isin(good_pos_names)
        yrdf = yrdf[good_pos]
        rbdf = rbdf.append(yrdf)
    
    # they should already be sorted properly, but let's check.
    rbdf = rbdf.sort_values(['year', 'week']).reset_index(drop=True)
    logging.info('saving relevant {} data to {}'.format(pos, fname))
    rbdf.to_csv(fname)
    return rbdf

    
def get_model_df( pos='RB', fname = None):
    if fname is None:
        fname = 'model_cache_{}.csv'.format(pos.lower())
    if os.path.isfile(fname):
        return pd.read_csv(fname)

    rbdf = get_pos_df(pos)
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
                    cdf = model.cdf(*mvars) # standardized to look like a gaussian
                    # res = (data-ev)/np.sqrt(var)
                    rbdf.loc[index,'{}_ev'.format(mname)] = ev
                    rbdf.loc[index,'{}_cdf'.format(mname)] = cdf
                    rbdf.loc[index,'{}_kld'.format(mname)] = kld
                    rbdf.loc[index,'{}_chisq'.format(mname)] = chisq
                    model.update_game(*mvars) # it's important that this is done last, after computing KLD and chi^2
            for _,mod in plmodels.items():
                mod.new_season()
                
        # logging.info('after {} year career, {} is modeled by:'.format(len(years), pname))
        # logging.info('  {}'.format(plmodels['rush_yds']))
        
    rbdf.to_csv(fname)
    return rbdf

def minimize_model_hyperparameters(pos, model_name='rush_att'):
    logging.info('will search for good hyperparameters for {}'.format(model_name))
    rbdf = get_pos_df(pos)
    playerids = rbdf['playerid'].unique()
    years = rbdf['year'].unique()

    model_defs = {
        'rush_att':{
            'model':RushAttModel,
            'par_bounds':[ # we could consider including parameter bounds in a class method of the models
                (0,None),(0,None),
                (0.,1.0),
                (0.1,1.0),(0.5,1.0),
            ],
            
        },
        'rush_yds':{
            'model':RushYdsModel,
            'par_bounds':[
                (0,None),(0,None),(0,None),(0,None),
                (0.0,10.0),
                (0.0,1.0),
                (0.0,5.0), # skew
                (0.2,1.0), # season memory
                (0.4,1.0),
                (0.2,1.0), # game memory - doesn't help much
                (0.4,1.0),
            ],
        },
        'rush_tds':{ # this default one has very poor residuals, or there is some other problem
            'model':RushTdModel,
            'par_bounds':[
                (1e-5,None),(1e-5,None),
                (0.0,10.0),# let's uncap learn rate
                (0.2,1.0),(0.5,1.0)
                ]
        }
    }
    learn = True
    mdef = model_defs[model_name]
    mdtype = mdef['model']
    pars0 = mdtype._default_hyperpars(pos)
    logging.info('starting with parameters {}'.format(pars0))

    def tot_kld(hparams):
        tot_week = 0
        tot_kld = 0
        for pid in playerids:
            pdf = rbdf[rbdf['playerid'] == pid]
            pname = pdf['name'].unique()[0]
            plmodel = mdtype(*hparams, learn=learn)
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
def top_rb_names(year, nadp=32, n1=4, n2=12):
    """
    nadp: top number to choose from ADP
    n1: number of times as a POS1 *on their team* to trigger inclusion
    n2: number of times as a POS2 on their team to be included
    """
    
    pos = 'RB'
    adpfname = 'adp_historical/adp_{}_{}.csv'.format(pos.lower(), year)
    if not os.path.isfile(adpfname):
        logging.error('cannot find ADP file. try running get_historical_adp.py')
    topdf = pd.read_csv(adpfname)
    # this should already be sorted
    topdf = topdf.head(nadp)
    topnames = topdf['name']

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

# returns a list of the top N receivers (tight ends or wideouts) by ADP in a given year, plus some that may become relevant
def top_wr_names(year, nadp=36, n1=4, n2=12):
    """
    nadp: top number to choose from ADP
    n1: number of times as a POS1 *on their team* to trigger inclusion
    n2: number of times as a POS2 on their team to be included
    """

    pos = 'WR'
    adpfname = 'adp_historical/adp_{}_{}.csv'.format(pos.lower(), year)
    if not os.path.isfile(adpfname):
        logging.error('cannot find ADP file. try running get_historical_adp.py')
    topdf = pd.read_csv(adpfname)
    # this should already be sorted
    topdf = topdf.head(nadp)
    topnames = topdf['name']
    
    # note that some of the fantasy data is incorrect, e.g. 2009 Fred Jackson is listed as a WR
    # toppldf = pd.DataFrame(['name', 'playerid', 'year', pos+'1', pos+'2']) # will hold how many times each player was an RB1
    # print (toppldf)

    yrdf = pd.read_csv('weekly_stats/fantasy_stats_year_{}.csv'.format(year))
    # manually remove QBs from here
    # actually we need to rely on this in order to distinguish from TEs
    yrdf = yrdf[(yrdf['pos'] == 'WR')]
    teams = yrdf['team'].unique()
    weeks = yrdf['week'].unique()
    top1,top2 = {},{}
    for week,team in itertools.product(weeks, teams):
        relpos = yrdf[(yrdf['week'] == week) & (yrdf['team'] == team)]
        relpos = relpos.assign(fp = relpos['receiving_rec'] + 0.1*relpos['receiving_yds']).sort_values('fp', ascending=False).drop('fp', axis=1)
        if len(relpos) == 0: continue # then it's probably a bye week
        toprec = relpos.iloc[0]
        if toprec['receiving_yds'] >= 10:
            t1name = toprec['name']
            top1[t1name] = top1.get(t1name, 0) + 1
            top2[t1name] = top2.get(t1name, 0) + 1
        if len(relpos) < 2: continue
        toprec = relpos.iloc[1]
        if toprec['receiving_yds'] >= 10:
            t2name = toprec['name']
            top2[t2name] = top2.get(t2name, 0) + 1

    pretop = topnames.copy()
    topnames = topnames.append(pd.Series([name for name,ntop in top1.items() if ntop >= n1]))
    topnames = topnames.append(pd.Series([name for name,ntop in top2.items() if ntop >= n2]))

    topnames = topnames.unique()
    return topnames

# returns a list of the top tight ends by ADP and performance in a given year
# is separate from WRs for convenience, since we'd like the parameters to be different (and 2nd place doesn't cut it)
# TODO: should probably rank TEs with how they stack up with other receivers on each team
# for TEs, ADP might just be enough...
def top_te_names(year, nadp=18,
                 # n1=8, n2=18
):
    """
    nadp: top number to choose from ADP
    n1: number of times as a POS1 *on their team* to trigger inclusion
    # n2: number of times as a POS2 on their team to be included
    """

    pos = 'TE'
    adpfname = 'adp_historical/adp_{}_{}.csv'.format(pos.lower(), year)
    if not os.path.isfile(adpfname):
        logging.error('cannot find ADP file. try running get_historical_adp.py')
    topdf = pd.read_csv(adpfname)
    # this should already be sorted
    topdf = topdf.head(nadp)
    topnames = topdf['name']
    
    # yrdf = pd.read_csv('weekly_stats/fantasy_stats_year_{}.csv'.format(year))
    # yrdf = yrdf[yrdf['pos'].isin(['TE', 'WR'])]
    # teams = yrdf['team'].unique()
    # weeks = yrdf['week'].unique()
    # top1,top2 = {},{}
    # for week,team in itertools.product(weeks, teams):
    #     relpos = yrdf[(yrdf['week'] == week) & (yrdf['team'] == team)]
    #     relpos = relpos.assign(fp = relpos['receiving_rec'] + 0.1*relpos['receiving_yds']).sort_values('fp', ascending=False).drop('fp', axis=1)
    #     if len(relpos) == 0: continue # then it's probably a bye week
    #     toprec = relpos.iloc[0]
    #     if toprec['receiving_yds'] >= 10:
    #         t1name = toprec['name']
    #         top1[t1name] = top1.get(t1name, 0) + 1
    #         # top2[t1name] = top2.get(t1name, 0) + 1
    #     if len(relpos) < 2: continue
    #     toprec = relpos.iloc[1]
    #     if toprec['receiving_yds'] >= 10:
    #         t2name = toprec['name']
    #         top2[t2name] = top2.get(t2name, 0) + 1

    # pretop = topnames.copy()
    # topnames = topnames.append(pd.Series([name for name,ntop in top1.items() if ntop >= n1]))
    # topnames = topnames.append(pd.Series([name for name,ntop in top2.items() if ntop >= n2]))
    # topnames = topnames.unique()
    
    return topnames

# for QBs just go by ADP right now.
# TODO: put more effort into validating that this gives us a good sample
def top_qb_names(year, nadp=18,
                 # n1=12
):
    """
    nadp: top number to choose from ADP
    n1: number of times as a POS1 *on their team* to trigger inclusion
    # n2: number of times as a POS2 on their team to be included
    """

    pos = 'QB'
    adpfname = 'adp_historical/adp_{}_{}.csv'.format(pos.lower(), year)
    if not os.path.isfile(adpfname):
        logging.error('cannot find ADP file. try running get_historical_adp.py')
    topdf = pd.read_csv(adpfname)
    # this should already be sorted
    topdf = topdf.head(nadp)
    topnames = topdf['name']
    
    # yrdf = pd.read_csv('weekly_stats/fantasy_stats_year_{}.csv'.format(year))
    # yrdf = yrdf[yrdf['pos'].isin(['TE', 'WR'])]
    # teams = yrdf['team'].unique()
    # weeks = yrdf['week'].unique()
    # top1,top2 = {},{}
    # for week,team in itertools.product(weeks, teams):
    #     relpos = yrdf[(yrdf['week'] == week) & (yrdf['team'] == team)]
    #     relpos = relpos.assign(fp = relpos['receiving_rec'] + 0.1*relpos['receiving_yds']).sort_values('fp', ascending=False).drop('fp', axis=1)
    #     if len(relpos) == 0: continue # then it's probably a bye week
    #     toprec = relpos.iloc[0]
    #     if toprec['receiving_yds'] >= 10:
    #         t1name = toprec['name']
    #         top1[t1name] = top1.get(t1name, 0) + 1
    #         # top2[t1name] = top2.get(t1name, 0) + 1
    #     if len(relpos) < 2: continue
    #     toprec = relpos.iloc[1]
    #     if toprec['receiving_yds'] >= 10:
    #         t2name = toprec['name']
    #         top2[t2name] = top2.get(t2name, 0) + 1

    # pretop = topnames.copy()
    # topnames = topnames.append(pd.Series([name for name,ntop in top1.items() if ntop >= n1]))
    # topnames = topnames.append(pd.Series([name for name,ntop in top2.items() if ntop >= n2]))
    # topnames = topnames.unique()
    
    return topnames

if __name__ == '__main__':
    main()
