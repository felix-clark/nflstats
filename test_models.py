#!/usr/bin/env python3
import dist_fit
from get_player_stats import *
from playermodels.positions import *
from tools import corr_spearman

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.optimize as opt
import itertools
import logging
import warnings
import os.path
import argparse
import random

def main():
    logging.getLogger().setLevel(logging.DEBUG)
    warnings.simplefilter('error') # quit on warnings
    np.set_printoptions(precision=4)
    sns.set()

    rush_models = ['rush_att', 'rush_yds', 'rush_td']
    rec_models = ['targets', 'rec', 'rec_yds', 'rec_td'] # we don't have the data for targets easily accessible yet
    pass_models = ['pass_att', 'pass_cmp', 'pass_yds', 'pass_td', 'pass_int']
    all_models = rush_models + rec_models + pass_models
    
    parser = argparse.ArgumentParser(description='optimize and analyze bayesian models')
    parser.add_argument('position',type=str,choices=['RB', 'QB', 'WR', 'TE'],help='which position to analyze')
    parser.add_argument('--opt-hyper',nargs='?',type=str,choices=all_models,help='try to improve hyperparameters for this model')

    args = parser.parse_args()

    position = args.position
    logging.info('working with {}s'.format(position))
    
    if args.opt_hyper:
        # some of the learn rates may be too large because we only have weekly data back to 2009 right now.
        # this means we're missing most of the beginnings of careers, and are starting in the middle.
        hps = find_model_hyperparameters(position, args.opt_hyper)
    
    posdf = get_model_df(position)
    posdf = posdf[posdf['game_num'] < 16]# .dropna() # don't necessarily remove nans

    # models = ['rush_att', 'rush_yds', 'rush_tds'] # edit this to suppress info we've already looked at
    # models = ['rush_yds'] # edit this to suppress info we've already looked at
    models = pass_models if position == 'QB' else rush_models if position == 'RB' else rec_models
    # models = rec_models # overwrite to look at RB rec
    for model in models:
        klds = posdf[model+'_kld']
        chisqs = posdf[model+'_chisq']
        logging.info('{} kld = {:.6f}, chisq = {:.4f} out of {} weeks (avg {:.6f}, {:.3f})'
                     .format(model, klds.sum(), chisqs.sum(),
                             klds.size, klds.mean(), chisqs.mean()))
        # print(posdf['{}_chisq'.format(model)].mean()) # yes, this gives the same result
        # plt.figure()
        # game_plt = sns.boxenplot(data=posdf, x='game_num', y=model+'_kld', hue='career_year')
        # plt.show(block=True)
        # plot_vars = ['kld', 'cdf']
        # for pname in plot_vars:
        #     plt.figure()
        #     year_plt = sns.boxenplot(data=posdf, x='career_year', y=model+'_'+pname) # hue = model # when we compare models (baseline would be nice)
        #     # year_plt = sns.lvplot(data=posdf, x='career_year', y=model+'_'+pname) # hue = model # when we compare models (baseline would be nice)
        # plt.show(block=True)

    # print(posdf[posdf['rush_att'] == 0])
    # TODO: we should be able to split the dataset in half randomly and see flat CDFs in both samples
    # pltvar = sns.distplot(posdf['rush_yds_cdf'])
    # pltvar.figure.show()
    # plt.show(block=True)
    # cdf_plt = sns.pairplot(posdf, #height = 4,
    #                        vars=['rush_yds_cdf', 'rush_att', 'rush_yds'],
    #                        hue='career_year',
    # )
    # plt.show(block=True)

    # exit(1)
        
    # for model in models:
    #     cdf = posdf['{}_cdf'.format(model)]
    #     cdf = cdf[cdf.notna()]
    #     plt_cdf = sns.distplot(cdf,
    #                            hist_kws={'log':False, 'align':'left'})
    #     # plt_cdf.figure.savefig('rush_att_res')
    #     plt_cdf.figure.show()
    # plt.show(block=True)

    if False:
        resnames = ['{}_cdf'.format(m) for m in models] # we could take the ppf of this to look at standardized residuals
        plt_corr = sns.pairplot(posdf, # height = 4,
                                dropna=True,
                                vars=resnames,
                                # kind='reg', # do linear regression to look for correlations
                                # diag_kind='hist', # causes error
                                hue='career_year'
        )
        # plt_corr = sns.pairplot(posdf, # height = 4,
        #                         dropna=True,
        #                         vars=resnames,
        #                         # kind='reg', # do linear regression to look for correlations
        #                         # diag_kind='hist', # causes error
        #                         hue='game_num'
        # )
        # plt_corr.map_diag(plt.hist)
        # plt_corr.map_offdiag(sns.kdeplot, n_levels=6)
        plt.show(block=True)
        
    good_pos = True
    if position == 'RB': good_pos = posdf['rush_att'] > 0
    if position == 'QB': good_pos = posdf['pass_cmp'] > 0
    if position in ['WR', 'TE']:
         # not clear we should filter these. include zero for now
        good_pos = posdf['rec'] >= 0
    
    corrdf = posdf[good_pos]
    
    # logging.warning('exiting early')
    # exit(0)
    
    ## # correlation calculations:
    plmodel = gen_player_model(position)
    stats = plmodel.stats
    print(stats)
    corr_mat = np.ones(shape=(len(stats), len(stats)))
    # for sa,sb in itertools.combinations(stats, 2):
    for ia in range(len(stats)):
        sa = stats[ia]
        for ib in range(ia):
            sb = stats[ib]
            cdfs = (corrdf[sa+'_cdf'].values, corrdf[sb+'_cdf'].values)
            # sr = st.spearmanr(*cdfs)
            # rho,p = sr
            # print('rho: {:.3f} \t p : {:.3g}'.format(rho, p))
            # quantities that are "per" another quantity should be weighted by that quantity so the correlation is dominated by relevant data
            weights = np.ones(shape=cdfs[0].shape)
            cmpsts = set((sa, sb))
            # we don't want to double-weight; for instance the correlation between
            # rush_yds and rush_td should be weighted by a single power of rush_att
            if len(cmpsts & set(['rush_yds', 'rush_td'])):
                # print('weighting {},{} by rush attempts'.format(sa, sb))
                weights *= corrdf['rush_att'].values
                
            if 'rec' in cmpsts:
                # print('weighting {},{} by targets'.format(sa, sb))
                weights *= corrdf['targets'].values
            elif len(cmpsts & set(['rec_yds', 'rec_td'])):
                # print('weighting {},{} by receptions'.format(sa, sb))
                weights *= corrdf['rec'].values

            if len(cmpsts & set(['pass_cmp', 'pass_int'])) > 0:
                # print('weighting {},{} by pass attempts'.format(sa, sb))
                weights *= corrdf['pass_att'].values
            elif len(cmpsts & set(['pass_yds', 'pass_td'])) > 0:
                # print('weighting {},{} by completions'.format(sa, sb))
                weights *= corrdf['pass_cmp'].values
                
            corr_sp = corr_spearman(*cdfs, weights=weights)
            corr_mat[ia,ib] = corr_mat[ib,ia] = corr_sp
            # print('weighted spearman correlation of {} and {}:'.format(sb, sa))
            # print('{:.3f}'.format(corr_sp))

    # array2string lets us print out w/ commas for easy copy-paste
    print('weighted covariance matrix for {} stats:'.format(position))
    print(np.array2string(corr_mat, precision=3, separator=',', suppress_small=True))
        
    logging.warning('exiting early')
    exit(0)    
    
    plt_corr = sns.pairplot(corrdf, #height = 4,
                            vars=[v+'_cdf' for v in stats],
                            dropna=True,
                            # kind='reg', # do linear regression to look for correlations
                            # hue='career_year'
    )
    plt.show(block=True)

    return


def get_pos_dfs(pos, fname = None):
    # this whole procedure can be done rather rapidly now, so there's no need to cache
    pos = pos.upper()
    pldfs = []

    good_col = None
    if pos == 'QB': good_col = lambda col: 'rec' not in col and 'kick' not in col and 'punt' not in col
    if pos == 'K': good_col = lambda col: 'pass' not in col and 'rush' not in col and 'rec' not in col and 'punt' not in col
    if pos in ['RB', 'WR', 'TE']: good_col = lambda col: 'pass' not in col and 'kick' not in col and 'punt' not in col
    players = get_pos_players(pos)
    pfrids = players['pfr_id']
    for pid in pfrids:
        pdf = get_player_stats(pid)
        if len(pdf) == 0:
            logging.error('empty data for {}'.format(pid))
            continue
        # pdf = pdf[pdf['gs'] == '*'] # only count where they started? # don't have the indicator for this, and it might rule out some real data points for e.g. RBs, TEs
        pdf.loc[:,'pfr_id'] = pid

        columns = [col for col in pdf.columns if good_col(col)]
        columns.remove('game_location')
        columns.remove('opp')
        columns.remove('game_result')
        pdf = pdf[columns].fillna(0)
        
        # they should already be sorted properly, but let's check.
        pdf = pdf.sort_values(['year', 'game_num']).reset_index(drop=True)
        pldfs.append(pdf)

    return pldfs


def get_model_df( pos='RB', fname = None):
    if fname is None:
        fname = 'model_cache_{}.csv'.format(pos.lower())
    if os.path.isfile(fname):
        return pd.read_csv(fname)

    posdfs = get_pos_dfs(pos)

    # basing rush attempts soley on the past is not so great.
    # ideally we use a team-based touch model.
    # we need to look into the discrepancies more to figure out the problems
    # i suspect injuries, trades, then matchups are the big ones.

    models = []
    if pos == 'QB':
        models.extend(['pass_att', 'pass_cmp', 'pass_yds', 'pass_td', 'pass_int'])
    if pos in ['RB', 'QB', 'WR']:
        models.extend(['rush_att', 'rush_yds', 'rush_td'])
    if pos in ['WR', 'TE', 'RB']:
        models.extend(['targets', 'rec', 'rec_yds', 'rec_td'])
    tot_week = 0
    
    for pdf in posdfs:
        # pname = pdf['name'].unique()[0]
        # pname = pdf['player'].unique()[0]
        
        pid = pdf['pfr_id'].unique()[0]
        plmodels = [get_stat_model(mod).for_position(pos) for mod in models]

        # insert zeros for stats that aren't saved for this player
        for var in np.concatenate([plmodel.var_names for plmodel in plmodels]):
            if var not in pdf:
                pdf[var] = 0

        years = pdf['year'].unique()
        # we could skip single-year seasons
        for icareer,year in enumerate(years):
            # can we use "group by" or something to reduce the depth of these loops?
            ypdf = pdf[(pdf['year'] == year) & (pdf['game_num'] < 16)] # week 17 is often funky
            for index, row in ypdf.iterrows():
                # these do point to the same one:
                # assert((row[['name', 'year', 'week']] == rbdf.loc[index][['name', 'year', 'week']]).all())
                tot_week += 1
                pdf.loc[index,'career_year'] = icareer+1
                for model in plmodels:
                    try:
                        mvars = [row[v] for v in model.var_names]
                    except:
                        print(row)
                        print (model.var_names)
                    kld = model.kld(*mvars)
                    chisq = model.chi_sq(*mvars)
                    data = row[model.pred_var]
                    depvars = [row[v] for v in model.dep_vars]
                    ev = model.ev(*depvars)
                    var = model.var(*depvars)
                    cdf = model.cdf(*mvars) # standardized to look like a gaussian
                    # if row[model.pred_var] == 0:
                    #     cdf = random.uniform(0,cdf) # smear the cdf to deal with small discrete numbers
                    # res = (data-ev)/np.sqrt(var)
                    pdf.loc[index,'{}_ev'.format(model.name)] = ev
                    pdf.loc[index,'{}_cdf'.format(model.name)] = cdf
                    pdf.loc[index,'{}_kld'.format(model.name)] = kld
                    pdf.loc[index,'{}_chisq'.format(model.name)] = chisq
                    model.update_game(*mvars) # it's important that this is done last, after computing KLD and chi^2
            for model in plmodels:
                model.new_season()
                
    combdf = pd.concat(posdfs, sort=False)
    combdf.to_csv(fname)
    return combdf


def find_model_hyperparameters(pos, model_name='rush_att'):
    logging.info('will search for good hyperparameters for {}'.format(model_name))
    posdfs = get_pos_dfs(pos)

    mdtype = get_stat_model(model_name)
    hpars0 = mdtype._default_hyperpars(pos)
    hparbounds = mdtype._hyperpar_bounds()
    logging.info('starting with parameters {}'.format(hpars0))
    
    assert(len(hpars0) == len(hparbounds))

    newmin = np.inf
    
    def tot_kld(hparams):
        nonlocal newmin
        tot_kld = 0
        for pdf in posdfs:
            plmodel = mdtype(*hparams)
            for var in plmodel.var_names:
                if var not in pdf:
                    pdf[var] = 0
            years = pdf['year'].unique()
            assert((np.diff(years) > 0).all()) # make sure the years are sorted
            # we could skip single-year seasons
            for icareer,year in enumerate(years):
                # can we use "group by" or something to reduce the depth of these loops?
                # ypdf = pdf[(pdf['year'] == year) & (pdf['week'] < 17)] # week 17 is often funky
                ypdf = pdf[(pdf['year'] == year) & (pdf['game_num'] < 16)] # week 17 is often funky
                for index, row in ypdf.iterrows():
                    # print(plmodel.var_names)
                    # print(row)
                    # print(row['targets'])
                    try:
                        mvars = [row[v] for v in plmodel.var_names]
                    except:
                        logging.error('missing some of {}'.format(plmodel.var_names))
                        print(row)
                        exit(1)
                    # if plmodel.name == 'rec':
                    #     if row['rec'] > row['targets']:
                    #         print(row)
                    kld = plmodel.kld(*mvars)
                    # if np.isnan(kld):
                    #     print(plmodel.var_names)
                    #     print(mvars)
                    #     exit(1)
                    tot_kld += kld
                    # it's important that updating is done after computing KLD
                    plmodel.update_game(*mvars)
                plmodel.new_season()
        if tot_kld < newmin:
            newmin = tot_kld
            print('kld = {} at {}'.format(tot_kld, hparams))
        if not np.isfinite(tot_kld):
            print('kld = {} at {}'.format(tot_kld, hparams))
        return tot_kld

    minned = opt.minimize(tot_kld, x0=hpars0,
                          # method='Nelder-Mead', # N-M can't deal w/ bounds
                          bounds=hparbounds,
                          # it'll take several iterations to start going in the right direction w/ the default algorithm
                          # tho there are several function calls per "iteration" w/ default
                          options={'maxiter':32,
                                   # 'ftol':1e-12 # seems like a waste
                          })
    print(minned)
    minpars = minned.x
    print(mdtype(*minpars))
    return minpars

### we are no longer using these functions to grad by ADP
### could we just delete these functions?

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
    yrdf = yrdf[(yrdf['rush_att'] > 4) & (yrdf['pos'] != 'QB')]
    teams = yrdf['team'].unique()
    weeks = yrdf['week'].unique()
    top1,top2 = {},{}
    for week,team in itertools.product(weeks, teams):
        relpos = yrdf[(yrdf['week'] == week) & (yrdf['team'] == team)].sort_values('rush_att', ascending=False)
        if len(relpos) == 0: continue # then it's probably a bye week
        toprsh = relpos.iloc[0]
        if toprsh['rush_yds'] >= 10:
            t1name = toprsh['name']
            top1[t1name] = top1.get(t1name, 0) + 1
            top2[t1name] = top2.get(t1name, 0) + 1
        if len(relpos) < 2: continue
        toprsh = relpos.iloc[1]
        if toprsh['rush_yds'] >= 10:
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
        relpos = relpos.assign(fp = relpos['rec'] + 0.1*relpos['rec_yds']).sort_values('fp', ascending=False).drop('fp', axis=1)
        if len(relpos) == 0: continue # then it's probably a bye week
        toprec = relpos.iloc[0]
        if toprec['rec_yds'] >= 10:
            t1name = toprec['name']
            top1[t1name] = top1.get(t1name, 0) + 1
            top2[t1name] = top2.get(t1name, 0) + 1
        if len(relpos) < 2: continue
        toprec = relpos.iloc[1]
        if toprec['rec_yds'] >= 10:
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
    #     relpos = relpos.assign(fp = relpos['rec'] + 0.1*relpos['rec_yds']).sort_values('fp', ascending=False).drop('fp', axis=1)
    #     if len(relpos) == 0: continue # then it's probably a bye week
    #     toprec = relpos.iloc[0]
    #     if toprec['rec_yds'] >= 10:
    #         t1name = toprec['name']
    #         top1[t1name] = top1.get(t1name, 0) + 1
    #         # top2[t1name] = top2.get(t1name, 0) + 1
    #     if len(relpos) < 2: continue
    #     toprec = relpos.iloc[1]
    #     if toprec['rec_yds'] >= 10:
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
    #     relpos = relpos.assign(fp = relpos['rec'] + 0.1*relpos['rec_yds']).sort_values('fp', ascending=False).drop('fp', axis=1)
    #     if len(relpos) == 0: continue # then it's probably a bye week
    #     toprec = relpos.iloc[0]
    #     if toprec['rec_yds'] >= 10:
    #         t1name = toprec['name']
    #         top1[t1name] = top1.get(t1name, 0) + 1
    #         # top2[t1name] = top2.get(t1name, 0) + 1
    #     if len(relpos) < 2: continue
    #     toprec = relpos.iloc[1]
    #     if toprec['rec_yds'] >= 10:
    #         t2name = toprec['name']
    #         top2[t2name] = top2.get(t2name, 0) + 1

    # pretop = topnames.copy()
    # topnames = topnames.append(pd.Series([name for name,ntop in top1.items() if ntop >= n1]))
    # topnames = topnames.append(pd.Series([name for name,ntop in top2.items() if ntop >= n2]))
    # topnames = topnames.unique()
    
    return topnames

if __name__ == '__main__':
    main()
