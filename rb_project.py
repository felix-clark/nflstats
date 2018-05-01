#!/usr/bin/env python3
import prediction_models as pm
# from getPoints import *
from ruleset import *
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from pandas.plotting import autocorrelation_plot
from numpy import sqrt

def get_rb_list(years, datadir='./yearly_stats/'):
    rbnames = pd.Series()
    for year in years:
        csvName = '{}/fantasy_{}.csv'.format(datadir,year)
        df = pd.read_csv(csvName)
        validrbs = df.loc[df['pos'] == 'RB']
        # we might want more filters, but for now use all in the dataset
        # Not all RBs are starters, due to 3rd-down backs. we'll use games played but use a higher threshold than qbs
        validrbs = validrbs.loc[validrbs['games_played'].astype(int) >= 8]
        validrbs = validrbs.loc[validrbs['rushing_att'].astype(int) >= 1]
        rbn = validrbs['name']
        # print('{} rbs in {}'.format(len(rbn), year))
        rbnames = pd.concat([rbnames,rbn], ignore_index=True, verify_integrity=True).drop_duplicates()
    rbnames.sort_values(inplace=True)
    rbnames.reset_index(drop=True,inplace=True)
    return rbnames

# get a dataframe of the relevant RBs
def get_rb_df(years, datadir='./yearly_stats/'):
    ls_rbdfs = []
    for year in years:
        csvName = '{}/fantasy_{}.csv'.format(datadir,year)
        df = pd.read_csv(csvName)
        validrbs = df.loc[df['pos'] == 'RB']
        validrbs = validrbs.loc[validrbs['games_played'].astype(int) >= 8]
        validrbs = validrbs.loc[validrbs['rushing_att'].astype(int) >= 1]
        validrbs['year'] = year
        # validrbs['yearly_rank'] = validrbs['passing_yds'].rank()
        ls_rbdfs.append(validrbs)
    allrbs = pd.concat(ls_rbdfs, ignore_index=True, verify_integrity=True).drop_duplicates()
    return allrbs

def get_points( rs, df ):
    """
    repeating ourselves a lot here; it would be good to combine code w/ above (or only use this)
    rs: rule set
    df: dataframe containing player stats
    name: name of decorated stat
    """
    return \
    + rs.ppRY * df['rushing_yds'] \
    + rs.ppRY10 * (df['rushing_yds'] / 10) \
    + rs.ppRTD * df['rushing_td'] \
    + rs.ppREY * df['receiving_yds'] \
    + rs.ppREY10 * (df['receiving_yds'] / 10) \
    + rs.ppREC * df['receiving_rec'] \
    + rs.ppRETD * df['receiving_td']


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    years = range(1990, 2018)
    rbnames = get_rb_list(years)
    years = range(1980, 2018)
    rbdf = get_rb_df(years)
    rbdf = rbdf.drop(columns=['pos', 'Unnamed: 0'])

    # rbdf['pass_att_pg'] = rbdf['passing_att'] / rbdf['games_played']
    # rbdf['cmp_pct'] = rbdf['passing_cmp'] / rbdf['passing_att']
    # rbdf['pass_yds_pc'] = rbdf['passing_yds'] / rbdf['passing_cmp']
    # # rbdf['pass_td_pg'] = rbdf['passing_td'] / rbdf['games_played']
    # rbdf['pass_td_pc'] = rbdf['passing_td'] / rbdf['passing_cmp']
    # rbdf['int_pct'] = rbdf['passing_int'] / rbdf['passing_att']

    rbdf['rush_att_pg'] = rbdf['rushing_att'] / rbdf['games_played']
    rbdf['rush_yds_pa'] = rbdf['rushing_yds'] / rbdf['rushing_att']
    rbdf['rush_td_pa'] = rbdf['rushing_td'] / rbdf['rushing_att']
    rbdf['rec_tgt_pg'] = rbdf['receiving_tgt'] / rbdf['games_played']
    rbdf['rec_rec_pt'] = rbdf['receiving_rec'] / rbdf['receiving_tgt']
    rbdf['rec_yds_pc'] = rbdf['receiving_yds'] / rbdf['receiving_rec']
    rbdf['rec_td_pc'] = rbdf['receiving_td'] / rbdf['receiving_rec']
    # some RBs will have zero targets. set the per-stats to zero.
    rbdf.fillna(0, inplace=True)
    
    checkWindowModels = False
    if checkWindowModels:
        logging.info('checking models')
    
        n_data = 0
        stats_to_predict = ['rush_att_pg', 'rush_yds_pa', 'rush_td_pa', 'rec_tgt_pg', 'rec_rec_pt', 'rec_yds_pc', 'rec_td_pc']
        avg_st = {st:rbdf[st].mean() for st in stats_to_predict}
        std_st = {st:rbdf[st].std() for st in stats_to_predict}
        naive_sse = {key:0 for key in stats_to_predict}
        md_naive = lambda data: pm.naive(data)
        mean_sse = {key:0 for key in stats_to_predict}
        md_mean = lambda data: pm.mean(data)
        alphas = [0.2, 0.25, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9]
        # exp_sse = {a:{key:0 for key in stats_to_predict} for a in alphas}
        exp_sse = {key:{a:0 for a in alphas} for key in stats_to_predict}
        md_exp = {}
        for a in alphas:
            # we won't count rookie years in the squared error calculation.
            # thus we won't provide a default value, since they're dependent on the statistic anyway.
            md_exp[a] = lambda data: pm.exp_window(data, alpha=a)
    
        for name in rbnames:
            pdf = rbdf[rbdf['name'] == name]
            n_data += pdf.size - 1
            for st in stats_to_predict:
                naive_sse[st] += pm.sse_subseries(pdf[st], md_naive)
                mean_sse[st] += pm.sse_subseries(pdf[st], md_mean)
                for a in alphas:
                    exp_sse[st][a] += pm.sse_subseries(pdf[st], md_exp[a])

        for st in stats_to_predict:
            logging.info('\n  model performance for {}:'.format(st))
            logging.info('{:.4g} \pm {:.4g}'.format(avg_st[st], std_st[st]))
            logging.info('naive RMSE: {:.4g}'.format(sqrt(naive_sse[st]/n_data)))
            logging.info('mean RMSE: {:.4g}'.format(sqrt(mean_sse[st]/n_data)))
            minalpha = min(exp_sse[st].items(), key=lambda x: x[1])
            logging.info('exp[{}] RMSE: {:.5g}'.format(minalpha[0], sqrt(minalpha[1]/(n_data-1))))
            
    # compare the dumb prediction methodology to experts for the last year
    # the dumb model really doesn't work well for RBs, seemingly because it's more a question of who gets touches
    rules = phys_league
    # the year to compare predictions with
    current_year = 2017
    current_rbnames = get_rb_list([current_year])
    current_rbdf = rbdf[(rbdf['name'].isin(current_rbnames)) & (rbdf['year'] < current_year)]

    pred_rbs = []
    for name in current_rbnames:
        rbdat = current_rbdf[current_rbdf['name'] == name]
        if rbdat.size == 0:
            print('skipping {}'.format(name))
            continue
        pred_data = pm.dumb_rb_predictions(rbdat)
        pred_data['name'] = name
        pred_rbs.append(pred_data)
    pred_rbdf = pd.DataFrame(pred_rbs)
    pred_rbdf['dumb_proj'] = get_points(rules, pred_rbdf)
    # hardcore in Zeke's suspension:
    if current_year == 2017:
        # six-game suspension
        pred_rbdf.loc[pred_rbdf['name'] == 'Ezekiel Elliott', 'dumb_proj'] *= (15.0-6.0)/15.0

    real_rbdat = get_rb_df([current_year])
    real_rbdat['fantasy_points'] = get_points(rules, real_rbdat)
    real_rbdat.loc[real_rbdat['games_played'] == 16, 'fantasy_points'] *= 15.0/16.0

    ex_proj = pd.read_csv('./preseason_rankings/project_fp_rb_pre{}.csv'.format(current_year))
    ex_proj['expert_proj'] = get_points(rules, ex_proj)
    
    pred_rbdf = pd.merge(pred_rbdf, ex_proj[['name','expert_proj']], on='name')
    pred_rbdf = pd.merge(pred_rbdf, real_rbdat[['name','fantasy_points']], on='name')
    logging.info(pred_rbdf.columns)
    pred_rbdf = pred_rbdf[(pred_rbdf['games_played'] > 10) & (pred_rbdf['expert_proj'] > 10) & (pred_rbdf['fantasy_points'] > 20)]
    
    pred_rbdf.sort_values('fantasy_points', inplace=True, ascending=False)
    pred_rbdf.reset_index(drop=True,inplace=True)
    logging.info('\n' + str(pred_rbdf[['name', 'dumb_proj', 'expert_proj', 'fantasy_points']]))
    expert_err = pred_rbdf['expert_proj'] - pred_rbdf['fantasy_points']
    dumb_err = pred_rbdf['dumb_proj'] - pred_rbdf['fantasy_points']
    logging.info( 'expert MSE: {}'.format( sqrt(sum(expert_err**2)/expert_err.size) ) )
    logging.info( 'dumb MSE: {}'.format( sqrt(sum(dumb_err**2)/dumb_err.size) ) )
    logging.info( 'expert MAE: {}'.format( sum(expert_err.abs())/expert_err.size ) )
    logging.info( 'dumb MAE: {}'.format( sum(dumb_err.abs())/dumb_err.size ) )
