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

def get_te_list(years, datadir='./yearly_stats/'):
    tenames = pd.Series()
    for year in years:
        csvName = '{}/fantasy_{}.csv'.format(datadir,year)
        df = pd.read_csv(csvName)
        validtes = df.loc[df['pos'] == 'TE']
        # we might want more filters, but for now use all in the dataset
        # Not all TEs are starters, due to 3rd-down backs. we'll use games played but use a higher threshold than qbs
        validtes = validtes.loc[validtes['games_played'].astype(int) >= 8]
        validtes = validtes.loc[validtes['receiving_rec'].astype(int) >= 1]
        ten = validtes['name']
        # print('{} tes in {}'.format(len(ten), year))
        tenames = pd.concat([tenames,ten], ignore_index=True, verify_integrity=True).drop_duplicates()
    tenames.sort_values(inplace=True)
    tenames.reset_index(drop=True,inplace=True)
    return tenames

# get a dataframe of the relevant TEs
def get_te_df(years, datadir='./yearly_stats/'):
    ls_tedfs = []
    for year in years:
        csvName = '{}/fantasy_{}.csv'.format(datadir,year)
        df = pd.read_csv(csvName)
        validtes = df.loc[df['pos'] == 'TE']
        validtes = validtes.loc[validtes['games_played'].astype(int) >= 8]
        validtes = validtes.loc[validtes['receiving_rec'].astype(int) >= 1]
        validtes['year'] = year
        # validtes['yearly_rank'] = validtes['passing_yds'].rank()
        ls_tedfs.append(validtes)
    alltes = pd.concat(ls_tedfs, ignore_index=True, verify_integrity=True).drop_duplicates()
    return alltes

def get_points( rs, df ):
    """
    repeating ourselves a lot here; it would be good to combine code w/ above (or only use this)
    rs: rule set
    df: dataframe containing player stats
    name: name of decorated stat
    """
    return rs.ppREY * df['receiving_yds'] \
    + rs.ppREY10 * (df['receiving_yds'] / 10) \
    + rs.ppREC * df['receiving_rec'] \
    + rs.ppRETD * df['receiving_td']

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    years = range(2000, 2018)
    # years = range(1990, 2018)
    tenames = get_te_list(years)
    # years = range(1980, 2018)
    years = range(1992, 2018)
    tedf = get_te_df(years)
    tedf = tedf.drop(columns=['pos', 'Unnamed: 0'])

    # tedf['pass_att_pg'] = tedf['passing_att'] / tedf['games_played']
    # tedf['cmp_pct'] = tedf['passing_cmp'] / tedf['passing_att']
    # tedf['pass_yds_pc'] = tedf['passing_yds'] / tedf['passing_cmp']
    # # tedf['pass_td_pg'] = tedf['passing_td'] / tedf['games_played']
    # tedf['pass_td_pc'] = tedf['passing_td'] / tedf['passing_cmp']
    # tedf['int_pct'] = tedf['passing_int'] / tedf['passing_att']

    # tedf['rush_att_pg'] = tedf['rushing_att'] / tedf['games_played']
    # tedf['rush_yds_pa'] = tedf['rushing_yds'] / tedf['rushing_att']
    # tedf['rush_td_pa'] = tedf['rushing_td'] / tedf['rushing_att']
    # old data doesn't even have receiving targets, so we maybe shouldn't rely on it
    logging.warning('old data does not have targets')
    tedf['rec_tgt_pg'] = tedf['receiving_tgt'] / tedf['games_played']
    tedf['rec_rec_pt'] = tedf['receiving_rec'] / tedf['receiving_tgt']
    tedf['rec_yds_pc'] = tedf['receiving_yds'] / tedf['receiving_rec']
    tedf['rec_td_pc'] = tedf['receiving_td'] / tedf['receiving_rec']
    # most TEs will have zero rushes. set the per-stats to zero.
    tedf.fillna(0, inplace=True)
    
    checkWindowModels = False
    if checkWindowModels:
        logging.info('checking models')
    
        n_data = 0
        stats_to_predict = ['rec_tgt_pg', 'rec_rec_pt', 'rec_yds_pc', 'rec_td_pc']
        avg_st = {st:tedf[st].mean() for st in stats_to_predict}
        std_st = {st:tedf[st].std() for st in stats_to_predict}
        naive_sse = {key:0 for key in stats_to_predict}
        md_naive = lambda data: pm.naive(data)
        mean_sse = {key:0 for key in stats_to_predict}
        md_mean = lambda data: pm.mean(data)
        alphas = [0.2, 0.25, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9]
        exp_sse = {key:{a:0 for a in alphas} for key in stats_to_predict}
        md_exp = {}
        for a in alphas:
            # we won't count rookie years in the squared error calculation.
            # thus we won't provide a default value, since they're dependent on the statistic anyway.
            md_exp[a] = lambda data: pm.exp_window(data, alpha=a)
    
        for name in tenames:
            pdf = tedf[tedf['name'] == name]
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
    rules = phys_league
    # the year to compare predictions with
    current_year = 2017
    current_tenames = get_te_list([current_year])
    current_tedf = tedf[(tedf['name'].isin(current_tenames)) & (tedf['year'] < current_year)]

    pred_tes = []
    for name in current_tenames:
        tedat = current_tedf[current_tedf['name'] == name]
        # skip rookies
        if tedat.size == 0:
            print('skipping {}'.format(name))
            continue
        pred_data = pm.dumb_te_predictions(tedat)
        pred_data['name'] = name
        pred_tes.append(pred_data)
    pred_tedf = pd.DataFrame(pred_tes)
    pred_tedf['dumb_proj'] = get_points(rules, pred_tedf)

    real_tedat = get_te_df([current_year])
    real_tedat['fantasy_points'] = get_points(rules, real_tedat)
    real_tedat.loc[real_tedat['games_played'] == 16, 'fantasy_points'] *= 15.0/16.0

    ex_proj = pd.read_csv('./preseason_rankings/project_fp_te_pre{}.csv'.format(current_year))
    ex_proj['expert_proj'] = get_points(rules, ex_proj)
    
    pred_tedf = pd.merge(pred_tedf, ex_proj[['name','expert_proj']], on='name')
    pred_tedf = pd.merge(pred_tedf, real_tedat[['name','fantasy_points']], on='name')
    logging.info(pred_tedf.columns)
    pred_tedf = pred_tedf[(pred_tedf['games_played'] > 10) & (pred_tedf['expert_proj'] > 32) & (pred_tedf['fantasy_points'] > 16)]
    
    pred_tedf.sort_values('fantasy_points', inplace=True, ascending=False)
    pred_tedf.reset_index(drop=True,inplace=True)
    logging.info('\n' + str(pred_tedf[['name', 'dumb_proj', 'expert_proj', 'fantasy_points']]))
    expert_err = pred_tedf['expert_proj'] - pred_tedf['fantasy_points']
    dumb_err = pred_tedf['dumb_proj'] - pred_tedf['fantasy_points']
    logging.info( 'expert MSE: {}'.format( sqrt(sum(expert_err**2)/expert_err.size) ) )
    logging.info( 'dumb MSE: {}'.format( sqrt(sum(dumb_err**2)/dumb_err.size) ) )
    logging.info( 'expert MAE: {}'.format( sum(expert_err.abs())/expert_err.size ) )
    logging.info( 'dumb MAE: {}'.format( sum(dumb_err.abs())/dumb_err.size ) )
