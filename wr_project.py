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

def get_wr_list(years, datadir='./yearly_stats/'):
    wrnames = pd.Series()
    for year in years:
        csvName = '{}/fantasy_{}.csv'.format(datadir,year)
        df = pd.read_csv(csvName)
        validwrs = df.loc[df['pos'] == 'WR']
        # we might want more filters, but for now use all in the dataset
        # Not all WRs are starters, due to 3rd-down backs. we'll use games played but use a higher threshold than qbs
        validwrs = validwrs.loc[validwrs['games_played'].astype(int) >= 8]
        validwrs = validwrs.loc[validwrs['receiving_rec'].astype(int) >= 1]
        wrn = validwrs['name']
        # print('{} wrs in {}'.format(len(wrn), year))
        wrnames = pd.concat([wrnames,wrn], ignore_index=True, verify_integrity=True).drop_duplicates()
    wrnames.sort_values(inplace=True)
    wrnames.reset_index(drop=True,inplace=True)
    return wrnames

# get a dataframe of the relevant WRs
def get_wr_df(years, datadir='./yearly_stats/'):
    ls_wrdfs = []
    for year in years:
        csvName = '{}/fantasy_{}.csv'.format(datadir,year)
        df = pd.read_csv(csvName)
        validwrs = df.loc[df['pos'] == 'WR']
        validwrs = validwrs.loc[validwrs['games_played'].astype(int) >= 8]
        validwrs = validwrs.loc[validwrs['receiving_rec'].astype(int) >= 1]
        validwrs['year'] = year
        # validwrs['yearly_rank'] = validwrs['passing_yds'].rank()
        ls_wrdfs.append(validwrs)
    allwrs = pd.concat(ls_wrdfs, ignore_index=True, verify_integrity=True).drop_duplicates()
    return allwrs

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
    wrnames = get_wr_list(years)
    # years = range(1980, 2018)
    years = range(1992, 2018)
    wrdf = get_wr_df(years)
    wrdf = wrdf.drop(columns=['pos', 'Unnamed: 0'])

    # wrdf['pass_att_pg'] = wrdf['passing_att'] / wrdf['games_played']
    # wrdf['cmp_pct'] = wrdf['passing_cmp'] / wrdf['passing_att']
    # wrdf['pass_yds_pc'] = wrdf['passing_yds'] / wrdf['passing_cmp']
    # # wrdf['pass_td_pg'] = wrdf['passing_td'] / wrdf['games_played']
    # wrdf['pass_td_pc'] = wrdf['passing_td'] / wrdf['passing_cmp']
    # wrdf['int_pct'] = wrdf['passing_int'] / wrdf['passing_att']

    # wrdf['rush_att_pg'] = wrdf['rushing_att'] / wrdf['games_played']
    # wrdf['rush_yds_pa'] = wrdf['rushing_yds'] / wrdf['rushing_att']
    # wrdf['rush_td_pa'] = wrdf['rushing_td'] / wrdf['rushing_att']
    # old data doesn't even have receiving targets, so we maybe shouldn't rely on it
    logging.warning('old data does not have targets')
    wrdf['rec_tgt_pg'] = wrdf['receiving_tgt'] / wrdf['games_played']
    wrdf['rec_rec_pt'] = wrdf['receiving_rec'] / wrdf['receiving_tgt']
    wrdf['rec_yds_pc'] = wrdf['receiving_yds'] / wrdf['receiving_rec']
    wrdf['rec_td_pc'] = wrdf['receiving_td'] / wrdf['receiving_rec']
    # most WRs will have zero rushes. set the per-stats to zero.
    wrdf.fillna(0, inplace=True)
    
    checkWindowModels = False
    if checkWindowModels:
        logging.info('checking models')
    
        n_data = 0
        stats_to_predict = ['rec_tgt_pg', 'rec_rec_pt', 'rec_yds_pc', 'rec_td_pc']
        avg_st = {st:wrdf[st].mean() for st in stats_to_predict}
        std_st = {st:wrdf[st].std() for st in stats_to_predict}
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
    
        for name in wrnames:
            pdf = wrdf[wrdf['name'] == name]
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
    current_wrnames = get_wr_list([current_year])
    current_wrdf = wrdf[(wrdf['name'].isin(current_wrnames)) & (wrdf['year'] < current_year)]

    pred_wrs = []
    for name in current_wrnames:
        wrdat = current_wrdf[current_wrdf['name'] == name]
        # skip rookies
        if wrdat.size == 0:
            print('skipping {}'.format(name))
            continue
        pred_data = pm.dumb_wr_predictions(wrdat)
        pred_data['name'] = name
        pred_wrs.append(pred_data)
    pred_wrdf = pd.DataFrame(pred_wrs)
    pred_wrdf['dumb_proj'] = get_points(rules, pred_wrdf)

    real_wrdat = get_wr_df([current_year])
    real_wrdat['fantasy_points'] = get_points(rules, real_wrdat)
    real_wrdat.loc[real_wrdat['games_played'] == 16, 'fantasy_points'] *= 15.0/16.0

    ex_proj = pd.read_csv('./preseason_rankings/project_fp_wr_pre{}.csv'.format(current_year))
    ex_proj['expert_proj'] = get_points(rules, ex_proj)
    
    pred_wrdf = pd.merge(pred_wrdf, ex_proj[['name','expert_proj']], on='name')
    pred_wrdf = pd.merge(pred_wrdf, real_wrdat[['name','fantasy_points']], on='name')
    logging.info(pred_wrdf.columns)
    pred_wrdf = pred_wrdf[(pred_wrdf['games_played'] > 10) & (pred_wrdf['expert_proj'] > 64) & (pred_wrdf['fantasy_points'] > 32)]
    
    pred_wrdf.sort_values('fantasy_points', inplace=True, ascending=False)
    pred_wrdf.reset_index(drop=True,inplace=True)
    logging.info(pred_wrdf[['name', 'dumb_proj', 'expert_proj', 'fantasy_points']])
    expert_err = pred_wrdf['expert_proj'] - pred_wrdf['fantasy_points']
    dumb_err = pred_wrdf['dumb_proj'] - pred_wrdf['fantasy_points']
    logging.info( 'expert MSE: {}'.format( sqrt(sum(expert_err**2)/expert_err.size) ) )
    logging.info( 'dumb MSE: {}'.format( sqrt(sum(dumb_err**2)/dumb_err.size) ) )
    logging.info( 'expert MAE: {}'.format( sum(expert_err.abs())/expert_err.size ) )
    logging.info( 'dumb MAE: {}'.format( sum(dumb_err.abs())/dumb_err.size ) )
