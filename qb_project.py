#!/usr/bin/env python3
import prediction_models as pm
# from getPoints import *
from ruleset import *
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from numpy import sqrt

def get_qb_list(years, datadir='./yearly_stats/'):
    qbnames = pd.Series()
    for year in years:
        csvName = '{}/fantasy_{}.csv'.format(datadir,year)
        df = pd.read_csv(csvName)
        validqbs = df.loc[df['pos'] == 'QB']
        # we might want more filters, but for now use all in the dataset
        # to be a fantasy factor, a qb should start in *at least* some number of games in at least *one* season
        validqbs = validqbs.loc[validqbs['games_started'].astype(int) >= 4]
        qbn = validqbs['name']
        # print('{} qbs in {}'.format(len(qbn), year))
        qbnames = pd.concat([qbnames,qbn], ignore_index=True, verify_integrity=True).drop_duplicates()
    qbnames.sort_values(inplace=True)
    qbnames.reset_index(drop=True,inplace=True)
    return qbnames

# get a dataframe of the relevant QBs
def get_qb_df(years, datadir='./yearly_stats/'):
    ls_qbdfs = []
    for year in years:
        csvName = '{}/fantasy_{}.csv'.format(datadir,year)
        df = pd.read_csv(csvName)
        validqbs = df.loc[df['pos'] == 'QB']
        # we might want more filters, but for now use all in the dataset
        # to be a fantasy factor, a qb should start in *at least* some number of games in at least *one* season
        validqbs = validqbs.loc[validqbs['games_started'].astype(int) >= 4]
        validqbs['year'] = year
        # validqbs['yearly_rank'] = validqbs['passing_yds'].rank()
        ls_qbdfs.append(validqbs)
    allqbs = pd.concat(ls_qbdfs, ignore_index=True, verify_integrity=True).drop_duplicates()
    return allqbs

def get_points( rs, df ):
    """
    repeating ourselves a lot here; it would be good to combine code w/ above (or only use this)
    rs: rule set
    df: dataframe containing player stats
    name: name of decorated stat
    """
    return \
    rs.ppPY * df['passing_yds'] \
    + rs.ppPY25 * (df['passing_yds'] / 25) \
    + rs.ppPC * df['passing_cmp'] \
    + rs.ppINC * (df['passing_att'] - df['passing_cmp']) \
    + rs.ppPTD * df['passing_td'] \
    + rs.ppINT * df['passing_int'] \
    # + rs.pp2PC * df['passing_twoptm'] \
    + rs.ppRY * df['rushing_yds'] \
    + rs.ppRY10 * (df['rushing_yds'] / 10) \
    + rs.ppRTD * df['rushing_td'] \
    + rs.pp2PR * df['rushing_twoptm']# \
    # + rs.ppREY * df['receiving_yds'] \
    # + rs.ppREY10 * (df['receiving_yds'] / 10) \
    # + rs.ppREC * df['receiving_rec'] \
    # + rs.ppRETD * df['receiving_td'] \
    # + rs.pp2PRE * df['receiving_twoptm'] \
    # + rs.ppFUML * df['fumbles_lost'] \
    # + rs.ppPAT * df['kicking_xpmade'] \
    # + rs.ppFGM * (df['kicking_fga'] - df['kicking_fgm']) \
    # + rs.ppFG0 * df['kicking_fgm']
# TODO: missing: missed PATs (ppPATM)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    # we can get about 300 unique QBs going back to 1980
    # but we don't want to go all the way to the edge; we
    # want to how well active QBs from a given range did before that range.
    years = range(1990, 2018)
    qbnames = get_qb_list(years)
    years = range(1980, 2018)
    qbdf = get_qb_df(years)
    qbdf = qbdf.drop(columns=['pos', 'Unnamed: 0'])

    qbdf['pass_att_pg'] = qbdf['passing_att'] / qbdf['games_played']
    qbdf['cmp_pct'] = qbdf['passing_cmp'] / qbdf['passing_att']
    qbdf['pass_yds_pc'] = qbdf['passing_yds'] / qbdf['passing_cmp']
    # qbdf['pass_td_pg'] = qbdf['passing_td'] / qbdf['games_played']
    qbdf['pass_td_pc'] = qbdf['passing_td'] / qbdf['passing_cmp']
    qbdf['int_pct'] = qbdf['passing_int'] / qbdf['passing_att']

    qbdf['rush_att_pg'] = qbdf['rushing_att'] / qbdf['games_played']
    qbdf['rush_yds_pa'] = qbdf['rushing_yds'] / qbdf['rushing_att']
    qbdf['rush_td_pa'] = qbdf['rushing_td'] / qbdf['rushing_att']
    
    # logging.info(qbdf.columns)

    sns.set()
    
    qbrushatt = sns.distplot(qbdf['rushing_att'] / qbdf['games_played'])
    qbrushatt.figure.savefig('qb_rush_att.png')
    qbrushyds = sns.distplot(qbdf['rushing_yds'])
    qbrushyds.figure.savefig('qb_rush_yds.png')
    qbrushydspa = sns.distplot(qbdf['rushing_yds'] / qbdf['rushing_att'])
    qbrushydspa.figure.savefig('qb_rush_yds_pa.png')

    # outputing an empty plot. it may expect values for every QB at every time step
    # pass_att_pg = sns.tsplot(data=qbdf, time='year', value='pass_att_pg', unit='name')
    # pass_att_pg.figure.show()
    # pass_att_pg.figure.savefig( 'pass_att_pg.png' )

    # plt.figure()

    # ls = [qbdf[qbdf['name'] == name] for name in ['Peyton Manning', 'Tom Brady']]
    # ac = autocorrelation_plot()

    n_data = 0
    stats_to_predict = ['pass_att_pg', 'cmp_pct', 'pass_yds_pc', 'pass_td_pc', 'int_pct'
                        ,'rush_att_pg', 'rush_yds_pa', 'rush_td_pa']
    avg_st = {st:qbdf[st].mean() for st in stats_to_predict}
    std_st = {st:qbdf[st].std() for st in stats_to_predict}
    naive_sse = {key:0 for key in stats_to_predict}
    md_naive = lambda data: pm.naive(data)
    mean_sse = {key:0 for key in stats_to_predict}
    md_mean = lambda data: pm.mean(data)
    alphas = [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9]
    # exp_sse = {a:{key:0 for key in stats_to_predict} for a in alphas}
    exp_sse = {key:{a:0 for a in alphas} for key in stats_to_predict}
    md_exp = {}
    for a in alphas:
        # we won't count rookie years in the squared error calculation.
        # thus we won't provide a default value, since they're dependent on the statistic anyway.
        md_exp[a] = lambda data: pm.exp_window(data, alpha=a)
    
    # for name in ['Peyton Manning', 'Tom Brady']:
    for name in qbnames:
        pdf = qbdf[qbdf['name'] == name]
        # logging.info(pdf)
        # ac = autocorrelation_plot(pdf['pass_att_pg'])
        # ac.figure.show()
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
        # for a in alphas:
            # logging.info('exp[{}] RMSE: {:.5g}'.format(a, sqrt(exp_sse[st][a]/(n_data-1))))

    # compare the dumb prediction methodology to experts for the last year
    # the year to compare predictions with
    rules = bro_league
    current_year = 2017
    current_qbnames = get_qb_list([current_year])
    current_qbdf = qbdf[(qbdf['name'].isin(current_qbnames)) & (qbdf['year'] < current_year)]

    pred_qbs = []
    for name in current_qbnames:
        pred_data = pm.dumb_qb_predictions(current_qbdf[current_qbdf['name'] == name])
        pred_data['name'] = name
        pred_qbs.append(pred_data)
    pred_qbdf = pd.DataFrame(pred_qbs)
    pred_qbdf['dumb_proj'] = get_points(rules, pred_qbdf)

    real_qbdat = get_qb_df([current_year])
    real_qbdat['fantasy_points'] = get_points(rules, real_qbdat)
    real_qbdat.loc[real_qbdat['games_played'] == 16, 'fantasy_points'] *= 15.0/16.0

    ex_proj = pd.read_csv('./preseason_rankings/project_fp_qb_pre{}.csv'.format(current_year))
    ex_proj['expert_proj'] = get_points(rules, ex_proj)
    
    pred_qbdf = pd.merge(pred_qbdf, ex_proj[['name','expert_proj']], on='name')
    pred_qbdf = pd.merge(pred_qbdf, real_qbdat[['name','fantasy_points']], on='name')
    logging.info(pred_qbdf.columns)
    pred_qbdf = pred_qbdf[(pred_qbdf['games_played'] > 10) & (pred_qbdf['expert_proj'] > 100) & (pred_qbdf['fantasy_points'] > 120)]
    
    pred_qbdf.sort_values('fantasy_points', inplace=True, ascending=False)
    pred_qbdf.reset_index(drop=True,inplace=True)
    logging.info(pred_qbdf[['name', 'dumb_proj', 'expert_proj', 'fantasy_points']])
    expert_sqe = (pred_qbdf['expert_proj'] - pred_qbdf['fantasy_points'])**2
    dumb_sqe = (pred_qbdf['dumb_proj'] - pred_qbdf['fantasy_points'])**2
    logging.info( 'expert MSE: {}'.format( sqrt(sum(expert_sqe)/(expert_sqe.size-1)) ) )
    logging.info( 'dumb MSE: {}'.format( sqrt(sum(dumb_sqe)/(dumb_sqe.size-1)) ) )
