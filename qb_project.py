#!/usr/bin/env python3
import prediction_models as pm
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
    qbdf['pass_td_pg'] = qbdf['passing_td'] / qbdf['games_played']
    qbdf['int_pct'] = qbdf['passing_int'] / qbdf['passing_att']
    
    # logging.info(qbdf.columns)
    
    sns.set()
    # outputing an empty plot. it may expect values for every QB at every time step
    # pass_att_pg = sns.tsplot(data=qbdf, time='year', value='pass_att_pg', unit='name')
    # pass_att_pg.figure.show()
    # pass_att_pg.figure.savefig( 'pass_att_pg.png' )

    # plt.figure()

    # ls = [qbdf[qbdf['name'] == name] for name in ['Peyton Manning', 'Tom Brady']]
    # ac = autocorrelation_plot()

    n_data = 0
    stats_to_predict = ['pass_att_pg', 'cmp_pct', 'pass_td_pg', 'int_pct']
    avg_st = {st:qbdf[st].mean() for st in stats_to_predict}
    std_st = {st:qbdf[st].std() for st in stats_to_predict}
    naive_sse = {key:0 for key in stats_to_predict}
    md_naive = lambda data: pm.naive(data)
    mean_sse = {key:0 for key in stats_to_predict}
    md_mean = lambda data: pm.mean(data)
    alphas = [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9]
    exp_sse = {a:{key:0 for key in stats_to_predict} for a in alphas}
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
                exp_sse[a][st] += pm.sse_subseries(pdf[st], md_exp[a])

    for st in stats_to_predict:
        logging.info('\n  model performance for {}:'.format(st))
        logging.info('{:.4g} \pm {:.4g}'.format(avg_st[st], std_st[st]))
        logging.info('naive RMSE: {:.4g}'.format(sqrt(naive_sse[st]/n_data)))
        logging.info('mean RMSE: {:.4g}'.format(sqrt(mean_sse[st]/n_data)))
        for a in alphas:
            logging.info('exp[{}] RMSE: {:.5g}'.format(a, sqrt(exp_sse[a][st]/(n_data-1))))

    
