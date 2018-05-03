#!/usr/bin/env python3
# import prediction_models as pm
# from getPoints import *
from ruleset import *
import dist_fit
import bayes_models as bay
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from pandas.plotting import autocorrelation_plot
import numpy as np
from sys import argv
from scipy.stats import *
# from scipy.stats import beta

# get a dataframe of the relevant positional players
def get_pos_df(pos, years, datadir='./yearly_stats/', keepnames=None):
    ls_dfs = []
    for year in years:
        csvName = '{}/fantasy_{}.csv'.format(datadir,year)
        df = pd.read_csv(csvName)
        df['year'] = year
        valids = df.loc[df['pos'] == pos.upper()]
        if keepnames is not None:
            valids = valids[valids['name'].isin(keepnames)]
        if pos.lower() == 'qb':
            valids = valids.loc[valids['passing_att'].astype(int) >= 4]
        if pos.lower() == 'rb':
            valids = valids.loc[valids['rushing_att'].astype(int) >= 4]
        if pos.lower() in ['wr', 'te']:
            valids = valids.loc[valids['receiving_rec'].astype(int) >= 4]
        if valids.size == 0:
            logging.warning('no {} in {}'.format(pos, year))
        ls_dfs.append(valids)
    allpos = pd.concat(ls_dfs, ignore_index=True, verify_integrity=True)
    return allpos

def get_pos_list(pos, years, datadir='./yearly_stats/'):
    posdf = get_pos_df(pos, years, datadir)
    posnames = posdf['name'].drop_duplicates().sort_values()
    posnames.reset_index(drop=True,inplace=True)
    return posnames


if __name__ == '__main__':
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    if len(argv) < 2:
        log.error('usage: {} <position>'.format(argv[0]))
        exit(1)
        
    pos = argv[1].lower()

    # starting at 1999 will let us have all data for everyone selected
    years = range(1999, 2018)
    posnames = get_pos_list(pos, years)
    years = range(1983, 2018)
    posdf = get_pos_df(pos, years, keepnames=posnames)
    posdf = posdf.drop(columns=['pos', 'Unnamed: 0'])

    maxgames = 16
    # we only care about games played for this script
    # Jerry Rice switch teams in 2004 and played 17 games, so we'll just set his to 16    
    gt16 = posdf['games_played'] > maxgames
    posdf.loc[gt16,'games_played'] = maxgames

    # rookiedat = []
    # for name in posnames:
    #     rookiedat.append(posdf[posdf['name'] == name].head(1))
    rookiedf = pd.concat([posdf[posdf['name'] == name].head(1) for name in posnames])
    log.debug('rookies:\n' + str(rookiedf[['name','year','team']]))

    # while other positions switch in-and-out, a QB is really relevant only if he is starting.
    # looking at starts rather than plays eliminates some noise w/ backups
    gp_stat = 'games_started' if pos == 'qb' else 'games_played'
    data_gp_rook = rookiedf[gp_stat]
    data_gp = posdf[gp_stat]
    
    
    _,(ark,brk),cov,llpdf = dist_fit.to_beta_binomial( (0,maxgames), data_gp_rook )
    log.info('rookie: alpha = {}, beta = {}, LL per dof = {}'.format(ark, brk, llpdf))
    log.info('covariance:\n' + str(cov))
    _,(ainc,binc),cov,llpdf = dist_fit.to_beta_binomial( (0,maxgames), data_gp )
    log.info('all: alpha = {}, beta = {}, LL per dof = {}'.format(ainc, binc, llpdf))
    log.info('covariance:\n' + str(cov))
    
    sns.set()
    xfvals = np.linspace(-0.5, maxgames+0.5, 128)
    plt_gp = sns.distplot(data_gp, bins=range(0,maxgames+2),
                          kde=False, norm_hist=True,
                          hist_kws={'log':False, 'align':'left'})
    plt.plot(xfvals, dist_fit.beta_binomial(xfvals, maxgames, ark, brk), '--', lw=2, color='violet')
    plt.title('rookies')
    plt_gp.figure.savefig('{}_{}.png'.format(gp_stat, pos))
    plt_gp.figure.show()
    # The bayesian update rule is:
    # alpha -> alpha + (games_played)
    # beta -> beta + (n - games_played)
    # we can just start on the default rookie values

    # for QBs we might want to adjust for year-in-league, or just filter on those which started many games

    # log.info('using rookie a,b = {},{}'.format(ark,brk))
    # alpha0,beta0 = ark,brk
    # log.info('using inclusive a,b = {}'.format(ainc, binc)) # does a bit worse
    # alpha0,beta0 = ainc,binc
    # m1 = data_gp_rook.mean()
    # m2 = (data_gp_rook**2).mean()
    log.info('using moment method for combined a,b = {},{}'.format(ark,brk))
    m1 = data_gp.mean()
    m2 = (data_gp**2).mean()
    denom = maxgames*(m2/m1 - m1 - 1) + m1
    alpha0 = (maxgames*m1 - m2)/denom
    beta0 = (maxgames-m1)*(maxgames - m2/m1)/denom
    log.info('starting mean = {}'.format(maxgames*alpha0/(alpha0+beta0)))
                 
    gp_avg_all = data_gp.mean()
    gp_var_all = data_gp.var()
    log.info('used for const model: average games_played = {} \pm {}'.format(gp_avg_all, np.sqrt(gp_var_all)))
    
    # entries = []
    mse_total_n = 0
    mse_bb_sum = 0.
    mse_const_sum = 0.
    mse_mean_sum = 0.
    kld_bb_sum = 0.
    kld_const_sum = 0.
    mae_bb_sum = 0.
    mae_const_sum = 0.
    
    for pname in posnames:
    # in this loop we should instead evaluate our bayesian model
        pdata = posdf[posdf['name'] == pname][gp_stat]
        alpha0p = 1.0*alpha0
        beta0p = 1.0*beta0
        # for QBs there may be no hope, but for WRs a bayes model w/ a slower learn rate seems to do well
        lrp = 0.25
        gp_mses_bb = bay.bbinom.mse(pdata, maxgames, alpha0p, beta0p, lr=lrp)
        # gp_mses_const = bay.mse_model_const(pdata, gp_avg_all, gp_var_all)
        gp_mses_const = bay.gauss_const.mse(pdata, gp_avg_all, gp_var_all)
        gp_mses_mean = bay.mse_model_mean(pdata, gp_avg_all) # could also use rookie average
        gp_maes_bb = bay.mae_model_bb(pdata, alpha0p, beta0p, lr=lrp)
        gp_maes_const = bay.mae_model_const(pdata, gp_avg_all)
        gp_kld_const = bay.gauss_const.kld(pdata, gp_avg_all, gp_var_all)
        gp_kld_bb = bay.bbinom.kld(pdata, maxgames, alpha0p, beta0p, lr=lrp)

        # log.info('{} {} {}'.format(pdata.size, gp_mses_bb.size, gp_mses_const.size))
        mse_total_n += pdata.size
        mse_bb_sum += gp_mses_bb.sum()
        mse_const_sum += gp_mses_const.sum()
        mse_mean_sum += gp_mses_mean.sum()
        mae_bb_sum += gp_maes_bb.sum()
        mae_const_sum += gp_maes_const.sum()
        kld_bb_sum += gp_kld_bb.sum()
        kld_const_sum += gp_kld_const.sum()

    # right now bayes does worse than just using the average
    log.info('RMSE for const model: {}'.format(np.sqrt(mse_const_sum/mse_total_n)))
    log.info('RMSE for mean model: {}'.format(np.sqrt(mse_mean_sum/mse_total_n)))
    log.info('RMSE for bayes model: {}'.format(np.sqrt(mse_bb_sum/mse_total_n)))
    log.info('MAE for const model: {}'.format(np.sqrt(mae_const_sum/mse_total_n)))
    log.info('MAE for bayes model: {}'.format(np.sqrt(mae_bb_sum/mse_total_n)))
    log.info('Kullback-Leibler divergence for const: {}'.format(kld_const_sum/mse_total_n))
    log.info('Kullback-Leibler divergence for bayes: {}'.format(kld_bb_sum/mse_total_n))
    log.info('total player-seasons: {}'.format(mse_total_n))
    
    # plt.show(block=True)
