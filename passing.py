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
import numpy as np
# from sys import argv
from scipy.stats import *

# get a dataframe of the relevant positional players
def get_qb_df(years, datadir='./yearly_stats/', keepnames=None):
    ls_dfs = []
    for year in years:
        csvName = '{}/fantasy_{}.csv'.format(datadir,year)
        df = pd.read_csv(csvName)
        df['year'] = year
        valids = df.loc[df['pos'] == 'QB']
        if keepnames is not None:
            valids = valids[valids['name'].isin(keepnames)]
        valids = valids.loc[valids['games_started'].astype(int) > 2]
        # somehow there are QBs who started but didn't throw any passes...
        valids = valids.loc[valids['passing_att'].astype(int) >= 4]
        if valids.size == 0:
            logging.warning('no qbs in {}'.format(year))
        ls_dfs.append(valids)
    allqbs = pd.concat(ls_dfs, ignore_index=True, verify_integrity=True)
    allqbs = allqbs.drop(columns=['pos', 'Unnamed: 0'])
    return allqbs

def get_qb_list(years, datadir='./yearly_stats/'):
    posdf = get_qb_df(years, datadir)
    posnames = posdf['name'].drop_duplicates().sort_values()
    posnames.reset_index(drop=True,inplace=True)
    return posnames


if __name__ == '__main__':
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)
    logging.getLogger('bayes_models').setLevel(logging.INFO)

    # if len(argv) < 2:
    #     log.error('usage: {} <position>'.format(argv[0]))
    #     exit(1)        
    # pos = argv[1].lower()

    # starting at 1999 will let us have all data for everyone selected
    years = range(1999, 2018)
    posnames = get_qb_list(years)
    years = range(1983, 2018)
    posdf = get_qb_df(years, keepnames=posnames)

    rookiedf = pd.concat([posdf[posdf['name'] == name].head(1) for name in posnames])
    log.debug('rookies:\n' + str(rookiedf[['name','year','team']]))

    data_papg_rook = rookiedf['passing_att'] / rookiedf['games_started']
    data_papg_inc = posdf['passing_att'] / posdf['games_started']
    
    
    _,(rrk,prk),cov,llpdf = dist_fit.to_neg_binomial( data_papg_rook )
    log.info('rookie: r = {}, p = {}, LL per dof = {}'.format(rrk, prk, llpdf))
    log.info('covariance:\n' + str(cov))
    _,(rinc,pinc),cov,llpdf = dist_fit.to_neg_binomial( data_papg_inc )
    log.info('all: r = {}, p = {}, LL per dof = {}'.format(rinc, pinc, llpdf))
    log.info('covariance:\n' + str(cov))
    
    sns.set()
    # drew bledsoe has the most pass attempts per game: 70
    xfvals = np.linspace(-0.5, 80+0.5, 128)
    bins_papg = None # range(0,80)
    plt_gp = sns.distplot(data_papg_rook, bins=bins_papg,
                          kde=False, norm_hist=True,
                          hist_kws={'log':False, 'align':'left'})
    plt.plot(xfvals, dist_fit.neg_binomial(xfvals, rrk, prk), '--', lw=2, color='red')
    plt.title('rookies')
    plt_gp.figure.savefig('pass_att_pg_rookie.png'.format())
    plt_gp.figure.show()
    plt_gp = sns.distplot(data_papg_inc, bins=bins_papg,
                          kde=False, norm_hist=True,
                          hist_kws={'log':False, 'align':'left'})
    plt.plot(xfvals, dist_fit.neg_binomial(xfvals, rinc, pinc), '--', lw=2, color='violet')
    plt.title('all seasons')
    plt_gp.figure.savefig('pass_att_pg.png'.format())
    plt_gp.figure.show()

    # plt.show(block=True)

    log.info('using rookie r,p = {},{}'.format(rrk,prk))
    r0,p0 = rrk,prk
    log.info('starting mean = {}'.format(r0*p0/(1-p0)))
                 
    gp_avg_all = data_papg_inc.mean()
    gp_var_all = data_papg_inc.var()
    log.info('used for const model: average attempts per game (inclusive) = {} \pm {}'.format(gp_avg_all, np.sqrt(gp_var_all)))
    
    total_n = 0
    mse_nb_sum = 0.
    mse_delta_sum = 0.
    mse_const_sum = 0.
    mse_mean_sum = 0.
    kld_nb_sum = 0.
    kld_const_sum = 0.
    # mae_nb_sum = 0.
    mae_delta_sum = 0.
    # mae_const_sum = 0. # this one is slow
    
    for pname in posnames:
        pdata_gs = posdf[posdf['name'] == pname]['games_started'].values
        pdata_pa = posdf[posdf['name'] == pname]['passing_att'].values

        log.debug('1: {}, {}'.format(pdata_gs,pdata_pa))

        pdata_papg = pdata_pa / pdata_gs
        r0p = 1.0*r0
        p0p = 1.0*p0
        lrp = 1.0
        gp_mses_nb = bay.nbinom.mse((pdata_pa,pdata_gs), r0p, p0p, lr=lrp)

        log.debug('2: {}, {}'.format(pdata_gs,pdata_pa))

        gp_mses_delta = bay.delta_const.mse(pdata_papg, gp_avg_all)
        gp_mses_const = bay.gauss_const.mse(pdata_papg, gp_avg_all, gp_var_all)
        gp_mses_mean = bay.mse_model_mean(pdata_papg, gp_avg_all) # need to figure out how to specify this model precisely 
        # gp_maes_nb = bay.nbinom.mae((pdata_pa,pdata_gs), r0p, p0p, lr=lrp) # not implemented, but also seems to have an error since we haven't implemented the two-par version
        gp_maes_delta = bay.delta_const.mae(pdata_papg, gp_avg_all)
        # gp_maes_const = bay.gauss_const.mae(pdata_papg, gp_avg_all, gp_var_all) # currently very slow
        gp_kld_const = bay.gauss_const.kld(pdata_papg, gp_avg_all, gp_var_all)
        gp_kld_nb = bay.nbinom.kld((pdata_pa,pdata_gs), r0p, p0p, lr=lrp)
        log.debug('3: {}, {}'.format(pdata_gs,pdata_pa))

        # it will also be useful to show error terms as function of career depth
        total_n += pdata_gs.size
        mse_nb_sum += gp_mses_nb.sum()
        mse_const_sum += gp_mses_const.sum()
        mse_delta_sum += gp_mses_delta.sum()
        mse_mean_sum += gp_mses_mean.sum()
        # mae_nb_sum += gp_maes_nb.sum() # do we actually have this one?
        # mae_const_sum += gp_maes_const.sum()
        mae_delta_sum += gp_maes_delta.sum()
        kld_nb_sum += gp_kld_nb.sum()
        kld_const_sum += gp_kld_const.sum()
        # delta KLD is infinite, and we need to specify variance in the "mean" model, or define it as a moving delta

    # right now bayes does worse than just using the average
    log.info('RMSE for const gauss model: {}'.format(np.sqrt(mse_const_sum/total_n)))
    log.info('RMSE for const delta model: {}'.format(np.sqrt(mse_delta_sum/total_n)))
    log.info('RMSE for mean model: {}'.format(np.sqrt(mse_mean_sum/total_n)))
    log.info('RMSE for NB bayes model: {}'.format(np.sqrt(mse_nb_sum/total_n)))
    # log.info('MAE for const model: {}'.format(np.sqrt(mae_const_sum/total_n)))
    log.info('MAE for const delta model: {}'.format(np.sqrt(mae_delta_sum/total_n)))
    # log.info('MAE for NB bayes model: {}'.format(np.sqrt(mae_nb_sum/total_n)))
    log.info('Kullback-Leibler divergence for const gauss: {}'.format(kld_const_sum/total_n))
    log.info('Kullback-Leibler divergence for NB bayes: {}'.format(kld_nb_sum/total_n))
    log.info('total player-seasons: {}'.format(total_n))
    
    # plt.show(block=True)
