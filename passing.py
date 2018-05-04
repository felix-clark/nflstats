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
import scipy.stats as st

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
    posdf['pass_att_pg'] = posdf['passing_att'] / posdf['games_played']

    rookiedf = pd.concat([posdf[posdf['name'] == name].head(1) for name in posnames])
    rookiedf['pass_att_pg'] = rookiedf['passing_att'] / rookiedf['games_played']
    # log.debug('rookies:\n' + str(rookiedf[['name','year','team']]))
    
    data_papg_inc = posdf['pass_att_pg']
    data_papg_rook = rookiedf['pass_att_pg']
    
    
    _,(rrk,prk),cov,llpdf = dist_fit.to_neg_binomial( data_papg_rook )
    log.info('rookie: r = {}, p = {}, LL per dof = {}'.format(rrk, prk, llpdf))
    log.info('covariance:\n' + str(cov))
    _,(rinc,pinc),cov,llpdf = dist_fit.to_neg_binomial( data_papg_inc )
    log.info('all: r = {}, p = {}, LL per dof = {}'.format(rinc, pinc, llpdf))
    log.info('covariance:\n' + str(cov))

    stdf,stloc,stscale = st.t.fit(data_papg_rook)
    log.info('fit to student\'s t distribution:\n{}'.format((stdf,stloc,stscale)))

    weib_rook_res = st.weibull_min.fit(data_papg_rook, floc=0)
    log.info('fit rookies to weibull distribution:\n{}'.format(weib_rook_res))
    weib_inc_res = st.weibull_min.fit(data_papg_inc, floc=0)
    log.info('fit all to weibull distribution:\n{}'.format(weib_inc_res))
    
    n_rookie_seasons = len(data_papg_rook)
    n_rookie_games = rookiedf['games_played'].sum()
    log.info('{} rookie seasons'.format(n_rookie_seasons))
    rk_weight_mean = rookiedf['passing_att'].sum() / n_rookie_games
    rk_weight_stddev = np.sqrt(n_rookie_seasons/(n_rookie_seasons-1)*(rookiedf['games_played'] * (rookiedf['pass_att_pg'] - rk_weight_mean)**2).sum()/ n_rookie_games)
    log.info('weighted by games played, rookie distribution has mean/std: {} \pm {}'.format(rk_weight_mean, rk_weight_stddev))
    
    sns.set()
    # drew bledsoe has the most pass attempts per game: 70
    xfvals = np.linspace(-0.5, 80+0.5, 128)
    bins_papg = None # range(0,80)
    plt_gp = sns.distplot(data_papg_rook, bins=bins_papg,
                          kde=False, norm_hist=True,
                          hist_kws={'log':False, 'align':'left'})
    plt.plot(xfvals, dist_fit.neg_binomial(xfvals, rrk, prk), '--', lw=2, color='violet')
    plt.plot(xfvals, st.t.pdf(xfvals, stdf, stloc, stscale), '-', lw=1, color='blue')
    plt.plot(xfvals, st.norm.pdf(xfvals, rk_weight_mean, rk_weight_stddev), '-', lw=1, color='green')
    plt.plot(xfvals, st.weibull_min.pdf(xfvals, *weib_rook_res), '-', lw=1, color='red')
    plt.title('rookies')
    plt_gp.figure.savefig('pass_att_pg_rookie.png'.format())
    plt_gp.figure.show()
    plt.figure() # create a new figure
    plt_gp = sns.distplot(data_papg_inc, bins=bins_papg,
                          kde=False, norm_hist=True,
                          hist_kws={'log':False, 'align':'left'})
    plt.plot(xfvals, dist_fit.neg_binomial(xfvals, rinc, pinc), '--', lw=2, color='violet')
    plt.plot(xfvals, st.weibull_min.pdf(xfvals, *weib_inc_res), '-', lw=1, color='red')
    plt.title('all seasons')
    plt_gp.figure.savefig('pass_att_pg.png'.format())
    plt_gp.figure.show()

    plt_gp = sns.pairplot(rookiedf, vars = ['games_started', 'pass_att_pg'])
    plt_gp.savefig('pass_att_gs_qb.png')
    # plt_gp.show() # no show for pairplot?
    
    # plt.show(block=True)

                 
    gp_avg_all = data_papg_inc.mean()
    gp_var_all = data_papg_inc.var()
    log.info('used for const model: average attempts per game (inclusive) = {} \pm {}'.format(gp_avg_all, np.sqrt(gp_var_all)))
    p0stat = gp_avg_all/gp_var_all
    r0stat = gp_avg_all**2/(gp_var_all-gp_avg_all)
    # the distribution for passing attempts is underdispersed compared to neg. bin.
    # it actually works well for pass completions, though
    log.info('using the inclusive stats to set r,p would yield {:.4g},{:.4g}'.format(r0stat,p0stat))
    p0stat = rk_weight_mean/rk_weight_stddev**2
    r0stat = rk_weight_mean**2/(rk_weight_stddev**2 - rk_weight_mean)
    log.info('using weighted rookie to set r,p would yield {:.4g},{:.4g}'.format(r0stat,p0stat))
    
    # r0,p0 = rrk,prk
    # log.info('using rookie r,p = {},{}'.format(r0,p0))
    # r0,p0 = rinc,pinc
    # log.info('using inclusive r,p = {},{}'.format(r0,p0))
    log.info('using (weighted) statistical mean and stddev')
    r0,p0 = r0stat,p0stat
    log.info('starting mean = {}'.format(r0*(1-p0)/p0))
    
    total_n = 0
    mse_nb_sum = 0.
    mse_degen_sum = 0.
    mse_const_sum = 0.
    mse_mean_sum = 0.
    kld_nb_sum = 0.
    kld_const_sum = 0.
    # mae_nb_sum = 0.
    # mae_degen_sum = 0.
    # mae_const_sum = 0. # this one is slow

    # save errors as function of year in league
    mses_nb_year = []
    mses_cgauss_year = []
    klds_nb_year = []
    klds_cgauss_year = []
    
    r0p = 1.0*r0
    # beta0 = p0/(1-p0) # mean = r/beta
    # # if scale down r and beta to be on the per-game level?
    # beta0 /= 1 # scaling doesn't seem to help
    # p0p = beta0/(1+beta0)
    p0p = p0
    log.info( 'using (possibly reduced) per-game r, b = {}, {}'.format(r0p, p0p))
    lrp = 1.0 # why does the bayes model have worse performance

    for pname in ['Peyton Manning', 'Tom Brady', 'Troy Aikman', 'Drew Brees']:
        tbdf = posdf[posdf['name'] == pname]
        mean_papg_tb = tbdf['pass_att_pg'].mean()
        var_papg_tb = tbdf['pass_att_pg'].var()
        log.info('{} \pm {} (index of dispersion {}) pa/pg for {}'
                 .format(mean_papg_tb, np.sqrt(var_papg_tb),
                         var_papg_tb/mean_papg_tb, pname))
    
    for pname in posnames:
        # explicitly turn into numpy arrays
        pdata_gs = posdf[posdf['name'] == pname]['games_started'].values
        pdata_pa = posdf[posdf['name'] == pname]['passing_att'].values
        pdata_papg = pdata_pa / pdata_gs
        career_length = pdata_gs.size
        
        for yeardata in [mses_nb_year, mses_cgauss_year, klds_nb_year, klds_cgauss_year]:
            # if we have not yet encountered a player of this long of a career,
            while len(yeardata) < career_length: yeardata.append([])
        
        
        mses_nb = bay.nbinom.mse((pdata_pa,pdata_gs), r0p, p0p, lr=lrp)
        mses_degen = bay.degen_const.mse(pdata_papg, gp_avg_all)
        # for iy,mse in enumerate(mses_degen): mses_degen_year[iy].push_back(mse)
        mses_const = bay.gauss_const.mse(pdata_papg, gp_avg_all, gp_var_all)
        mses_mean = bay.mse_model_mean(pdata_papg, gp_avg_all) # need to figure out how to specify this model precisely 
        # maes_nb = bay.nbinom.mae((pdata_pa,pdata_gs), r0p, p0p, lr=lrp) # not implemented, but also seems to have an error since we haven't implemented the two-par version
        # maes_degen = bay.degen_const.mae(pdata_papg, gp_avg_all)
        # maes_const = bay.gauss_const.mae(pdata_papg, gp_avg_all, gp_var_all) # currently very slow
        klds_const = bay.gauss_const.kld(pdata_papg, gp_avg_all, gp_var_all)
        klds_nb = bay.nbinom.kld((pdata_pa,pdata_gs), r0p, p0p, lr=lrp)

        # it will also be useful to show error terms as function of career depth
        for iy,mse in enumerate(mses_nb): mses_nb_year[iy].append(mse)
        for iy,mse in enumerate(mses_const): mses_cgauss_year[iy].append(mse)
        for iy,kld in enumerate(klds_nb): mses_nb_year[iy].append(kld)
        for iy,kld in enumerate(klds_const): mses_cgauss_year[iy].append(kld)
        
        total_n += career_length
        # do we need to save the sums separately when we are saving by career length?
        mse_nb_sum += mses_nb.sum()
        mse_const_sum += mses_const.sum()
        mse_degen_sum += mses_degen.sum()
        mse_mean_sum += mses_mean.sum()
        # mae_nb_sum += maes_nb.sum() # do we actually have this one?
        # mae_const_sum += maes_const.sum()
        # mae_degen_sum += maes_degen.sum()
        kld_nb_sum += klds_nb.sum()
        kld_const_sum += klds_const.sum()
        # degen KLD is infinite, and we need to specify variance in the "mean" model, or define it as a moving degen

    # right now bayes does worse than just using the average
    log.info('RMSE for const gauss model: {}'.format(np.sqrt(mse_const_sum/total_n)))
    log.info('RMSE for const degen model: {}'.format(np.sqrt(mse_degen_sum/total_n)))
    log.info('RMSE for mean model: {}'.format(np.sqrt(mse_mean_sum/total_n)))
    log.info('RMSE for NB bayes model: {}'.format(np.sqrt(mse_nb_sum/total_n)))
    # log.info('MAE for const model: {}'.format(np.sqrt(mae_const_sum/total_n)))
    # log.info('MAE for const degen model: {}'.format(np.sqrt(mae_degen_sum/total_n)))
    # log.info('MAE for NB bayes model: {}'.format(np.sqrt(mae_nb_sum/total_n)))
    log.info('Kullback-Leibler divergence for const gauss: {}'.format(kld_const_sum/total_n))
    log.info('Kullback-Leibler divergence for NB bayes: {}'.format(kld_nb_sum/total_n))
    log.info('total player-seasons: {}'.format(total_n))
    
    # plt.show(block=True)
