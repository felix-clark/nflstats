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

    # weib_rook_res = st.weibull_min.fit(data_papg_rook, floc=0)
    # log.info('fit rookies to weibull distribution:\n{}'.format(weib_rook_res))
    # weib_inc_res = st.weibull_min.fit(data_papg_inc, floc=0)
    # log.info('fit all to weibull distribution:\n{}'.format(weib_inc_res))
    
    n_rookie_seasons = len(data_papg_rook)
    n_rookie_games = rookiedf['games_played'].sum()
    log.info('{} rookie seasons'.format(n_rookie_seasons))
    rk_weight_mean = rookiedf['passing_att'].sum() / n_rookie_games
    rk_weight_stddev = np.sqrt(n_rookie_seasons/(n_rookie_seasons-1)*(rookiedf['games_played'] * (rookiedf['pass_att_pg'] - rk_weight_mean)**2).sum()/ n_rookie_games)
    log.info('weighted by games played, rookie distribution has mean/std: {} \pm {}'.format(rk_weight_mean, rk_weight_stddev))

    
    
    sns.set()
    # # drew bledsoe has the most pass attempts per game: 70
    # xfvals = np.linspace(-0.5, 80+0.5, 128)
    # bins_papg = None # range(0,80)
    # plt_gp = sns.distplot(data_papg_rook, bins=bins_papg,
    #                       kde=False, norm_hist=True,
    #                       hist_kws={'log':False, 'align':'left'})
    # plt.plot(xfvals, dist_fit.neg_binomial(xfvals, rrk, prk), '--', lw=2, color='violet')
    # plt.plot(xfvals, st.t.pdf(xfvals, stdf, stloc, stscale), '-', lw=1, color='blue')
    # plt.plot(xfvals, st.norm.pdf(xfvals, rk_weight_mean, rk_weight_stddev), '-', lw=1, color='green')
    # plt.plot(xfvals, st.weibull_min.pdf(xfvals, *weib_rook_res), '-', lw=1, color='red')
    # plt.title('rookies')
    # plt_gp.figure.savefig('pass_att_pg_rookie.png'.format())
    # plt_gp.figure.show()
    # plt.figure() # create a new figure
    # plt_gp = sns.distplot(data_papg_inc, bins=bins_papg,
    #                       kde=False, norm_hist=True,
    #                       hist_kws={'log':False, 'align':'left'})
    # plt.plot(xfvals, dist_fit.neg_binomial(xfvals, rinc, pinc), '--', lw=2, color='violet')
    # plt.plot(xfvals, st.weibull_min.pdf(xfvals, *weib_inc_res), '-', lw=1, color='red')
    # plt.title('all seasons')
    # plt_gp.figure.savefig('pass_att_pg.png'.format())
    # plt_gp.figure.show()

    # plt_gp = sns.pairplot(rookiedf, vars = ['games_started', 'pass_att_pg'])
    # plt_gp.savefig('pass_att_gs_qb.png')
    
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
    
    ###############################################################
    
    # define a model that is just a constant gaussian w/ the inclusive distribution
    cgaussmodel = bay.const_gauss_model(gp_avg_all, gp_var_all)
    
    # r0,p0 = rrk,prk
    # log.info('using rookie r,p = {},{}'.format(r0,p0))
    # r0,p0 = rinc,pinc
    # log.info('using inclusive r,p = {},{}'.format(r0,p0))
    log.info('using (weighted) statistical mean and stddev')
    r0,p0 = r0stat,p0stat
    log.info('starting mean = {}'.format(r0*(1-p0)/p0))
    r0 = 1.0*r0
    # # if scale down r and beta to be on the per-game level?
    # beta0 /= 1 # scaling doesn't seem to help
    # p0p = beta0/(1+beta0)
    p0 = 1.0*p0
    log.info( 'using (possibly reduced) per-game r, b = {}, {}'.format(r0, p0))
    lrp = 1.0 # why does the bayes model have worse performance

    nbmodel = bay.neg_binomial_model(r0, p0, lrp)
    
    # hyperparameters for t model
    t_mu0 = rk_weight_mean
    t_mu0 = gp_avg_all
    t_nu0 = 1 # turning this down low reduces EVSE and MSE but increases the KLD
    t_alpha0 = 2.0 # needs to be > 1 to have a well-defined variance.
    # if alpha is large, the contribution to the MSE term is small, though in practice it doesn't seem to change it much
    # t_alpha0 = n_rookie_seasons # 211 rookie seasons, tho turning this up makes the variance not change much (?)
    # t_beta0 = rk_weight_stddev**2 * (t_alpha0-1)/(1+1.0/t_nu0)
    t_beta0 = gp_var_all * (t_alpha0-1)/(1+1.0/t_nu0)

    tmodel = bay.t_model(t_mu0, t_nu0, t_alpha0, t_beta0, lrp)
    
    # for pname in ['Peyton Manning', 'Tom Brady', 'Troy Aikman', 'Drew Brees']:
    #     tbdf = posdf[posdf['name'] == pname]
    #     mean_papg_tb = tbdf['pass_att_pg'].mean()
    #     var_papg_tb = tbdf['pass_att_pg'].var()
    #     log.info('{} \pm {} (index of dispersion {}) pa/pg for {}'
    #              .format(mean_papg_tb, np.sqrt(var_papg_tb),
    #                      var_papg_tb/mean_papg_tb, pname)) # they're all under-dispersed

    modeldf = pd.DataFrame()
    
    for pname in posnames:
        # explicitly turn into numpy arrays
        pdata_gs = posdf[posdf['name'] == pname]['games_played'].values
        pdata_pa = posdf[posdf['name'] == pname]['passing_att'].values
        pdata_papg = pdata_pa / pdata_gs
        career_length = pdata_gs.size
        if career_length < 2: continue

        weights = np.full(pdata_gs.shape, 1.0)
        # weights = pdata_gs / np.max(pdata_gs)

        # we should probably weight all these by the # of games played each season.
        # the learning should be weighted too, so we need to implement this in the bayes models.
        mses_nb = nbmodel.mse((pdata_pa,pdata_gs), weights=weights)*weights
        mses_t = tmodel.mse(pdata_papg, weights=weights)*weights
        mses_cgauss = cgaussmodel.mse(pdata_papg, weights=weights)*weights
        normkld = False
        klds_cgauss = cgaussmodel.kld(pdata_papg, weights=weights, normalize=normkld)*weights
        klds_nb = nbmodel.kld((pdata_pa,pdata_gs), weights=weights, normalize=normkld)*weights
        klds_t = tmodel.kld(pdata_papg, weights=weights, normalize=normkld)
        evses_nb = nbmodel.evse((pdata_pa,pdata_gs), weights=weights)
        evses_t = tmodel.evse(pdata_papg, weights=weights)
        evses_cgauss = cgaussmodel.evse(pdata_papg, weights=weights)
        res_nb = nbmodel.residuals((pdata_pa,pdata_gs), weights=weights)
        res_t = tmodel.residuals(pdata_papg, weights=weights)
        res_cgauss = cgaussmodel.residuals(pdata_papg, weights=weights)
        evs_nb = nbmodel.evs((pdata_pa,pdata_gs), weights=weights)
        evs_t = tmodel.evs(pdata_papg, weights=weights)*weights
        evs_cgauss = cgaussmodel.evs(pdata_papg, weights=weights)
        vars_nb = nbmodel.vars((pdata_pa,pdata_gs), weights=weights)
        vars_t = tmodel.vars(pdata_papg, weights=weights)
        vars_cgauss = cgaussmodel.vars(pdata_papg, weights=weights)

        for iy in range(career_length):            
            dfeldata = {'name':pname, 'model':'data', 'career_year':iy+1,
                        'ev':pdata_papg[iy], 'kld':0, 'weight':weights[iy]}
            dfelnb = {'name':pname, 'model':'nbinom', 'residuals':res_nb[iy],
                      'ev':evs_nb[iy], 'scale':np.sqrt(vars_nb[iy]), 'mse':mses_nb[iy], 'kld':klds_nb[iy],
                      'evse':evses_nb[iy], 'career_year':iy+1, 'weight':weights[iy]}
            dfelcgauss = {'name':pname, 'model':'cgauss', 'residuals':res_cgauss[iy],
                          'ev':evs_cgauss[iy], 'scale':np.sqrt(vars_cgauss[iy]), 'mse':mses_cgauss[iy], 'kld':klds_cgauss[iy],
                          'evse':evses_cgauss[iy], 'career_year':iy+1, 'weight':weights[iy]}
            dfelt = {'name':pname, 'model':'studentt', 'residuals':res_t[iy],
                     'ev':evs_t[iy], 'scale':np.sqrt(vars_t[iy]), 'mse':mses_t[iy], 'kld':klds_t[iy],
                     'evse':evses_t[iy], 'career_year':iy+1, 'weight':weights[iy]}
            modeldf = modeldf.append([dfeldata, dfelnb, dfelcgauss, dfelt], ignore_index=True)
        
    modeldf.reset_index(drop=True, inplace=True)
        
    # with all the data stored, add some extra stats to the data "model"
    career_long = modeldf['career_year'].max()
    for iy in range(1,career_long+1):
        mask = (modeldf['model'] == 'data') & (modeldf['career_year'] == iy)
        reldf = modeldf[mask]
        meanyr = reldf['ev'].mean()
        stdyr = reldf['ev'].std()
        modeldf.loc[mask,'scale'] = stdyr
        modeldf.loc[mask,'residuals'] = (reldf['ev'] - meanyr)/stdyr

    modeldf['rmse'] = np.sqrt(modeldf['mse'])
    modeldf['revse'] = np.sqrt(modeldf['evse'])
    modeldf['norm_rmse'] = modeldf['rmse'] / modeldf['scale']
    modeldf['norm_revse'] = modeldf['revse'] / modeldf['scale']
        
    log.info('total player-seasons: {}'.format(modeldf[modeldf['model'] == 'data']['weight'].sum()))
    for stat in ['evse', 'mse', 'kld']:
        for model in ['cgauss', 'nbinom', 'studentt']:
            thismodel = modeldf[modeldf['model'] == model]
            val = (thismodel[stat]*thismodel['weight']).sum()/thismodel['weight'].sum()
            if stat in ['evse', 'mse']: val = np.sqrt(val)
            log.info('{} for {} model: {}'.format(stat, model, val))

    plot_vars = ['ev', 'residuals', 'norm_rmse', 'norm_revse']
    for var in plot_vars:
        plt.figure()
        varplt = sns.lvplot(data=modeldf, x='career_year', y=var, hue='model')

    plt.show(block=True)
