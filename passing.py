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
    logging.info('dummy')     # somehow this line is needed for the custom log to work...

    # starting at 1999 will let us have all data for everyone selected
    years = range(1999, 2018)
    posnames = get_qb_list(years)
    years = range(1983, 2018)
    posdf = get_qb_df(years, keepnames=posnames)
    posdf['pass_att_pg'] = posdf['passing_att'] / posdf['games_played']
    posdf['pass_cmp_pa'] = posdf['passing_cmp'] / posdf['passing_att']
    # since completion percentage is not independent of the yardage, it may be better to consider yds/att.
    posdf['pass_yds_pc'] = posdf['passing_yds'] / posdf['passing_cmp']
    # TDs per attempt instead of per game scales out short games properly
    posdf['pass_td_pa'] = posdf['passing_td'] / posdf['passing_att']

    rookiedf = pd.concat([posdf[posdf['name'] == name].head(1) for name in posnames])

    # should be a little slicker to use numpy arrays
    data_papg_inc = posdf['pass_att_pg'].values
    data_pcpa_inc = posdf['pass_cmp_pa'].values
    data_pypc_inc = posdf['pass_yds_pc'].values
    data_ptdpa_inc = posdf['pass_td_pa'].values
    data_papg_rook = rookiedf['pass_att_pg'].values
    data_pcpa_rook = rookiedf['pass_cmp_pa'].values
    data_pypc_rook = rookiedf['pass_yds_pc'].values
    data_ptdpa_rook = rookiedf['pass_td_pa'].values

    # _,(rrk,prk),cov,llpdf = dist_fit.to_neg_binomial( data_papg_rook )
    # log.info('rookie: r = {}, p = {}, LL per dof = {}'.format(rrk, prk, llpdf))
    # log.info('covariance:\n' + str(cov))
    # _,(rinc,pinc),cov,llpdf = dist_fit.to_neg_binomial( data_papg_inc )
    # log.info('all: r = {}, p = {}, LL per dof = {}'.format(rinc, pinc, llpdf))
    # log.info('covariance:\n' + str(cov))

    # stdf,stloc,stscale = st.t.fit(data_papg_rook)
    # log.info('fit to student\'s t distribution:\n{}'.format((stdf,stloc,stscale)))

    # weib_rook_res = st.weibull_min.fit(data_papg_rook, floc=0)
    # log.info('fit rookies to weibull distribution:\n{}'.format(weib_rook_res))
    # weib_inc_res = st.weibull_min.fit(data_papg_inc, floc=0)
    # log.info('fit all to weibull distribution:\n{}'.format(weib_inc_res))
    
    n_rookie_seasons = len(data_papg_rook)
    n_rookie_games = rookiedf['games_played'].sum()
    log.info('{} rookie seasons'.format(n_rookie_seasons))
    rk_weight_mean = rookiedf['passing_att'].sum() / n_rookie_games
    rk_weight_stddev = np.sqrt(n_rookie_seasons/(n_rookie_seasons-1)*(rookiedf['games_played'] * (rookiedf['pass_att_pg'] - rk_weight_mean)**2).sum()/ n_rookie_games)
    log.info('weighted by games played, rookie pass attempt distribution has mean/std: {:.5g} \pm {:.5g}'.format(rk_weight_mean, rk_weight_stddev))
    
    
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

    # plt_gp = sns.pairplot(rookiedf, vars = ['games_played', 'pass_att_pg'])
    # plt_gp.savefig('pass_att_gs_qb.png')
    
    # plt.show(block=True)

    ## completion PCT
    pct_mean = np.mean(data_pcpa_rook)
    pct_var = np.var(data_pcpa_rook)
    alpha_pcpa_rook = pct_mean*( pct_mean*(1-pct_mean)/pct_var - 1 )
    beta_pcpa_rook = (1-pct_mean)*( pct_mean*(1-pct_mean)/pct_var - 1 )
    log.info('using statistical rookie cmp%: a = {}, b = {}'.format(alpha_pcpa_rook, beta_pcpa_rook))
    xfvals = np.linspace(0.0, 1.0, 128)
    bins_pcpa = np.linspace(0,1,64+1)
    plt_gp = sns.distplot(data_pcpa_rook, bins=bins_pcpa,
                          kde=False, norm_hist=True,
                          hist_kws={'log':False, 'align':'mid'})
    plt.plot(xfvals, st.beta.pdf(xfvals, alpha_pcpa_rook, beta_pcpa_rook), '--', lw=2, color='violet')
    plt.title('rookie completion percentage')
    plt_gp.figure.savefig('pass_cmp_pa_rookie.png'.format())
    plt_gp.figure.show()

    
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
    
    # define a baseline model that is just a constant gaussian w/ the inclusive distribution
    cgaussmodel = bay.const_gauss_model(gp_avg_all, gp_var_all)

    memory = 0.875 # a bit of memory loss can account for overall changes. this is like the exponential window.
    memory = 1 - 1/2**4
    
    # r0,p0 = rrk,prk
    # log.info('using rookie r,p = {},{}'.format(r0,p0))
    # r0,p0 = rinc,pinc
    # log.info('using inclusive r,p = {},{}'.format(r0,p0))
    log.info('using (weighted) statistical mean and stddev')
    r0,p0 = r0stat,p0stat
    log.info('starting mean = {:.4g}'.format(r0*(1-p0)/p0))
    r0 = 1.0*r0
    # # if scale down r and beta to be on the per-game level?
    # beta0 /= 1 # scaling doesn't seem to help
    # p0p = beta0/(1+beta0)
    p0 = 1.0*p0
    log.info( 'using (possibly reduced) per-game r, b = {:.4g}, {:.4g}'.format(r0, p0))
    lrp = 1.0 # why does the bayes model have worse performance

    nbmodel = bay.neg_binomial_model(r0, p0, lrp, mem=memory)
    
    # hyperparameters for t model
    t_mu0 = rk_weight_mean
    t_mu0 = gp_avg_all
    t_nu0 = 1 # turning this down low reduces EVSE and MSE but increases the KLD
    t_alpha0 = 2.0 # needs to be > 1 to have a well-defined variance.
    # if alpha is large, the contribution to the MSE term is small, though in practice it doesn't seem to change it much
    # t_alpha0 = n_rookie_seasons # 211 rookie seasons, tho turning this up makes the variance not change much (?)
    # t_beta0 = rk_weight_stddev**2 * (t_alpha0-1)/(1+1.0/t_nu0)
    t_beta0 = gp_var_all * (t_alpha0-1)/(1+1.0/t_nu0)

    tmodel = bay.t_model(t_mu0, t_nu0, t_alpha0, t_beta0, lrp, mem=memory)

    ## model for cmp %
    lr_cmp_pct = 1.0
    mem_cmp_pct = 1-1/2**4
    pcpa_betamodel = bay.beta_model(alpha_pcpa_rook, beta_pcpa_rook, lr=lr_cmp_pct, mem=mem_cmp_pct)
    
    modeldf = pd.DataFrame()
    pcpadf = pd.DataFrame()
    
    for pname in posnames:
        pdata = posdf[posdf['name'] == pname]
        # explicitly turn into numpy arrays
        pdata_gs = pdata['games_played'].values
        pdata_pa = pdata['passing_att'].values
        pdata_pc = pdata['passing_cmp'].values
        pdata_papg = pdata_pa / pdata_gs
        pdata_pcpa = pdata_pc / pdata_pa
        # we should really do an analysis of the covariance between attempts and completion % on a per-player basis
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

        mses_pcpa_beta_sep = pcpa_betamodel.mse((pdata_pc,pdata_pa))
        klds_pcpa_beta_sep = pcpa_betamodel.kld((pdata_pc,pdata_pa))
        evs_pcpa_beta_sep = pcpa_betamodel.evs((pdata_pc,pdata_pa))
        evses_pcpa_beta_sep = pcpa_betamodel.evse((pdata_pc,pdata_pa))
        vars_pcpa_beta_sep = pcpa_betamodel.vars((pdata_pc,pdata_pa))
        res_pcpa_beta_sep = pcpa_betamodel.residuals((pdata_pc,pdata_pa))
        mses_pcpa_beta_ratio = pcpa_betamodel.mse(pdata_pcpa, weights=weights)
        klds_pcpa_beta_ratio = pcpa_betamodel.kld(pdata_pcpa, weights=weights)
        evs_pcpa_beta_ratio = pcpa_betamodel.evs(pdata_pcpa, weights=weights)
        evses_pcpa_beta_ratio = pcpa_betamodel.evse(pdata_pcpa, weights=weights)
        vars_pcpa_beta_ratio = pcpa_betamodel.vars(pdata_pcpa, weights=weights)
        res_pcpa_beta_ratio = pcpa_betamodel.residuals(pdata_pcpa, weights=weights)
        
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

            dfpcpadata = {'name':pname, 'model':'data', 'career_year':iy+1,
                        'ev':pdata_pcpa[iy], 'kld':0, 'weight':weights[iy]}
            dfpcpabeta_sep = {'name':pname, 'model':'beta_sep', 'career_year':iy+1, 'weight':1,
                            'ev':evs_pcpa_beta_sep[iy], 'kld':klds_pcpa_beta_sep[iy],
                            'mse':mses_pcpa_beta_sep[iy], 'evse':evses_pcpa_beta_sep[iy],
                            'scale':np.sqrt(vars_pcpa_beta_sep[iy]), 'residuals':res_pcpa_beta_sep[iy]}
            dfpcpabeta_ratio = {'name':pname, 'model':'beta_ratio', 'career_year':iy+1, 'weight':weights[iy],
                            'ev':evs_pcpa_beta_ratio[iy], 'kld':klds_pcpa_beta_ratio[iy],
                            'mse':mses_pcpa_beta_ratio[iy], 'evse':evses_pcpa_beta_ratio[iy],
                                'scale':np.sqrt(vars_pcpa_beta_ratio[iy]), 'residuals':res_pcpa_beta_ratio[iy]}
            pcpadf = pcpadf.append([dfpcpadata, dfpcpabeta_sep, dfpcpabeta_ratio], ignore_index=True)
        
    modeldf.reset_index(drop=True, inplace=True)
    pcpadf.reset_index(drop=True, inplace=True)
        
    # with all the data stored, add some extra stats to the data "model"
    career_long = modeldf['career_year'].max()
    for iy in range(1,career_long+1):
        mask = (modeldf['model'] == 'data') & (modeldf['career_year'] == iy)
        reldf = modeldf[mask]
        meanyr = reldf['ev'].mean()
        stdyr = reldf['ev'].std()
        modeldf.loc[mask,'scale'] = stdyr
        modeldf.loc[mask,'residuals'] = (reldf['ev'] - meanyr)/stdyr

    for df in [modeldf, pcpadf]:
        df['rmse'] = np.sqrt(df['mse'])
        df['revse'] = np.sqrt(df['evse'])
        df['norm_rmse'] = df['rmse'] / df['scale']
        df['norm_revse'] = df['revse'] / df['scale']

    log.info('total player-seasons: {}'.format(modeldf[modeldf['model'] == 'data']['weight'].sum()))
    log.info('  evaluation of pass attempt models:')
    for model in ['cgauss', 'nbinom', 'studentt']:
        for stat in ['evse', 'mse', 'kld']:
            thismodel = modeldf[modeldf['model'] == model]
            val = (thismodel[stat]*thismodel['weight']).sum()/thismodel['weight'].sum()
            if stat in ['evse', 'mse']: val = np.sqrt(val)
            log.info('{} for {} model: {:.4g}'.format(stat, model, val))
    log.info('  evaluation of completion pct models:')
    for model in ['beta_sep', 'beta_ratio']:
        for stat in ['evse', 'mse', 'kld']:
            thismodel = pcpadf[pcpadf['model'] == model]
            val = (thismodel[stat]*thismodel['weight']).sum()/thismodel['weight'].sum()
            if stat in ['evse', 'mse']: val = np.sqrt(val)
            log.info('{} for {} model: {:.4g}'.format(stat, model, val))


    plot_vars = ['ev', 'residuals'
                 , 'scale'
                 # , 'norm_rmse', 'norm_revse'
                 # , 'rmse', 'revse', 'kld'
    ]
    pltdf = pcpadf
    for var in plot_vars:
        plt.figure()
        varplt = sns.lvplot(data=pltdf, x='career_year', y=var, hue='model')
        # plt.title('passing attempts per game played')
        plt.title('completion percentage')

    plt.show(block=True)
