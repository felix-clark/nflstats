#!/usr/bin/env python3
# import prediction_models as pm
from ruleset import *
import dist_fit
import bayes_models as bay
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sys import argv
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
    posdf['pass_yds_pa'] = posdf['passing_yds'] / posdf['passing_att']
    # TDs per attempt instead of per game scales out short games properly
    # posdf['pass_td_pg'] = posdf['passing_td'] / posdf['games_played'] # this is too highly correlated w/ pass attempts per game
    posdf['pass_td_pa'] = posdf['passing_td'] / posdf['passing_att']
    posdf['pass_td_pc'] = posdf['passing_td'] / posdf['passing_cmp']
    posdf['pass_td_py'] = posdf['passing_td'] / posdf['passing_yds']

    # these aren't necessarily rookies, but they represent 1st year playing in the NFL
    rookiedf = pd.concat([posdf[posdf['name'] == name].head(1) for name in posnames])

    # should be a little slicker to use numpy arrays
    data_papg_inc = posdf['pass_att_pg'].values
    data_pcpa_inc = posdf['pass_cmp_pa'].values
    data_pypc_inc = posdf['pass_yds_pc'].values
    data_pypa_inc = posdf['pass_yds_pa'].values
    data_ptdpa_inc = posdf['pass_td_pa'].values
    data_ptdpc_inc = posdf['pass_td_pc'].values
    data_papg_rook = rookiedf['pass_att_pg'].values
    data_pcpa_rook = rookiedf['pass_cmp_pa'].values
    data_pypc_rook = rookiedf['pass_yds_pc'].values
    data_pypa_rook = rookiedf['pass_yds_pa'].values
    data_ptdpa_rook = rookiedf['pass_td_pa'].values
    data_ptdpc_rook = rookiedf['pass_td_pc'].values

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

    # plt.figure()
    # rookplt = sns.pairplot(rookiedf, vars=['pass_att_pg','pass_cmp_pa','pass_yds_pc','pass_td_pc'])
    # rookplt.savefig('rookie_qb_corrs.png')
    # plt.figure()
    # rookplt = sns.pairplot(rookiedf, vars=['pass_yds_pc','pass_td_pa','pass_td_pc','pass_td_py'])
    # rookplt = sns.jointplot(['pass_td_pc','pass_td_py'], ['pass_cmp_pa','pass_yds_pc'], data=rookiedf)
    # rookplt.figure.savefig('rookie_qb_td_corrs.png')    
    tdcorr = rookiedf[['pass_att_pg', 'pass_cmp_pa', 'pass_yds_pc', 'pass_td_pa', 'pass_td_pc', 'pass_td_py']].corr()
    tdplt = sns.heatmap(tdcorr)
    plt.show()
    
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
    pct_mean_rook = np.mean(data_pcpa_rook)
    pct_var_rook = np.var(data_pcpa_rook)
    alpha_pcpa_rook = pct_mean_rook*( pct_mean_rook*(1-pct_mean_rook)/pct_var_rook - 1 )
    beta_pcpa_rook = (1-pct_mean_rook)*( pct_mean_rook*(1-pct_mean_rook)/pct_var_rook - 1 )
    log.info('using statistical rookie cmp%: beta dist pars: a = {:.4g}, b = {:.4g}'.format(alpha_pcpa_rook, beta_pcpa_rook))
    pct_mean_inc = np.mean(data_pcpa_inc)
    pct_var_inc = np.var(data_pcpa_inc)
    alpha_pcpa_inc = pct_mean_inc*( pct_mean_inc*(1-pct_mean_inc)/pct_var_inc - 1 )
    beta_pcpa_inc = (1-pct_mean_inc)*( pct_mean_inc*(1-pct_mean_inc)/pct_var_inc - 1 )
    log.info('using statistical inclusive cmp%: beta dist pars: a = {:.4g}, b = {:.4g}'.format(alpha_pcpa_inc, beta_pcpa_inc))
    # xfvals = np.linspace(0.0, 1.0, 128)
    # bins_pcpa = np.linspace(0,1,64+1)
    # plt_gp = sns.distplot(data_pcpa_rook, bins=bins_pcpa,
    #                       kde=False, norm_hist=True,
    #                       hist_kws={'log':False, 'align':'mid'})
    # plt.plot(xfvals, st.beta.pdf(xfvals, alpha_pcpa_rook, beta_pcpa_rook), '--', lw=2, color='violet')
    # plt.title('rookie completion percentage')
    # plt_gp.figure.savefig('pass_cmp_pa_rookie.png'.format())
    # plt_gp.figure.show()

    
    papg_avg_all = data_papg_inc.mean()
    papg_var_all = data_papg_inc.var()
    log.info('used for const model: average attempts per game (inclusive) = {:.4g} \pm {:.4g}'.format(papg_avg_all, np.sqrt(papg_var_all)))
    p0stat = papg_avg_all/papg_var_all
    r0stat = papg_avg_all**2/(papg_var_all-papg_avg_all)
    # the distribution for passing attempts is underdispersed compared to neg. bin.
    # it actually works well for pass completions, though
    log.info('using the inclusive stats to set r,p would yield {:.4g},{:.4g}'.format(r0stat,p0stat))
    p0stat = rk_weight_mean/rk_weight_stddev**2
    r0stat = rk_weight_mean**2/(rk_weight_stddev**2 - rk_weight_mean)
    log.info('using weighted rookie to set r,p would yield {:.4g},{:.4g}'.format(r0stat,p0stat))


    pypc_avg_rook = data_pypc_rook.mean()
    pypc_var_rook = data_pypc_rook.var()
    xfvals = np.linspace(0, 40, 128)
    bins_pypc = np.linspace(5,20,64)
    plt.figure()
    plt_gp = sns.distplot(data_pypc_rook, bins=bins_pypc,
                          kde=False, norm_hist=True,
                          hist_kws={'log':False, 'align':'mid'})
    # plt.plot(xfvals, st.beta.pdf(xfvals, alpha_pcpa_rook, beta_pcpa_rook), '--', lw=2, color='violet')
    plt.title('rookie yds/cmp')
    plt_gp.figure.savefig('pass_yds_pc_rookie.png'.format())
    plt_gp.figure.show()
    pypa_avg_rook = data_pypa_rook.mean()
    pypa_var_rook = data_pypa_rook.var()
    log.info('sigma/mu for pass yards per completion: {:.4g}'.format(np.sqrt(pypc_var_rook)/pypc_avg_rook))
    log.info('sigma/mu for pass yards per attempt: {:.4g}'.format(np.sqrt(pypa_var_rook)/pypa_avg_rook))
    
    ###############################################################
    
    # define a baseline model that is just a constant gaussian w/ the inclusive distribution
    cgaussmodel = bay.const_gauss_model(papg_avg_all, papg_var_all)

    # a memory of 1-1/N corresponds to an exponentially-falling window w/ length scale N
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
    t_mu0 = papg_avg_all
    t_nu0 = 1 # turning this down low reduces EVSE and MSE but increases the KLD
    t_alpha0 = 2.0 # needs to be > 1 to have a well-defined variance.
    # if alpha is large, the contribution to the MSE term is small, though in practice it doesn't seem to change it much
    # t_alpha0 = n_rookie_seasons # 211 rookie seasons, tho turning this up makes the variance not change much (?)
    # t_beta0 = rk_weight_stddev**2 * (t_alpha0-1)/(1+1.0/t_nu0)
    t_beta0 = papg_var_all * (t_alpha0-1)/(1+1.0/t_nu0)

    tmodel = bay.t_model(t_mu0, t_nu0, t_alpha0, t_beta0, lrp, mem=memory)

    ## model for cmp %
    lr_cmp_pct = 1.0
    mem_cmp_pct = 1-1/2**3
    # the separated model works if the learn rate is slowed
    # the ratio model is decent too if we speed up learning, but the scale is not very dynamic (all widths the same)
    # both slightly under-predict the data (positive residuals, on average); possibly because completion % tends to increase w/ time?
    # maybe using the full (non-rookie) would help -- there's not much difference. it's more important to use the separated model
    pcpa_betamodel_sep = bay.beta_model(alpha_pcpa_rook, beta_pcpa_rook, lr=lr_cmp_pct/16, mem=mem_cmp_pct)
    pcpa_betamodel_ratio = bay.beta_model(alpha_pcpa_rook, beta_pcpa_rook, lr=lr_cmp_pct*16, mem=mem_cmp_pct)

    ## models for ypc
    ypc_r0 = pypc_avg_rook**2/(pypc_var_rook - pypc_avg_rook)
    ypc_p0 = pypc_avg_rook/pypc_var_rook
    pypc_nbmodel = bay.neg_binomial_model(ypc_r0, ypc_p0, lr=1.0, mem=1-1/2**4)
    # maybe use inclusive instead of rookie?
    ypc_mu0 = pypc_avg_rook
    ypc_nu0 = 1
    ypc_alpha0 = 2
    ypc_beta0 = pypc_var_rook * (ypc_alpha0-1)/(1+1/ypc_nu0)
    pypc_tmodel = bay.t_model(ypc_mu0, ypc_nu0, ypc_alpha0, ypc_beta0, lr=1.0, mem=1.0)

    # collect models for easy abstraction
    papg_struct = {'df':pd.DataFrame(), 'desc':'pass attempts per game', 'models':{'cgauss':cgaussmodel, 'studentt':tmodel}, 'models_sep':{'nbinom':nbmodel}}
    pcpa_struct = {'df':pd.DataFrame(), 'desc':'completion percentage', 'models':{'beta_ratio':pcpa_betamodel_ratio}, 'models_sep':{'beta_sep':pcpa_betamodel_sep}}
    pypc_struct = {'df':pd.DataFrame(), 'desc':'yards per completion', 'models':{'studentt':pypc_tmodel}, 'models_sep':{'nbinom':pypc_nbmodel}}

    for pname in posnames:
        pdata = posdf[posdf['name'] == pname]
        # explicitly turn into numpy arrays
        pdata_gs = pdata['games_played'].values
        pdata_pa = pdata['passing_att'].values
        pdata_pc = pdata['passing_cmp'].values
        pdata_pyds = pdata['passing_yds'].values
        # pdata_papg = pdata_pa / pdata_gs
        # pdata_pcpa = pdata_pc / pdata_pa
        # we should really do an analysis of the covariance between attempts and completion % on a per-player basis
        career_length = pdata_gs.size
        if career_length < 2: continue

        weights = np.full(pdata_gs.shape, 1.0)
        # weights = pdata_gs / np.max(pdata_gs)
        normkld = False

        qlist = [(papg_struct,pdata_pa,pdata_gs)
                 ,(pcpa_struct,pdata_pc,pdata_pa)
                 ,(pypc_struct,pdata_pyds,pdata_pc)]
        for qstruct,numq,denq in qlist:
            ratioq = numq/denq
            df = qstruct['df']
            df = df.append(pd.DataFrame([{'name':pname, 'model':'data', 'career_year':iy+1,
                             'ev':ratioq[iy], 'kld':0,
                             'weight':weights[iy]} for iy in range(career_length)]), ignore_index=True)
            for mname,model in qstruct['models'].items():
                mses = model.mse(ratioq, weights=weights)*weights
                klds = model.kld(ratioq, weights=weights, normalize=normkld)*weights
                evses = model.evse(ratioq, weights=weights)
                evs = model.evs(ratioq, weights=weights)
                vrs = model.vars(ratioq, weights=weights)
                res = model.residuals(ratioq, weights=weights)
                df = df.append([{'name':pname, 'model':mname, 'residuals':res[iy],
                                 'ev':evs[iy], 'scale':np.sqrt(vrs[iy]),
                                 'mse':mses[iy], 'kld':klds[iy],
                                 'evse':evses[iy], 'career_year':iy+1,
                                 'weight':weights[iy]} for iy in range(career_length)], ignore_index=True)
            for mname,model in qstruct['models_sep'].items():
                mses = model.mse((numq,denq), weights=weights)*weights
                klds = model.kld((numq,denq), weights=weights, normalize=normkld)*weights
                evses = model.evse((numq,denq), weights=weights)
                evs = model.evs((numq,denq), weights=weights)
                vrs = model.vars((numq,denq), weights=weights)
                res = model.residuals((numq,denq), weights=weights)
                df = df.append([{'name':pname, 'model':mname, 'residuals':res[iy],
                                 'ev':evs[iy], 'scale':np.sqrt(vrs[iy]),
                                 'mse':mses[iy], 'kld':klds[iy],
                                 'evse':evses[iy], 'career_year':iy+1,
                                 'weight':weights[iy]} for iy in range(career_length)], ignore_index=True)
            qstruct['df'] = df # we have to re-assign

            
    papgdf = papg_struct['df']
    pcpadf = pcpa_struct['df']
    papgdf.reset_index(drop=True, inplace=True)
    pcpadf.reset_index(drop=True, inplace=True)

    # with all the data stored, add some extra stats to the data "model"
    career_long = papgdf['career_year'].max()
    for iy in range(1,career_long+1):
        mask = (papgdf['model'] == 'data') & (papgdf['career_year'] == iy)
        reldf = papgdf[mask]
        meanyr = reldf['ev'].mean()
        stdyr = reldf['ev'].std()
        papgdf.loc[mask,'scale'] = stdyr
        papgdf.loc[mask,'residuals'] = (reldf['ev'] - meanyr)/stdyr

    for df in [papgdf, pcpadf]:
        df['rmse'] = np.sqrt(df['mse'])
        df['revse'] = np.sqrt(df['evse'])
        df['norm_rmse'] = df['rmse'] / df['scale']
        df['norm_revse'] = df['revse'] / df['scale']

    log.info('total player-seasons: {}'.format(papgdf[papgdf['model'] == 'data']['weight'].sum()))
    for st in [papg_struct, pcpa_struct, pypc_struct]:
        log.info('  evaluation of {} models:'.format(st['desc']))
        mnames = [mn for mn in st['models'].keys()] + [mn for mn in st['models_sep'].keys()]
        for model in mnames:
            for stat in ['evse', 'mse', 'kld']:
                df = st['df']
                thismodel = df[df['model'] == model]
                val = (thismodel[stat]*thismodel['weight']).sum()/thismodel['weight'].sum()
                if stat in ['evse', 'mse']: val = np.sqrt(val)
                log.info('{} for {} model: {:.4g}'.format(stat, model, val))

    plot_vars = []
    if 'noplt' not in argv: plot_vars += ['ev', 'residuals', 'scale']
    if 'all' in argv:
        plot_vars += ['norm_rmse', 'norm_revse', 'rmse', 'revse', 'kld']
    
    plt_structs = []
    if 'att' in argv: plt_structs.append(papg_struct)
    if 'cmp' in argv: plt_structs.append(pcpa_struct)
    if 'yds' in argv: plt_structs.append(pypc_struct)
    for st in plt_structs:
        for var in plot_vars:
            plt.figure()
            varplt = sns.lvplot(data=st['df'], x='career_year', y=var, hue='model')
            plt.title(st['desc'])

    plt.show(block=True)
