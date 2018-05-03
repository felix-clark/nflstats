#!/usr/bin/env python3
# import prediction_models as pm
# from getPoints import *
from ruleset import *
import dist_fit
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

# a model that uses the average
def gp_mse_model_const(data, const=0, weights=None):
    return (const-data)**2

# model that uses mean of past
def gp_mse_model_mean(data, default=0):
    mses = []
    if data.size == 0:
        return default
    for i_d in range(data.size):
        mean_so_far = data.iloc[:i_d].mean() if i_d > 0 else default
        mses.append( (mean_so_far-data.iloc[i_d])**2 )
    return np.array(mses)

# beta-binomial model w/ bayesian updating of parameters
# alpha -> alpha + gp
# beta -> beta + (n-gp)
def gp_mse_model_bb(data, alpha0, beta0, lr=1.0, n=16):
    # lr can be used to suppress learning
    # we can also apply multiplicatively after adding, which will result in variance decay even in long careers
    assert((data >= 0).all() and (data <= n).all())
    mses = []
    alpha,beta = alpha0,beta0
    # domain of summation for EV computation:
    support = np.arange(0,n+1)
    for d in data:
        probs = dist_fit.beta_binomial( support, n, alpha, beta )
        mses.append( sum(probs*(support-d)**2) )
        alpha += lr*d
        beta += lr*(n-d)
    # logging.debug('alpha, beta = {},{}'.format(alpha,beta))
    return np.array(mses)
        

    # final alpha and beta for future predictions are computed by here, but are not used

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    if len(argv) < 2:
        logging.error('usage: {} <position>'.format(argv[0]))
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
    logging.debug('rookies:\n' + str(rookiedf[['name','year','team']]))

    # while other positions switch in-and-out, a QB is really relevant only if he is starting.
    # looking at starts rather than plays eliminates some noise w/ backups
    gp_stat = 'games_started' if pos == 'qb' else 'games_played'
    data_gp_rook = rookiedf[gp_stat]
    data_gp = posdf[gp_stat]
    
    
    _,(ark,brk),cov,llpdf = dist_fit.to_beta_binomial( (0,maxgames), data_gp_rook )
    logging.info('rookie: alpha = {}, beta = {}, LL per dof = {}'.format(ark, brk, llpdf))
    logging.info('covariance:\n' + str(cov))
    _,(ainc,binc),cov,llpdf = dist_fit.to_beta_binomial( (0,maxgames), data_gp )
    logging.info('all: alpha = {}, beta = {}, LL per dof = {}'.format(ainc, binc, llpdf))
    logging.info('covariance:\n' + str(cov))
    
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

    # logging.info('using rookie a,b = {},{}'.format(ark,brk))
    # alpha0,beta0 = ark,brk
    # logging.info('using inclusive a,b = {}'.format(ainc, binc)) # does a bit worse
    # alpha0,beta0 = ainc,binc
    # m1 = data_gp_rook.mean()
    # m2 = (data_gp_rook**2).mean()
    logging.info('using moment method for combined a,b = {},{}'.format(ark,brk))
    m1 = data_gp.mean()
    m2 = (data_gp**2).mean()
    denom = maxgames*(m2/m1 - m1 - 1) + m1
    alpha0 = (maxgames*m1 - m2)/denom
    beta0 = (maxgames-m1)*(maxgames - m2/m1)/denom
    logging.info('starting mean = {}'.format(maxgames*alpha0/(alpha0+beta0)))
                 
    gp_avg_all = data_gp.mean()
    logging.info('used for const model: average games_played = {}'.format(gp_avg_all))
    
    # entries = []
    mse_bb_sum = 0.
    mse_const_sum = 0.
    mse_mean_sum = 0.
    mse_total_n = 0
    
    for pname in posnames:
    # in this loop we should instead evaluate our bayesian model
        pdata = posdf[posdf['name'] == pname][gp_stat]
        gp_mses_bb = gp_mse_model_bb(pdata, 16*alpha0, 16*beta0, lr=0.5)
        gp_mses_const = gp_mse_model_const(pdata, gp_avg_all)
        gp_mses_mean = gp_mse_model_mean(pdata, gp_avg_all) # could also use rookie average

        # logging.info('{} {} {}'.format(pdata.size, gp_mses_bb.size, gp_mses_const.size))
        mse_total_n += pdata.size
        mse_bb_sum += gp_mses_bb.sum()
        mse_const_sum += gp_mses_const.sum()
        mse_mean_sum += gp_mses_mean.sum()
        # get an a and b parameter for each player, to form a prior distribution for the values
    #     entry = {'name':pname}
    #     career_length = pdata.size
    #     if career_length < 4:
    #         logging.debug('{} had short career of length {} - will be overdetermined'.format(pname, career_length))
    #         continue
    #     success,(pa,pb),pcov,llpdf = dist_fit.to_beta_binomial( (0,maxgames), pdata )
    #     if not success:
    #         continue
    #     entry['alpha'] = pa
    #     entry['beta'] = pb
    #     entry['mean'] = maxgames*pa/(pa+pb)
    #     entry['rho'] = 1.0/(pa+pb+1)
    #     entry['var'] = entry['mean']*entry['rho']*(pa+pb+maxgames)/(pa+pb)
    #     entry['career_length'] = career_length
    #     entries.append(entry)
    # prior_gp_df = pd.DataFrame(entries)
    # plt_abprior = sns.pairplot(prior_gp_df, hue='career_length', vars=['alpha', 'beta', 'rho','career_length'])
    # plt_abprior.savefig('gp_ab_prior_{}.png'.format(pos))

    # right now bayes does worse than just using the average
    logging.info('RMSE for const model: {}'.format(np.sqrt(mse_const_sum/mse_total_n)))
    logging.info('RMSE for mean model: {}'.format(np.sqrt(mse_mean_sum/mse_total_n)))
    logging.info('RMSE for bayes model: {}'.format(np.sqrt(mse_bb_sum/mse_total_n)))
    logging.info('total player-seasons: {}'.format(mse_total_n))
    
    # plt.show(block=True)
