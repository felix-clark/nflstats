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

    # we only care about games played for this script
    # Jerry Rice switch teams in 2004 and played 17 games, so we'll just set his to 16    
    gt16 = posdf['games_played'] > 16
    posdf.loc[gt16,'games_played'] = 16

    # rookiedat = []
    # for name in posnames:
    #     rookiedat.append(posdf[posdf['name'] == name].head(1))
    rookiedf = pd.concat([posdf[posdf['name'] == name].head(1) for name in posnames])
    logging.debug('rookies:\n' + str(rookiedf[['name','year','team']]))

    # while other positions switch in-and-out, a QB is really relevant only if he is starting.
    # looking at starts rather than plays eliminates some noise w/ backups
    gp_stat = 'games_started' if pos == 'qb' else 'games_played'
    # data_gp = rookiedf[gp_stat]
    data_gp = posdf[gp_stat]
    
    sns.set()
    
    
    _,(fa,fb),cov,llpdf = dist_fit.to_beta_binomial( (0,16), data_gp )
    logging.info('alpha = {}, beta = {}, LL per dof = {}'.format(fa, fb, llpdf))
    logging.info('covariance:\n' + str(cov))
    xfvals = np.linspace(-0.5, 16.5, 128)
    
    plt_gp = sns.distplot(data_gp, bins=range(0,16+2),
                          kde=False, norm_hist=True,
                          hist_kws={'log':False, 'align':'left'})
    plt.plot(xfvals, dist_fit.beta_binomial(xfvals, 16, fa, fb), '--', lw=2, color='violet')
    plt_gp.figure.savefig('{}_{}.png'.format(gp_stat, pos))
    plt_gp.figure.show()
    # the covariance matrix from the fit can provide a bayesian prior for a and b (gamma functions? but they are correlated)
    # for QBs we might want to adjust for year-in-league, or just filter on those which started many games
    entries = []
    for pname in posnames:
        # get an a and b parameter for each player, to form a prior distribution for the values
        entry = {'name':pname}
        pdata = posdf[posdf['name'] == pname][gp_stat]
        career_length = pdata.size
        if career_length < 4:
            logging.info('{} had short career of length {} - will be overdetermined'.format(pname, career_length))
            continue
        success,(pa,pb),pcov,llpdf = dist_fit.to_beta_binomial( (0,16), pdata )
        if not success:
            continue
        entry['alpha'] = pa
        entry['beta'] = pb
        entry['mean'] = 16*pa/(pa+pb)
        entry['rho'] = 1.0/(pa+pb+1)
        entry['var'] = entry['mean']*entry['rho']*(pa+pb+16)/(pa+pb)
        entry['career_length'] = career_length
        entries.append(entry)
    prior_gp_df = pd.DataFrame(entries)
    plt_abprior = sns.pairplot(prior_gp_df, hue='career_length', vars=['alpha', 'beta', 'mean', 'rho','career_length'])
    plt_abprior.savefig('gp_ab_prior_{}.png'.format(pos))
    # plt_abprior.show()
    plt.show(block=True)
