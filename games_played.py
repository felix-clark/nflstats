#!/usr/bin/env python3
# import prediction_models as pm
# from getPoints import *
from ruleset import *
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from pandas.plotting import autocorrelation_plot
from numpy import sqrt
from sys import argv
from scipy.stats import *
# from scipy.stats import beta

# get a dataframe of the relevant positional players
def get_pos_df(pos, years, datadir='./yearly_stats/'):
    ls_dfs = []
    for year in years:
        csvName = '{}/fantasy_{}.csv'.format(datadir,year)
        df = pd.read_csv(csvName)
        valids = df.loc[df['pos'] == pos.upper()]
        valids = valids.loc[valids['games_started'].astype(int) >= 1]
        if pos.lower() == 'qb':
            valids = valids.loc[valids['passing_att'].astype(int) >= 8]
        if pos.lower() == 'rb':
            valids = valids.loc[valids['rushing_att'].astype(int) >= 8]
        if pos.lower() in ['wr', 'te']:
            valids = valids.loc[valids['receiving_rec'].astype(int) >= 8]
        valids['year'] = year
        ls_dfs.append(valids)
    allpos = pd.concat(ls_dfs, ignore_index=True, verify_integrity=True).drop_duplicates()
    return allpos

def get_pos_list(pos, years, datadir='./yearly_stats/'):
    posdf = get_pos_df(pos, years, datadir)
    posnames = posdf['name'].sort_values()
    posnames.reset_index(drop=True,inplace=True)
    return posnames


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    if len(argv) < 2:
        logging.error('usage: {} <position>'.format(argv[0]))
        exit(1)
        
    pos = argv[1].lower()
    
    years = range(1990, 2018)
    posnames = get_pos_list(pos, years)
    years = range(1980, 2018)
    posdf = get_pos_df(pos, years)
    posdf = posdf.drop(columns=['pos', 'Unnamed: 0'])

    # we literally just care about games played for this script
    sns.set()
    data_gp = posdf['games_played']
    plt_gp = sns.distplot(data_gp, bins=range(0,16+2),
                          kde=False, norm_hist=True, fit=beta,
                          hist_kws={'log':False, 'align':'mid'})
    plt_gp.figure.savefig('games_played_{}.png'.format(pos))
    plt_gp.figure.show()
    # we can't use "fit" w/ discrete distributions. need to customize beta-binomial.
    a1, b1, loc1, scale1 = beta.fit(data_gp, floc=0, fscale=17)
    print(a1, b1, loc1, scale1)
    plt.show(block=True)
    
