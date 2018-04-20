#!/usr/bin/env python3
import logging
import pandas as pd


def get_qb_list(years=None, datadir='./yearly_stats/'):
    if years is None:
        years = range(2015,2018) # just pick a modest range for testing
    qbnames = pd.Series()
    for year in years:
        csvName = '{}/fantasy_{}.csv'.format(datadir,year)
        df = pd.read_csv(csvName)
        validqbs = df.loc[df['pos'] == 'QB']
        # we might want more filters, but for now use all in the dataset
        # to be a fantasy factor, a qb should start in *at least* some number of games in at least *one* season
        validqbs = validqbs.loc[validqbs['games_started'].astype(int) >= 4]
        qbn = validqbs['name']
        print('{} qbs in {}'.format(len(qbn), year))
        qbnames = pd.concat([qbnames,qbn], ignore_index=True, verify_integrity=True).drop_duplicates()
    qbnames.sort_values(inplace=True)
    qbnames.reset_index(drop=True,inplace=True)
    return qbnames
    

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    # we can get about 300 unique QBs going back to 1980
    # but we don't want to go all the way to the edge; we
    # want to how well active QBs from a given range did before that range.
    years = range(1990, 2018)
    qbnames = get_qb_list(years)
    logging.info(qbnames)
