#!/usr/bin/env python3
import logging
import os.path
from sys import argv
from time import sleep
from retrying import retry
import pandas as pd


# decorate this w/ @retry, so it can respond to failed
# requests intelligently and not overload the server
# wait 1 second between retries w/ exponential increase of wait times up to 8 seconds
# NOT USED: replaced by general fantasy scraper
@retry(wait_exponential_multiplier=1000, wait_exponential_max=8001)
def get_passing_df_pfr(year=2017):
    sleep(0.5) # let's be conservative to not spam the site
    # pfr requires more cleaning but it automatically includes all players
    url = 'https://www.pro-football-reference.com/years/{}/passing.htm'.format(year)
    tables = pd.read_html(url) # returns a list
    if (len(tables) > 1):
        logging.error('multiple tables scraped from {}'.format(url))
    # unfiltered dataframe is first entry of list
    df = tables[0].rename(columns={'Unnamed: 1':'Name', 'Yds.1':'SkYds'})
    # fulltime QBs don't have a NaN in their record entry.
    # we don't want everyone who ever throws passes, just those who start as QB in at least 1 game.
    # we also want to filter out label rows, which have a NaN for the name.
    df = df.loc[(df['QBrec'].notna()) & (df['Name'].notna())]
    # remove pro-bowl indicators
    df['Name'] = df['Name'].map(lambda n: n.strip('+*'))
    # put position in upper-case format.
    # df['Pos'] = df['Pos'].str.upper()
    # qbs on 2 teams do not have their position recorded properly
    df['Pos'] = 'QB'
    # drop the "rank" column
    df = df.drop(columns='Rk')
    # remove the bullshit made-up stats
    df = df.drop(columns=['QBR', 'AY/A', 'ANY/A'])
    return df


@retry(wait_exponential_multiplier=1000, wait_exponential_max=8001)
def get_fantasy_df_pfr(year):
    # retrieves a summary table of fantasy stats for all players
    sleep(0.5) # make sure we're not spamming the site
    url = 'https://www.pro-football-reference.com/years/{}/fantasy.htm'.format(year)
    # header = 1 causes the top line to be ignored
    # this seems simpler to deal w/ than the multi-level, which doesn't get parsed well.
    tables = pd.read_html(url, header=1) # returns a list
    if (len(tables) > 1):
        logging.error('multiple tables scraped from {}'.format(url))
    # unfiltered dataframe is first entry of list
    # drop some fantasy-related data: we can compute these ourselves based on our rulesets
    # also drop some redundant data to save space (i.e. yards per rush attempt)
    df = tables[0].drop(columns=['Y/A', 'Y/R', 'FantPt', 'PPR', 'DKPt', 'FDPt', 'VBD', 'PosRank', 'OvRank'])
    rename_dict = {'Unnamed: 1':'name',
                   'Tm':'team',
                   'FantPos':'pos',
                   'Age':'age',
                   'G':'games_played',
                   'GS':'games_started',
                   'Cmp':'passing_cmp',
                   'Att':'passing_att',
                   'Yds':'passing_yds',
                   'TD':'passing_tds',
                   'Int':'passing_int',
                   '2PM':'twoptm', # general rushing / receiving
                   '2PP':'passing_twoptm',
                   'Att.1':'rushing_att',
                   'Yds.1':'rushing_yds',
                   'TD.1':'rushing_tds',
                   'Rec':'receiving_rec',
                   'Yds.2':'receiving_yds',
                   'TD.2':'receiving_tds'
    }
    if 'Tgt' in df:
        # targets were not recorded before 1992
        rename_dict['Tgt']='receiving_tgt'

    df = df.rename(columns=rename_dict)
    # label rows have a NaN for the name.
    df = df.loc[df['name'].notna()]
    # there are some players w/out positions. these seem to be those who never started a game.
    # they typically do have games played, though. it will be extra work to work around this
    # and these guys aren't fantasy-relevant anyway, typically.
    df = df.loc[df['pos'].notna()]
    # players w/ a NaN in GS did not start any games, and will cause the next check to fail.
    # these seem to occur only when games_played = 0
    # df = df.loc[df['games_played'].notna()]
    # df = df.loc[df['games_played'].astype(int) > 0]
    # drop players who never started a game
    # df = df.loc[df['games_started'].notna()]
    # df = df.loc[df['games_started'].astype(int) > 0]
    # remove pro-bowl indicators from name field
    df['name'] = df['name'].map(lambda n: n.strip('+*'))
    # set NaN receiving targets to zero
    if 'reveiving_tgt' in df:
        df['receiving_tgt'].loc[df['receiving_tgt'].isna()] = 0
    # drop the "rank" column
    df = df.drop(columns='Rk')
    # re-index to rank based on our list
    df.reset_index(drop=True, inplace=True)
    # df['year'] = year # we can do this when we combine
    return df



if __name__ == '__main__':
    # changes the default logger
    logging.getLogger().setLevel(logging.DEBUG)
    
    if not os.path.isdir('yearly_stats'):
        logging.info('creating directory yearly_stats')
        os.mkdir('yearly_stats')

    # 1978 marks the beginning of the 16-game regular season
    first_year = int(argv[1]) if len(argv) > 1 else 1978
    logging.info('scanning back to {}'.format(first_year))
        
    for year in range(2017,first_year-1,-1):
        fantCsvName = 'yearly_stats/fantasy_{}.csv'.format(year)
        if not os.path.exists(fantCsvName):
            logging.info('scraping for {} season'.format(year))
            df = get_fantasy_df_pfr(year)
            # there are several players w/ the same name, so we must differentiate by age as well
            dupes = df.duplicated(['name','age'], keep=False)
            if dupes.any():
                logging.warning('{}{}'.format('possible duplicate entries:\n',
                                              df[dupes][['name', 'team', 'pos', 'age', 'games_played', 'games_started']].sort_values('name')))
            df.to_csv(fantCsvName)
        else:
            logging.debug('{} already exists. skipping.'.format(fantCsvName))
            
    # we can also get more detailed stats
    #     qbs = get_passing_df_pfr(year)
    #     qbs.to_csv('yearly_stats/passing_{}.csv'.format(year))
