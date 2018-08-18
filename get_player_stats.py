from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import logging
import os

def get_player_stats(pfrid, years):
    """
    get the dataframe of the player's weekly stats
    pfrid: pro-football-reference id (e.g. GurlTo01)
    """
    logging.getLogger().setLevel(logging.DEBUG)
    
    f = 'data/players/{id}.csv'.format(id=pfrid)
    df = None
    if os.path.isfile(f):
        try:
            df = pd.read_csv(f)
        except e:
            logging.error('could not read {}: {}'.format(f, e))
    if df is None:
        logging.info('making cache for {}'.format(pfrid))
        df = _make_cache(pfrid, years)
        
    return df

def _make_cache(pfrid, years):
    if not os.path.isdir('data'):
        logging.info('creating data/')
        os.mkdir('data')
    if not os.path.isdir('data/players'):
        logging.info('creating data/players/')
        os.mkdir('data/players')

    # don't save some useless of redundant data
    ignore_cols = ['game_date', 'age',
                   'pass_cmp_perc',
                   'all_td',
                   'scoring',
                   ]
    
    df = pd.DataFrame()
    
    for year in years:
        url = 'https://www.pro-football-reference.com/players/{}/{}/gamelog/{year}/'.format(pfrid[0], pfrid, year=year)
        html = urlopen(url)
        soup = BeautifulSoup(html, 'lxml')
        table_rows = soup.select('#stats tr')
        stats = _get_stats(table_rows, ignore_cols)
        stats['year'] = year
        df = df.append(stats, ignore_index=True)
        
    f = 'data/players/{id}.csv'.format(id=pfrid)
    df.to_csv(f, index=False)
    return df

def _get_stats(table_rows, ignore_cols=None):
    """
    get the player data from the table data (td) elements in the table rows (tr)
    """
    if ignore_cols is None:
        ignore_cols = []

    # set the first few columns to an organized order
    df = pd.DataFrame(columns=['year', 'game_num', 'team', 'game_location', 'opp', 'game_result'])
    
    for row in table_rows:
        # skip some empty rows, like the label rows
        if(len(row.find_all('td')) == 0):
            continue
        player_dict = {}
        for thing in row:
            key = thing.get('data-stat')
            value = thing.get_text()
            if key not in ignore_cols and '_pct' not in key and 'yds_per' not in key:
                player_dict[key] = value
        rker = player_dict.pop('ranker')
        if rker:
            df = df.append(player_dict, ignore_index=True)
    return df


if __name__ == '__main__':
    # just do some things to test
    tgid = 'GurlTo01'
    tg = get_player_stats(tgid, [2015, 2016, 2017])
    print(tg)
