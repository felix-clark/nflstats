#!/usr/bin/env python3
from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import logging
import os

def main():
    logging.getLogger().setLevel(logging.DEBUG)
    if not os.path.isdir('data'):
        os.mkdir('data')
    if not os.path.isdir('data/draft'):
        os.mkdir('data/draft')
    
    years = 1992,2018+1 # 1992 is first year that targets were recorded
    for year in range(*years):
        logging.info('scraping for year {}'.format(year))
        url = 'https://www.pro-football-reference.com/years/{year}/draft.htm'.format(year=year)
        page = urlopen(url)
        soup = BeautifulSoup(page, 'lxml')
        # select() instead of find() returns a list
        table_rows = soup.select('#drafts tr')
        pd = get_players(table_rows, ignore_cols=['career_av', 'draft_av', 'college_id', 'college_link'])
        pd['year'] = year
        fout = 'data/draft/class_{year}.csv'.format(year=year)
        pd.to_csv(fout, index=False)
    

def get_players(table_rows, ignore_cols=None):
    """
    get the player data from the table data (td) elements in the table rows (tr)
    """
    if ignore_cols is None:
        ignore_cols = []
        
    df = pd.DataFrame()
    
    for row in table_rows:
        # skip some empty rows, like the label rows
        if(len(row.find_all('td')) == 0):
            continue
        player_dict = {}
        
        for thing in row:
            key = thing.get('data-stat')
            value = thing.get_text()
            if key == 'player':
                if value[-1] == '*': value = value[:-1]
                # remove Hall of Fame designation from some players' names
                if value.endswith(' HOF'): value = value[:-4]
            if key not in ignore_cols:
                player_dict[key] = value
            if key == 'player':
                any_links = thing.find('a', href=True)
                if any_links is not None:
                    pid = any_links['href'][:-4].split('/')[-1]
                    # print(pid)
                    player_dict['pfr_id'] = pid
                else:
                    # this usually corresponds with a player not being a primary starter anyway
                    logging.warning('Could not find pro-football-reference id for {}'.format(value))
        # print(row.find_all('td', attrs={'data-stat':'player'}))
        
        # we can skip players w/ no years as a primary starter
        # if int(player_dict['years_as_primary_starter']) > 0:
        # safer to just make sure players have an ID
        if 'pfr_id' in player_dict:
            df = df.append(player_dict, ignore_index=True)
    return df
    
if __name__ == '__main__':
    main()
