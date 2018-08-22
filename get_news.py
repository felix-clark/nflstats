#!/usr/bin/env python3
# will retrieve player news, including suspensions
import pandas as pd

off_pos = ['QB', 'RB', 'WR', 'TE', 'K']

def main():
    url = 'https://www.pro-football-reference.com/players/injuries.htm'
    tables = pd.read_html(url, attrs={'id':'injuries'})
    assert(len(tables) == 1)
    newsdf = tables[0].rename(str.lower, axis='columns')
    tf = lambda col: 'team' if col == 'tm' else col # use 'team' instead of 'tm' for compatibility
    newsdf = newsdf.rename(tf, axis='columns')
    newsdf = newsdf[newsdf.pos.isin(off_pos)]
    newsdf.to_csv('data/news.csv', index=False)
    print('wrote to data/news.csv')
    sussdf = newsdf[newsdf['type'].str.lower().str.contains('suspension')].copy()
    sussdf['games_suspended'] = ''
    sussdf.to_csv('data/suspensions.csv', index=False)
    print('wrote to data/suspensions.csv. this will require manual parsing of the games suspended.')

if __name__ == '__main__':
    main()
