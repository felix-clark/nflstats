#!/usr/bin/env python3
# will retrieve player news, including suspensions
import pandas as pd
import os

off_pos = ['QB', 'RB', 'WR', 'TE', 'K']


def main():
    nflstats_dir = os.getenv('NFLSTATS_DIR') or '.'
    print("WARNING: The target link will probably not work anymore.")
    # This link doesn't seem to work the same anymore
    url = 'https://www.pro-football-reference.com/players/injuries.htm'
    tables = pd.read_html(url, attrs={'id': 'injuries'})
    assert len(tables) == 1
    newsdf = tables[0].rename(str.lower, axis='columns')

    def tf(col):
        """use 'team' instead of 'tm' for compatability"""
        return 'team' if col == 'tm' else col
    # tf = lambda col: 'team' if col == 'tm' else col
    newsdf = newsdf.rename(tf, axis='columns')
    # sometimes the position information is missing from the suspensions
    # newsdf = newsdf[newsdf.pos.isin(off_pos)]
    newsdf.to_csv(f'{nflstats_dir}/data/news.csv', index=False)
    print('wrote to data/news.csv')
    sussdf = newsdf[newsdf['type'].str.lower().str.contains('suspension')].copy()
    print('suspensions:')
    sussdf['games_suspended'] = ''
    sussdf.to_csv(f'{nflstats_dir}/data/suspensions.csv', index=False)
    print('wrote to data/suspensions.csv. this will require manual parsing of the games suspended, and possibly of the positions.')


if __name__ == '__main__':
    main()
