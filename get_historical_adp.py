#!/usr/bin/env python3
import pandas as pd
import logging
import os.path

def main():
    logging.getLogger().setLevel(logging.DEBUG)
    if not os.path.isdir('adp_historical'):
        logging.info('creating directory adp_historical')
        os.mkdir('adp_historical')
    
    # first year on this site is 1998. first year with any data is 1999.
    years = (1999, 2018+1)
    poscounts = [('QB',20), ('RB', 64), ('WR', 64), ('TE', 20), ('PK', 16), ('Def',16)]
    for year in range(*years):
        logging.info('working on year {}'.format(year))
        for pos,count in poscounts:
            results = pd.read_html('http://www03.myfantasyleague.com/{}/adp?COUNT={}&POS={}&ROOKIES=0&INJURED=1&CUTOFF=5&FRANCHISES=12&IS_PPR=1&IS_KEEPER=0&IS_MOCK=0&TIME='.format(year,count,pos), header=0)
            results = [r for r in results if r.shape[0] >= 16]
            if len(results) != 1:
                if pos == 'Def':
                    # defense ADP doesn't go back all the way. who cares.
                    continue;
                else:
                    exit(1)
            result = results[0]
            result = result.drop(columns=['#', 'Min. Pick', 'Max. Pick'])
            columns = [c for c in result.columns if c != 'Player']
            result['name'] = result['Player'].apply(get_name)
            result['team'] = result['Player'].apply(get_team)
            result['pos'] = result['Player'].apply(get_pos)
            result = result[['name', 'team', 'pos'] + columns]
            result.to_csv('adp_historical/adp_{}_{}.csv'.format(pos.lower(), year))
    
def get_name(player):
    sc = player.split(',')
    assert(len(sc) == 2)
    ln = sc[0]
    splits = sc[1].strip(' ').split(' ')
    fn = ' '.join(splits[:-2])
    name = ' '.join([fn, ln])
    return name

def get_team(player):
    splits = player.split(' ')
    return splits[-2]

def get_pos(player):
    splits = player.split(' ')
    return splits[-1]
            
if __name__ == '__main__':
    main()
