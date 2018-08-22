#!/usr/bin/env python3
from get_player_stats import *
from playermodels.positions import *
from ruleset import *
from get_fantasy_points import get_points_from_data_frame
import os.path
import argparse
import numpy as np
import numpy.random as rand

def main():
    logging.getLogger().setLevel(logging.DEBUG)
    np.set_printoptions(precision=4)
    pd.options.display.precision = 2 # default is 6
    
    parser = argparse.ArgumentParser(description='generate projections')
    parser.add_argument('position', type=str, choices=['QB', 'RB', 'WR', 'TE'], help='which position to simulate')
    parser.add_argument('--ruleset', type=str, choices=['phys', 'dude', 'bro', 'nycfc'],
                        default='phys', help='which ruleset to use')
    parser.add_argument('--year',nargs='?', type=int, default=2018, help='what is the current year')

    args = parser.parse_args()
    pos = args.position
    current_year = args.year
    
    if args.ruleset == 'phys':
        rules = phys_league
    if args.ruleset == 'dude':
        rules = dude_league
    if args.ruleset == 'bro':
        rules = bro_league
    if args.ruleset == 'nycfc':
        rules = nycfc_league


    # get player index
    pidx = get_pos_players(pos)
    pidx = pidx[(pidx['pos'] == pos) & (pidx['year_max'] >= current_year-1)]
    ngames = 16
    nseasons = 128

    evdf = pd.DataFrame(columns=['player', 'pos'], dtype=int)
    
    for _,prow in pidx.iterrows():
        pname,pid = prow[['player', 'pfr_id']]
        logging.info('training model for {}'.format(pname))
        pdf = get_player_stats(prow['pfr_id']).fillna(0)
        pmod = gen_player_model(pos)
        for model in pmod.models:
            if model.pred_var not in pdf:
                pdf[model.pred_var] = 0
        years = pdf['year'].unique()
        assert((np.diff(years) > 0).all())
        for year in years:
            ydf = pdf[pdf['year'] == year]
            games = ydf['game_num']
            assert((np.diff(games) > 0).all()) # this might fail when players are traded mid-week
            for _,game in ydf.iterrows():
                pmod.update_game(game)
            pmod.new_season()
        fpdf = pd.DataFrame([pmod.gen_game() for _ in range(ngames)])
        fps = pd.concat((get_points_from_data_frame( rules, fpdf ) for _ in range(nseasons)), ignore_index=True)
        # print('std dev / mean: {}'.format(fps.std()/fps.mean()))
        evdat = {key:(ngames*val) for key,val in pmod.evs().items()}
        evdat['player'] = pname
        evdat['pos'] = pos
        evdat['fpts_ev'] = get_points_from_data_frame( rules, evdat )
        evdat['fpts_sim'] = fps.mean()
        evdat['fpts_simstd'] = fps.std()
        # evdat['fpts'] = fps.sum() # this is random
        evdf = evdf.append(evdat, ignore_index=True)
        
    print(evdf.sort_values('fpts_ev', ascending=False))
    
    return

if __name__ == '__main__':
    main()
