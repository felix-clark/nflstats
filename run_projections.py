#!/usr/bin/env python3
from get_player_stats import *
from playermodels.positions import *
from ruleset import *
from get_fantasy_points import get_points
import os.path
import argparse
import numpy as np
import numpy.random as rand

# return a player from a dataframe if a unique one exists, else return None
# we could make this a bit more robust and flexible and move it to tools.py
# (e.g. option to select by team)
def get_player_from_df(df, pname, pos=None, team=None):
    pix = (df.player == pname)
    if pos is not None: pix &= (df.pos == pos)
    if team is not None: pix &= (df.team == team)
    if pix.any():
        # pl = df.iloc[df.index[pix]]
        pl = df[pix]
        assert(len(pl) == 1)
        return pl.iloc[0]
    # try removing jr. / sr.
    if pname[-3:].lower() in ['jr.', 'sr.']:
        return get_player_from_df(df, pname[:-3].strip(), pos=pos, team=team)
    return None
    

def main():
    logging.getLogger().setLevel(logging.DEBUG)
    np.set_printoptions(precision=4)
    pd.options.display.precision = 2 # default is 6
    
    parser = argparse.ArgumentParser(description='generate projections')
    parser.add_argument('position', type=str, choices=['QB', 'RB', 'WR', 'TE'], help='which position to simulate')
    parser.add_argument('--ruleset', type=str, choices=['phys', 'dude', 'bro', 'nycfc'],
                        default='phys', help='which ruleset to use')
    parser.add_argument('--year',nargs='?', type=int, default=2018, help='what is the current year')
    parser.add_argument('--expert-touch', nargs='?', type=bool, default=False, help='scale models to meet expert consensus for rush attempts and targets')
    parser.add_argument('--n-seasons',nargs='?', type=int, default=128, help='number of seasons to simulate')

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

    scale_touch = args.expert_touch

    # get player index
    pidx = get_pos_players(pos)
    # players like Luck who didn't play last year will be ruled out here.
    # we have the expert list to compare to so we can allow another year back.
    pidx = pidx[(pidx['pos'] == pos) & (pidx['year_max'] >= current_year-2)]
    ngames = 16
    nseasons = args.n_seasons

    # get expert projections so we can adjust to touches
    expertdf = pd.read_csv('preseason_rankings/project_fp_{}_pre{}.csv'.format(pos.lower(), current_year))

    # any known suspension data
    sussdf = pd.read_csv('data/suspensions.csv')

    # data of expectation values to print out at the end (and possibly save)
    evdf = pd.DataFrame(columns=['player', 'pos'], dtype=int)

    np.random.seed(3490) # pick a constant seed so we can debug weird outcomes
    
    # for _,prow in pidx.iterrows():
    for _,exproj in expertdf.iterrows():
        pname = exproj['player']
        if exproj['fp_projection'] < 32:
            logging.debug('skipping {} as irrelevant'.format(pname))
            continue
        # pname,pid = prow[['player', 'pfr_id']]
        logging.info('training model for {}'.format(pname))

        prow = get_player_from_df(pidx, pname)
        # exproj = get_player_from_df(expertdf, pname)
        # if exproj is None:
        #     # they are probably retired; let's not waste time simulating them
        #     logging.warning('no expert projection for {}. skipping.'.format(pname))
        #     continue
                
        pmod = gen_player_model(pos)
        
        pdf = get_player_stats(prow['pfr_id']).fillna(0) if prow is not None else pd.DataFrame(columns=['player', 'pos', 'team', 'year'])
        if pdf.size == 0:
            logging.error('Could not find record for {}'.format(pname))
        stat_vars = [model.pred_var for model in pmod.models]
        for st in stat_vars:
            if st not in pdf:
                pdf[st] = 0 # set non-existent values to zero

        years = pdf['year'].unique()
        # if len(years) == 0:
            # then we need to debug why this player isn't being read, tho this should be fine for rookies
            # logging.error(' no player data for {}!'.format(pname))
            
        assert((np.diff(years) > 0).all())
        pcterrs = []
        for year in years:
            # logging.debug('training for year {}'.format(year))
            ydf = pdf[pdf['year'] == year]
            games = ydf['game_num']
            if not (np.diff(games) > 0).all():
                # this sometimes fails when players are traded mid-week. we could just pick the one with the most points (so far just manually deleting)
                logging.error('{} has a strange weekly schedule in {} (traded mid-season?)'.format(pname, year) )
            meanpts = get_points(rules, ydf).mean()
            
            for _,game in ydf.iterrows():
                # evs = pmod.evs() # expected outcome
                # expt = get_points(rules, evs) # works from dict too?
                if meanpts != 0:
                    actpt = get_points(rules, game)
                    pcterrs.append((actpt-meanpts)/meanpts)
                pmod.update_game(game)
            if year != current_year:
                pmod.new_season()

        pcterrs = np.array(pcterrs)
        if np.isnan(pcterrs).any():
            print(pcterrs)
            exit(1)
    
        # now we're done training; do simulations next
        # get the number of games a player is expected to play
        pgames = ngames # number of games this player expects to play. we'll check suspensions:
        psus = get_player_from_df(sussdf, pname, pos)
        if psus is not None:
            gsus = psus.games_suspended
            logging.info(psus.details)
            if not np.isnan(gsus):
                pgames -= int(gsus)
                logging.info(' -- {} game suspension'.format(gsus))
            else:
                logging.info('suspension time unknown.')

        if scale_touch:
            re_ev_dict = {}
            for touchvar in set(stat_vars) & set(['pass_att', 'rush_att']):
                re_ev_dict[touchvar] = exproj[touchvar]/pgames
            if 'targets' in stat_vars:
                # expert projections from this source don't have targets, just receptions
                modevs = pmod.evs()
                re_ev_dict['targets'] = modevs['targets'] * exproj['rec'] / modevs['rec'] / pgames
            pmod.revert_evs(re_ev_dict)
        
        # if pname in ['Todd Gurley', 'Ezekiel Elliott', 'Le\'Veon Bell', 'Saquon Barkley', 'Royce Freeman']:
        # if pname in ['DeAndre Hopkins', 'Odell Beckham Jr.']:
        # print(pmod)

        fpdf = pd.concat([pd.DataFrame((pmod.gen_game() for _ in range(pgames))) for _ in range(nseasons)], ignore_index=True)
        # fps = pd.concat((get_points( rules, fpdf )), ignore_index=True)
        fps = get_points( rules, fpdf )

        largegames = fps > 50
        if largegames.any():
            print(pname)
            print(fpdf[largegames])
        
        fp_2d,fp_1d,fp_med,fp_1u,fp_2u = fps.quantile((0.02275, 0.15865, 0.5, 0.84135, 0.97725))
        evdat = {key:(pgames*val) for key,val in pmod.evs().items()}
        evdat['player'] = pname
        evdat['pos'] = pos
        evdat['g'] = pgames
        # evdat['ex_pred'] = exproj['fp_projection']
        evdat['fpts_ev'] = get_points( rules, evdat )
        evdat['fpts_sim'] = fps.mean()*pgames
        evdat['fpts_med'] = fp_med
        evdat['fpts_simstd'] = fps.std()*np.sqrt(pgames)
        evdat['volatility'] = np.sqrt(np.mean(pcterrs**2))
        # if fp_med > 0:
        #     evdat['vol1'] = 0.5*(fp_1u - fp_1d)/fp_med
        #     evdat['vol2'] = 0.5*(fp_2u - fp_2d)/fp_med
        # evdat['fpts_u1'] = fp_1u
        # evdat['fpts_d1'] = fp_1d
        
        evdf = evdf.append(evdat, ignore_index=True)
        
    print(evdf.sort_values('fpts_ev', ascending=False))
    evdf.to_csv('data/{}_simulations_{}.csv'.format(pos.lower(), current_year), index=False)
    
    return

if __name__ == '__main__':
    main()
