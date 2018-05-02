#!/usr/bin/env python3
import prediction_models as pm
from ruleset import *
import logging
import pandas as pd
from numpy import sqrt
from sys import argv


# get a dataframe of the relevant positional players
def get_pos_df(pos, years, datadir='./yearly_stats/', keepnames=None):
    ls_dfs = []
    for year in years:
        csvName = '{}/fantasy_{}.csv'.format(datadir,year)
        df = pd.read_csv(csvName)
        valids = df.loc[df['pos'] == pos.upper()]
        if keepnames is not None:
            valids = valids[valids['name'].isin(keepnames)]
        valids = valids.loc[valids['games_played'].astype(int) >= 1]
        if pos.lower() == 'qb':
            valids = valids.loc[valids['passing_att'].astype(int) >= 8]
        if pos.lower() == 'rb':
            valids = valids.loc[valids['rushing_att'].astype(int) >= 8]
        if pos.lower() in ['wr', 'te']:
            valids = valids.loc[valids['receiving_rec'].astype(int) >= 8]
        if valids.size == 0:
            logging.warning('No valid {} in {}'.format(pos, year))
        valids['year'] = year
        ls_dfs.append(valids)
    allpos = pd.concat(ls_dfs, ignore_index=True, verify_integrity=True)
    return allpos

def get_pos_list(pos, years, datadir='./yearly_stats/'):
    posdf = get_pos_df(pos, years, datadir)
    posnames = posdf['name'].drop_duplicates().sort_values()
    posnames.reset_index(drop=True,inplace=True)
    return posnames

def get_points( pos, rs, df ):
    """
    rs: rule set
    df: dataframe containing player stats
    pos: position
    """
    fp = 0
    if pos.lower() == 'qb':
        fp += rs.ppPY * df['passing_yds'] \
              + rs.ppPY25 * (df['passing_yds'] / 25) \
              + rs.ppPC * df['passing_cmp'] \
              + rs.ppINC * (df['passing_att'] - df['passing_cmp']) \
              + rs.ppPTD * df['passing_td'] \
              + rs.ppINT * df['passing_int']
    if pos.lower() in ['qb', 'rb']:
        fp += rs.ppRY * df['rushing_yds'] \
              + rs.ppRY10 * (df['rushing_yds'] / 10) \
              + rs.ppRTD * df['rushing_td']
    if pos.lower() in ['rb', 'wr', 'te']:
        fp += rs.ppREY * df['receiving_yds'] \
              + rs.ppREY10 * (df['receiving_yds'] / 10) \
              + rs.ppREC * df['receiving_rec'] \
              + rs.ppRETD * df['receiving_td']
    return fp


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    if len(argv) < 2:
        logging.error('usage: {} <position> <rules>'.format(argv[0]))
        exit(1)
        
    pos = argv[1].lower()
    rules = phys_league
    rulestr = argv[2] if len(argv) > 2 else 'phys'
    if rulestr == 'phys': rules = phys_league
    if rulestr == 'bro': rules = bro_league
    if rulestr == 'dude': rules = dude_league
    if rulestr in ['nyc', 'nycfc']: rules = nycfc_league
    
    # years = range(1998, 2018)
    years = range(2008, 2018) # oldest QB in this year started in 92, when targets were recorded
    posnames = get_pos_list(pos, years)
    years = range(1992, 2018)
    posdf = get_pos_df(pos, years, keepnames=posnames)
    posdf = posdf.drop(columns=['pos', 'Unnamed: 0'])

    if pos == 'qb':
        posdf['pass_att_pg'] = posdf['passing_att'] / posdf['games_played']
        posdf['cmp_pct'] = posdf['passing_cmp'] / posdf['passing_att']
        posdf['pass_yds_pc'] = posdf['passing_yds'] / posdf['passing_cmp']
        posdf['pass_td_pc'] = posdf['passing_td'] / posdf['passing_cmp']
        posdf['int_pct'] = posdf['passing_int'] / posdf['passing_att']

    if pos in ['qb', 'rb']:
        posdf['rush_att_pg'] = posdf['rushing_att'] / posdf['games_played']
        posdf['rush_yds_pa'] = posdf['rushing_yds'] / posdf['rushing_att']
        posdf['rush_td_pa'] = posdf['rushing_td'] / posdf['rushing_att']
    if pos in ['rb', 'wr', 'te']:
        posdf['rec_tgt_pg'] = posdf['receiving_tgt'] / posdf['games_played']
        posdf['rec_rec_pt'] = posdf['receiving_rec'] / posdf['receiving_tgt']
        posdf['rec_yds_pc'] = posdf['receiving_yds'] / posdf['receiving_rec']
        posdf['rec_td_pc'] = posdf['receiving_td'] / posdf['receiving_rec']
        
    # some positions will have zero targets/rush attempts. set the per-stats to zero.
    posdf.fillna(0, inplace=True)
    
    checkWindowModels = False
    if checkWindowModels:
        logging.info('checking models')
    
        n_data = 0
        stats_to_predict = []
        if pos == 'qb': stats_to_predict = ['games_played', 'pass_att_pg', 'cmp_pct', 'pass_yds_pc', 'pass_td_pc', 'int_pct'
                                            ,'rush_att_pg', 'rush_yds_pa', 'rush_td_pa']
        if pos == 'rb': stats_to_predict = ['games_played', 'rush_att_pg', 'rush_yds_pa', 'rush_td_pa',
                                            'rec_tgt_pg', 'rec_rec_pt', 'rec_yds_pc', 'rec_td_pc']
        if pos in ['wr', 'te']: stats_to_predict = ['games_played', 'rec_tgt_pg', 'rec_rec_pt', 'rec_yds_pc', 'rec_td_pc']
        stats_to_predict = ['games_played']
        avg_st = {st:posdf[st].mean() for st in stats_to_predict}
        std_st = {st:posdf[st].std() for st in stats_to_predict}
        md_gp_const = lambda data: 0.4*16 + 0.6*pm.mean(data) # compare games played models to a constant, full value
        const_gp_sse = 0
        naive_sse = {key:0 for key in stats_to_predict}
        md_naive = lambda data: pm.naive(data)
        mean_sse = {key:0 for key in stats_to_predict}
        md_mean = lambda data: pm.mean(data)
        alphas = [0.2, 0.25, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9]
        # exp_sse = {a:{key:0 for key in stats_to_predict} for a in alphas}
        exp_sse = {key:{a:0 for a in alphas} for key in stats_to_predict}
        md_exp = {}
        for a in alphas:
            # we won't count rookie years in the squared error calculation.
            # thus we won't provide a default value, since they're dependent on the statistic anyway.
            md_exp[a] = lambda data: pm.exp_window(data, alpha=a)
    
        for name in posnames:
            pdf = posdf[posdf['name'] == name]
            n_data += pdf.size - 1
            const_gp_sse += pm.sse_subseries(pdf['games_played'], md_gp_const)
            for st in stats_to_predict:
                naive_sse[st] += pm.sse_subseries(pdf[st], md_naive)
                mean_sse[st] += pm.sse_subseries(pdf[st], md_mean)
                for a in alphas:
                    exp_sse[st][a] += pm.sse_subseries(pdf[st], md_exp[a])

        
        for st in stats_to_predict:
            logging.info('\n  model performance for {}:'.format(st))
            logging.info('{:.4g} \pm {:.4g}'.format(avg_st[st], std_st[st]))
            if st == 'games_played':
                logging.info('constant (16) RMSE: {:.4g}'.format(sqrt(const_gp_sse/n_data)))
            logging.info('naive RMSE: {:.4g}'.format(sqrt(naive_sse[st]/n_data)))
            logging.info('mean RMSE: {:.4g}'.format(sqrt(mean_sse[st]/n_data)))
            minalpha = min(exp_sse[st].items(), key=lambda x: x[1])
            logging.info('exp[{}] RMSE: {:.5g}'.format(minalpha[0], sqrt(minalpha[1]/(n_data-1))))
            
    # compare the dumb prediction methodology to experts for the last year
    # the dumb model really doesn't work well for RBs, seemingly because it's more a question of who gets touches
    # the year to compare predictions with
    current_year = 2017
    current_posnames = get_pos_list(pos, [current_year])
    current_posdf = posdf[(posdf['name'].isin(current_posnames)) & (posdf['year'] < current_year)]

    pred_pos = []
    for name in current_posnames:
        pldat = current_posdf[current_posdf['name'] == name]
        if pldat.size == 0:
            print('skipping {}'.format(name))
            continue
        pred_data = pm.dumb_pos_predictions(pos, pldat)
        pred_data['name'] = name
        pred_pos.append(pred_data)
    pred_posdf = pd.DataFrame(pred_pos)
    # pred_posdf['dumb_proj'] = get_points(pos, rules, pred_posdf) / pred_posdf['games_played'] # effectively gives ppg and scales up by 15
    pred_posdf['dumb_proj'] = get_points(pos, rules, pred_posdf) / 15 # accounts for our prediction of games played. works better for TE, WR, RB, but QB is not improved

    real_posdat = get_pos_df(pos, [current_year])
    real_posdat['fantasy_ppg'] = get_points(pos, rules, real_posdat) / real_posdat['games_played']

    ex_proj = pd.read_csv('./preseason_rankings/project_fp_{}_pre{}.csv'.format(pos, current_year))
    ex_proj['expert_proj'] = get_points(pos, rules, ex_proj)
    
    pred_posdf = pd.merge(pred_posdf.drop(columns='games_played'), ex_proj[['name','expert_proj']], on='name')
    pred_posdf = pd.merge(pred_posdf, real_posdat[['name','games_played','fantasy_ppg']], on='name')
    pred_posdf['expert_proj'] /= 16

    # hardcode in some fixes for suspensions
    if current_year == 2017:
        pred_posdf.loc[pred_posdf['name']=='Ezekiel Elliott', 'expert_proj'] *= 16/10
        pred_posdf.loc[pred_posdf['name']=='Doug Margin', 'expert_proj'] *= 16/12
        pred_posdf.loc[pred_posdf['name']=='Michael Floyd', 'expert_proj'] *= 16/12
        pred_posdf.loc[pred_posdf['name']=='Austin Seferian-Jenkins', 'expert_proj'] *= 16/12
    if current_year == 2016:
        # incomplete list
        pred_posdf.loc[pred_posdf['name']=='Tom Brady', 'expert_proj'] *= 16/12
        pred_posdf.loc[pred_posdf['name']=='Le\'veon Bell', 'expert_proj'] *= 16/13
    
    pred_posdf = pred_posdf[(pred_posdf['games_played'] >= 1) & (pred_posdf['expert_proj'] > 1) & (pred_posdf['fantasy_ppg'] > 1)]
    pred_posdf.sort_values('fantasy_ppg', inplace=True, ascending=False)
    pred_posdf.reset_index(drop=True,inplace=True)
    
    pd.options.display.precision = 3
    logging.info('2017 comparison:\n' + str(pred_posdf[['name', 'dumb_proj', 'expert_proj', 'fantasy_ppg', 'games_played']]))
    # logging.info('2017 comparison:\n' + str(pred_posdf))
    
    # when computing the error, weight by games played
    expert_err = (pred_posdf['expert_proj'] - pred_posdf['fantasy_ppg'])
    dumb_err = (pred_posdf['dumb_proj'] - pred_posdf['fantasy_ppg'])
    gp = pred_posdf['games_played']
    logging.info( 'expert MSE: {}'.format( sqrt(sum(expert_err**2 * gp)/gp.sum()) ) )
    logging.info( 'dumb MSE: {}'.format( sqrt(sum(dumb_err**2 * gp)/gp.sum()) ) )
    logging.info( 'expert MAE: {}'.format( sum(expert_err.abs()* gp)/gp.sum() ) )
    logging.info( 'dumb MAE: {}'.format( sum(dumb_err.abs() * gp)/gp.sum() ) )

    # next year's predictions
    pred_ny = []
    rel_ny_indices = posdf['name'].isin(current_posnames)
    ny_posdf = posdf[rel_ny_indices]
    for name in current_posnames:
        pred_data = pm.dumb_pos_predictions(pos, ny_posdf[ny_posdf['name'] == name])
        pred_data['name'] = name
        pred_ny.append(pred_data)
    pred_ny = pd.DataFrame(pred_ny)
    # logging.info('2018 dumb predictions:\n' + str(pred_ny))
    pred_ny['dumb_ppg'] = get_points(pos, rules, pred_ny) / pred_ny['games_played']
    pred_ny.sort_values('dumb_ppg', inplace=True, ascending=False)
    pred_ny.reset_index(inplace=True)
    pred_ny['dumb_proj'] = pred_ny['dumb_ppg'] * pred_ny['games_played']
    # should look at individual columns more carefully - there are some head-scratchers
    logging.info('2018 dumb predictions:\n' + str(pred_ny[['name', 'dumb_ppg', 'games_played', 'dumb_proj']].head(32)))
    # logging.info('2018 dumb predictions:\n' + str(pred_ny.head(32)))
