import numpy as np

def naive(data, default=0, weights=None):
    if data.size == 0: return default
    return data.iloc[-1]

def mean(data, default=0, weights=None):
    if data.size == 0: return default
    if weights is None:
        return data.mean()
    else:
        return sum(data*weights)/weights.sum()

def exp_window(data, alpha=0.5, default=0, weights=None):
    if (alpha == 0): return naive(data, default)
    if (alpha == 1): return mean(data, default)
    T = data.size
    if T == 0: return default
    if weights is None:
        ws = np.array([alpha**i for i in range(T)])[::-1]
        norm = (1.0-alpha)/(1.0-alpha**T)
    else:
        ws = np.array([alpha**i for i in range(T)])[::-1] * weights
        norm = 1.0/(ws.sum())
    return norm * sum(ws*data)

# sum of squared error given a model
def sse_subseries(data, model):
    return sum(((model(data.iloc[:sl])-data.iloc[sl])**2 for sl in range(1,len(data)-1)))

# predictions using exponential averages w/ parameters picked by observation
def dumb_qb_predictions(df):
    '''
    input dataframe
    '''
    # make sure data is sorted sequentially by year
    df = df.sort_values('year')
    predictions = {}
    # assume by default that all 16 games are played
    # only 15 games are relevant for fantasy though
    predictions['games_played'] = 15 # games played (or started) are just hard to predict from past data alone. keep the default here for now
    # the QB MSE for points is not improved by trying to account for games played, unless we make it quite small
    # predictions['games_played'] = 15.0/16.0*(0.9*16 + 0.1*mean(df['games_played']))
    # bayesian model using beta-binomial for predictive posterior:
    # instead of scaling we should add the part in the 16 bin to 15 -- will predict more games played
    # predictions['games_played'] = 15.0*(0.816 + 0.0*sum(df['games_played']))/(0.816 + 1.03 + 0.0*df.shape[0]*16.0)
    wt = df['games_played'] # weight average by games played per year
    data_pa_pg = df['passing_att'] / df['games_played']
    # these defaults are for all QBs, not rookies
    pred_pa_pg = exp_window(data_pa_pg, alpha=0.6, default=26.4, weights=wt)
    data_cmp_pct = df['passing_cmp'] / df['passing_att']
    pred_cmp_pct = exp_window(data_cmp_pct, alpha=0.6, default=0.582, weights=wt)
    data_yds_pc = df['passing_yds'] / df['passing_cmp']
    pred_yds_pc = exp_window(data_yds_pc, alpha=0.8, default=11.4, weights=wt)
    # data_ptd_pg = df['passing_td'] / df['games_played']
    # pred_ptd_pg = exp_window(data_ptd_pg, alpha=0.6, default=1.16, weights=wt)
    data_ptd_pc = df['passing_td'] / df['passing_cmp']
    pred_ptd_pc = mean(data_ptd_pc, default=0.0630, weights=wt)
    data_int_pct = df['passing_int'] / df['passing_att']
    pred_int_pct = exp_window(data_int_pct, alpha=0.75, default=0.0335, weights=wt)
    data_rushatt_pg = df['rushing_att'] / df['games_played']
    pred_rushatt_pg = exp_window(data_rushatt_pg, alpha=0.5, default=2.28, weights=wt)
    data_rushyds_pa = (df['rushing_yds'] / df['rushing_att']).fillna(0)
    pred_rushyds_pa = exp_window(data_rushyds_pa, alpha=0.75, default=3.22, weights=wt)
    data_rushtd_pa = (df['rushing_td'] / df['rushing_att']).fillna(0)
    pred_rushtd_pa = mean(data_rushtd_pa, default=0.0297, weights=wt)
    
    predictions['passing_att'] = pred_pa_pg * predictions['games_played']
    predictions['passing_cmp'] = pred_cmp_pct * predictions['passing_att']
    predictions['passing_yds'] = pred_yds_pc * predictions['passing_cmp']
    # predictions['passing_td'] = pred_ptd_pg * predictions['games_played']
    predictions['passing_td'] = pred_ptd_pc * predictions['passing_cmp']
    predictions['passing_int'] = pred_int_pct * predictions['passing_att']
    predictions['rushing_att'] = pred_rushatt_pg * predictions['games_played']
    predictions['rushing_yds'] = pred_rushyds_pa * predictions['rushing_att']
    predictions['rushing_td'] = pred_rushtd_pa * predictions['rushing_att']
    
    return predictions

# predictions using exponential averages w/ parameters picked by observation
def dumb_rb_predictions(df):
    '''
    input dataframe
    '''
    # make sure data is sorted sequentially by year
    df = df.sort_values('year')
    predictions = {}
    # assume by default that all 16 games are played
    # only 15 games are relevant for fantasy though
    predictions['games_played'] = 15
    # RB MSE is small around 50-50 (for phys rules)
    # predictions['games_played'] = 15.0/16.0*(0.5*16 + 0.5*mean(df['games_played']))
    # with no bayesian updating, we actually do better than the above
    # bayes_damp = 0.00 # updating actually does worse.
    # version for rookies
    # predictions['games_played'] = 15.0*(2.08 + bayes_damp*sum(df['games_played']))/(2.08 + 0.481 + bayes_damp*df.shape[0]*16.0)
    # version for all years. this works better w/out updating.
    # predictions['games_played'] = 15.0*(1.96 + bayes_damp*sum(df['games_played']))/(1.96 + 0.379 + bayes_damp*df.shape[0]*16.0)
    wt = df['games_played'] # weight average by games played per year
    data_rushatt_pg = df['rushing_att'] / df['games_played']
    pred_rushatt_pg = exp_window(data_rushatt_pg, alpha=0.4, default=9.65, weights=wt)
    data_rushyds_pa = df['rushing_yds'] / df['rushing_att']
    pred_rushyds_pa = exp_window(data_rushyds_pa, alpha=0.75, default=3.95, weights=wt)
    data_rushtd_pa = df['rushing_td'] / df['rushing_att']
    pred_rushtd_pa = mean(data_rushtd_pa, default=0.0282, weights=wt)
    data_rectgt_pg = df['receiving_tgt'] / df['games_played']
    pred_rectgt_pg = exp_window(data_rectgt_pg, alpha=0.5, default=2.45, weights=wt)
    data_rec_pt = (df['receiving_rec'] / df['receiving_tgt']).fillna(0)
    pred_rec_pt = exp_window(data_rec_pt, alpha=0.8, default=0.714, weights=wt)
    data_recyds_pc = (df['receiving_yds'] / df['receiving_rec']).fillna(0)
    pred_recyds_pc = exp_window(data_recyds_pc, alpha=0.8, default=7.71, weights=wt)
    data_rectd_pc = (df['receiving_td'] / df['receiving_rec']).fillna(0)
    pred_rectd_pc = exp_window(data_rectd_pc, alpha=0.75, default=0.0290, weights=wt)
    
    predictions['rushing_att'] = pred_rushatt_pg * predictions['games_played']
    predictions['rushing_yds'] = pred_rushyds_pa * predictions['rushing_att']
    predictions['rushing_td'] = pred_rushtd_pa * predictions['rushing_att']
    predictions['receiving_tgt'] = pred_rectgt_pg * predictions['games_played']
    predictions['receiving_rec'] = pred_rec_pt * predictions['receiving_tgt']
    predictions['receiving_yds'] = pred_recyds_pc * predictions['receiving_rec']
    predictions['receiving_td'] = pred_rectd_pc * predictions['receiving_rec']
    
    return predictions

# predictions using exponential averages w/ parameters picked by observation
def dumb_wr_predictions(df):
    '''
    input dataframe
    '''
    # make sure data is sorted sequentially by year
    df = df.sort_values('year')
    predictions = {}
    # assume by default that all 16 games are played
    # only 15 games are relevant for fantasy though
    # predictions['games_played'] = 15
    predictions['games_played'] = 15.0/16.0*(0.25*16 + 0.75*mean(df['games_played']))
    wt = df['games_played'] # weight average by games played per year
    # data_rushatt_pg = df['rushing_att'] / df['games_played']
    # pred_rushatt_pg = exp_window(data_rushatt_pg, alpha=0.75, default=0.162, weights=wt)
    # data_rushyds_pa = df['rushing_yds'] / df['rushing_att']
    # pred_rushyds_pa = exp_window(data_rushyds_pa, weights=wt) # currently have an infinity
    # data_rushtd_pa = df['rushing_td'] / df['rushing_att']
    # pred_rushtd_pa = mean(data_rushtd_pa, alpha=0.5, default=0.00961, weights=wt)
    data_rectgt_pg = df['receiving_tgt'] / df['games_played']
    pred_rectgt_pg = exp_window(data_rectgt_pg, alpha=0.4, default=5.46, weights=wt)
    data_rec_pt = (df['receiving_rec'] / df['receiving_tgt']).fillna(0)
    pred_rec_pt = exp_window(data_rec_pt, alpha=0.75, default=0.565, weights=wt)
    data_recyds_pc = (df['receiving_yds'] / df['receiving_rec']).fillna(0)
    pred_recyds_pc = exp_window(data_recyds_pc, alpha=0.75, default=13.4, weights=wt)
    data_rectd_pc = (df['receiving_td'] / df['receiving_rec']).fillna(0)
    pred_rectd_pc = exp_window(data_rectd_pc, alpha=0.8, default=0.0778, weights=wt)
    
    # predictions['rushing_att'] = pred_rushatt_pg * predictions['games_played']
    # predictions['rushing_yds'] = pred_rushyds_pa * predictions['rushing_att']
    # predictions['rushing_td'] = pred_rushtd_pa * predictions['rushing_att']
    predictions['receiving_tgt'] = pred_rectgt_pg * predictions['games_played']
    predictions['receiving_rec'] = pred_rec_pt * predictions['receiving_tgt']
    predictions['receiving_yds'] = pred_recyds_pc * predictions['receiving_rec']
    predictions['receiving_td'] = pred_rectd_pc * predictions['receiving_rec']
    
    return predictions

# predictions using exponential averages w/ parameters picked by observation
def dumb_te_predictions(df):
    '''
    input dataframe
    '''
    # make sure data is sorted sequentially by year
    df = df.sort_values('year')
    predictions = {}
    # assume by default that all 16 games are played
    # only 15 games are relevant for fantasy though
    # predictions['games_played'] = 15
    predictions['games_played'] = 15.0/16.0*(0.25*16 + 0.75*mean(df['games_played']))
    wt = df['games_played'] # weight average by games played per year
    data_rectgt_pg = df['receiving_tgt'] / df['games_played']
    pred_rectgt_pg = exp_window(data_rectgt_pg, alpha=0.4, default=3.35, weights=wt)
    data_rec_pt = (df['receiving_rec'] / df['receiving_tgt']).fillna(0)
    pred_rec_pt = exp_window(data_rec_pt, alpha=0.8, default=0.642, weights=wt) # replace by mean?
    data_recyds_pc = (df['receiving_yds'] / df['receiving_rec']).fillna(0)
    pred_recyds_pc = exp_window(data_recyds_pc, alpha=0.75, default=10.6, weights=wt) # consider using alpha=0.75 for all positions
    data_rectd_pc = (df['receiving_td'] / df['receiving_rec']).fillna(0)
    pred_rectd_pc = mean(data_rectd_pc, default=0.0926, weights=wt)
    
    predictions['receiving_tgt'] = pred_rectgt_pg * predictions['games_played']
    predictions['receiving_rec'] = pred_rec_pt * predictions['receiving_tgt']
    predictions['receiving_yds'] = pred_recyds_pc * predictions['receiving_rec']
    predictions['receiving_td'] = pred_rectd_pc * predictions['receiving_rec']
    
    return predictions

def dumb_pos_predictions(pos, df):
    if pos.lower() == 'qb':
        return dumb_qb_predictions(df)
    if pos.lower() == 'rb':
        return dumb_rb_predictions(df)
    if pos.lower() == 'wr':
        return dumb_wr_predictions(df)
    if pos.lower() == 'te':
        return dumb_te_predictions(df)
