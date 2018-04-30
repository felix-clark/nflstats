import numpy as np

def naive(data, default=0):
    if data.size == 0: return default
    return data.iloc[-1]

def mean(data, default=0):
    if data.size == 0: return default
    return data.mean()

def exp_window(data, alpha=0.5, default=0):
    if (alpha == 0): return naive(data, default)
    if (alpha == 1): return mean(data, default)
    T = data.size
    if T == 0: return default
    weights = np.array([alpha**i for i in range(T)])
    norm = (1.0-alpha)/(1.0-alpha**T)
    return norm * sum(weights*data[::-1])

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
    predictions['games_played'] = 15
    data_pa_pg = df['passing_att'] / df['games_played']
    # these defaults are for all QBs, not rookies
    pred_pa_pg = exp_window(data_pa_pg, alpha=0.5, default=28.5)
    data_cmp_pct = df['passing_cmp'] / df['passing_att']
    pred_cmp_pct = exp_window(data_cmp_pct, alpha=0.5, default=0.581)
    data_ptd_pg = df['passing_td'] / df['games_played']
    pred_ptd_pg = exp_window(data_ptd_pg, alpha=0.6, default=1.16)
    data_int_pct = df['passing_int'] / df['passing_att']
    pred_int_pct = exp_window(data_int_pct, alpha=0.75, default=0.0336)
    
    predictions['passing_att'] = pred_pa_pg * predictions['games_played']
    predictions['passing_cmp'] = pred_cmp_pct * predictions['passing_att']
    predictions['passing_td'] = pred_ptd_pg * predictions['games_played']
    predictions['passing_int'] = pred_int_pct * predictions['passing_att']

    return predictions
