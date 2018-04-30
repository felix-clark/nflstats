import numpy as np

def naive(data, default=0):
    if data.size == 0: return default
    # print(data.last_valid_index())
    return data.iloc[-1]
    # return data.iloc[data.last_valid_index()]

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
    # for sl in range(1,len(data)-1):
    #     print('sl = {}'.format(sl))
    #     print(data.iloc[:sl])
    #     print(model(data.iloc[:sl]))
    #     print(data.iloc[sl])

