
def naive(data, default=0):
    if data.size == 0: return default
    return data.iloc[data.last_valid_index()]
