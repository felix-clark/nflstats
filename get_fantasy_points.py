
# import ruleset
# import pandas as pd

# doesn't cover every rule, just the main ones.
# some stats are not as easily extracted
def getBasePoints( rs, plyr ):
    # missing bonus 40 and 50 yd tds here. other method getBonusPoints, which looks at weekly stats??
    # also missing extra points for longer field goals
    # note: for some reason a line like
    # + some_val
    # is valid python, but will not get added
    fpts = \
    + rs.ppPY * plyr.passing_yds \
    + rs.ppPY25 * (plyr.passing_yds / 25) \
    + rs.ppPC * plyr.passing_cmp \
    + rs.ppINC * (plyr.passing_att - plyr.passing_cmp) \
    + rs.ppPTD * plyr.passing_tds \
    + rs.ppINT * plyr.passing_int \
    + rs.pp2PC * plyr.passing_twoptm \
    + rs.ppRY * plyr.rushing_yds \
    + rs.ppRY10 * (plyr.rushing_yds / 10) \
    + rs.ppRTD * plyr.rushing_tds \
    + rs.pp2PR * plyr.rushing_twoptm \
    + rs.ppREY * plyr.receiving_yds \
    + rs.ppREY10 * (plyr.receiving_yds / 10) \
    + rs.ppREC * plyr.receiving_rec \
    + rs.ppRETD * plyr.receiving_tds \
    + rs.pp2PRE * plyr.receiving_twoptm \
    + rs.ppFUML * plyr.fumbles_lost \
    + rs.ppPAT * plyr.kicking_xpmade \
    + rs.ppFGM * (plyr.kicking_fga - plyr.kicking_fgm) \
    + rs.ppFG0 * plyr.kicking_fgm
    return fpts

# def decorateDataFrame( rs, df, name='fantasy_points' ):
#     """
#     repeating ourselves a lot here; it would be good to combine code w/ above (or only use this)
#     rs: rule set
#     df: dataframe containing player stats
#     name: name of decorated stat
#     """
#     df[name] = \
#     + rs.ppPY * df['passing_yds'] \
#     + rs.ppPY25 * (df['passing_yds'] / 25) \
#     + rs.ppPC * df['passing_cmp'] \
#     + rs.ppINC * (df['passing_att'] - df['passing_cmp']) \
#     + rs.ppPTD * df['passing_tds'] \
#     + rs.ppINT * df['passing_int'] \
#     + rs.pp2PC * df['passing_twoptm'] \
#     + rs.ppRY * df['rushing_yds'] \
#     + rs.ppRY10 * (df['rushing_yds'] / 10) \
#     + rs.ppRTD * df['rushing_tds'] \
#     + rs.pp2PR * df['rushing_twoptm'] \
#     + rs.ppREY * df['receiving_yds'] \
#     + rs.ppREY10 * (df['receiving_yds'] / 10) \
#     + rs.ppREC * df['receiving_rec'] \
#     + rs.ppRETD * df['receiving_tds'] \
#     + rs.pp2PRE * df['receiving_twoptm'] \
#     + rs.ppFUML * df['fumbles_lost'] \
#     + rs.ppPAT * df['kicking_xpmade'] \
#     + rs.ppFGM * (df['kicking_fga'] - df['kicking_fgm']) \
#     + rs.ppFG0 * df['kicking_fgm']
#     return df

def get_points_from_data_frame( rs, df ):
    """
    repeating ourselves a lot here; it would be good to combine code w/ above (or only use this)
    rs: rule set
    df: dataframe containing player stats
    name: name of decorated stat
    """
    return \
    rs.ppPY * df.get('pass_yds', default=0) \
    + rs.ppPY25 * (df.get('pass_yds', default=0) // 25) \
    + rs.ppPC * df.get('pass_cmp', default=0) \
    + rs.ppINC * (df.get('pass_att', default=0) - df.get('pass_cmp', default=0)) \
    + rs.ppPTD * df.get('pass_td', default=0) \
    + rs.ppINT * df.get('pass_int', default=0) \
    + rs.pp2PC * df.get('pass_twoptm', default=0) \
    + rs.ppRY * df.get('rush_yds', default=0) \
    + rs.ppRY10 * (df.get('rush_yds', default=0) // 10) \
    + rs.ppRTD * df.get('rush_tds', default=0) \
    + rs.pp2PR * df.get('rush_twoptm', default=0) \
    + rs.ppREY * df.get('rec_yds', default=0) \
    + rs.ppREY10 * (df.get('rec_yds', default=0) // 10) \
    + rs.ppREC * df.get('rec', default=0) \
    + rs.ppRETD * df.get('rec_tds', default=0) \
    + rs.pp2PRE * df.get('rec_twoptm', default=0) \
    + rs.ppFUML * df.get('fumbles_lost', default=0) \
    + rs.ppPAT * df.get('xpm', default=0) \
    + rs.ppFGM * (df.get('fga', default=0) - df.get('fgm', default=0)) \
    + rs.ppFG0 * df.get('fgm', default=0)
    # return df
# could manually suffix these w/ "kick_". note that we can probably use more stats now (with PFR).
# TODO: missing: missed PATs (ppPATM)
