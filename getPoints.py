
# import ruleset

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
    + rs.ppINT * plyr.passing_ints \
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
