#!/usr/bin/python
import numpy as np
import pandas as pd
from scipy import stats
import os.path
import itertools
import random

VERBOSE = False

random.seed() # no argument (or None) seeds from urandom if available

n_teams = 14
# n_roster_per_team = {'QB':1,'RB':2,'WR':2,'TE':1,'K':1,'FLEX':1}
n_roster_per_team = {'QB':1,'RB':2,'WR':2,'TE':1,'K':1}
flex_pos = ['RB', 'WR', 'TE']
n_draft_rounds = sum( (n for (_,n) in n_roster_per_team.items()) )
n_starters = {}
for pos,nper in n_roster_per_team.items():
    n_starters[pos] = nper * n_teams

def hasExtraFlexPicks( team_roster ):
    """
    this function returns true if the team has filled up their starting lineup for FLEX-type positions, but still has extra picks for the FLEX spot(s).
    """
    n_flex_players = sum( (len(team_roster[pos]) for pos in flex_pos) )
    n_flex_positions = sum( (n_roster_per_team[pos] for pos in flex_pos ) )
    n_flex_players_over_base = n_flex_players - n_flex_positions
    return n_flex_players_over_base >= 0 and n_flex_players_over_base < n_roster_per_team['FLEX']

def picksTilNextTurn( n_teams, pick_in_first_round, n_this_round ):
    """
    given number of teams, the assigned 1st round pick of this team, and the round, compute how many picks there are to go until the next pick for this team in a snake draft.
    pick_in_first_round: starts counting at zero
    """
    if n_this_round % 2 == 0:
        #  first / third / fifth round (start at zero)
        return 2*(n_teams - pick_in_first_round) - 1
    else:
        # second / fourth / sixth ... round
        return 2*pick_in_first_round + 1

def stratPickMaxVal( pos_values_dict, team_roster, **kwargs ):
    maxval = -1000
    bestpos = ''
    for pos,vals in pos_values_dict.items():
        posval = vals[0] # the top value in this position
        if len(team_roster[pos]) < n_roster_per_team[pos] and posval > maxval:
            bestpos,maxval = pos,posval
    # check flex option
    if hasExtraFlexPicks( team_roster ):
        # we can still pick at least one additional flex player
        for pos in flex_pos:
            posval = pos_values_dict[pos][0]
            if posval > maxval:
                bestpos,maxval = pos,posval
    return bestpos

def stratPickMaxValOverWorstStarter( pos_values_dict, team_roster, **kwargs ):
    maxval = -1000
    bestpos = ''
    worst_starters_dict = kwargs['worst_starters']
    for pos,pos_vals in pos_values_dict.items():
        val = pos_vals[0] - worst_starters_dict[pos]
        if len(team_roster[pos]) < n_roster_per_team[pos] and val > maxval:
            bestpos,maxval = pos,val
    if hasExtraFlexPicks( team_roster ):
        worst_flex = worst_starters_dict['FLEX']
        for pos in flex_pos:
            flexval = pos_values_dict[pos][0] - worst_flex
            if flexval > maxval:
                bestpos,maxval = pos,flexval
    return bestpos

## this one sucks, probably jsut remove it
# def stratPickMaxValOverMeanStarter( pos_values_dict, pos_picked_league_dict, team_roster, n_picks_til_next_turn=None ):
#     maxval = -1000
#     bestpos = ''
#     for pos,pos_vals in pos_values_dict.items():
#         npicked = pos_picked_league_dict[pos]
#         val = pos_vals[npicked] - np.mean(pos_vals[0:n_starters[pos]+1])
#         if len(team_roster[pos]) < n_roster_per_team[pos] and val > maxval:
#             bestpos,maxval = pos,val
#     return bestpos

def stratPickMaxValOverMeanRemainingStarter( pos_values_dict, team_roster, **kwargs ):
    maxval = -1000
    bestpos = ''
    pos_picked_league_dict = kwargs['pos_picked_league']
    for pos,pos_vals in pos_values_dict.items():
        if len(team_roster[pos]) >= n_roster_per_team[pos]: continue;
        npicked = pos_picked_league_dict[pos]
        nstarters = n_starters[pos]
        mean_remaining = np.mean(pos_vals[:nstarters-npicked]) if nstarters > npicked else 0
        posval = pos_vals[0] - mean_remaining
        if posval > maxval:
            bestpos,maxval = pos,posval
    if hasExtraFlexPicks( team_roster ):
        flex_list = np.sort( list( itertools.chain.from_iterable( (pos_values_dict[pos] for pos in flex_pos) ) ) )
        n_flex_picked = sum( (pos_picked_league_dict[pos] for pos in flex_pos) )
        n_flex_starters = sum( (n_starters[pos] for pos in flex_pos) ) + n_starters['FLEX']
        mean_remaining_flex = np.mean( flex_list[:n_flex_starters-n_flex_picked] )
        for pos in flex_pos:
            posval = pos_values_dict[pos][0] - mean_remaining_flex
            if posval > maxval:
                bestpos,maxval = pos,posval
    return bestpos

# TODO: rename this something that indicates it doesn't just average the mean remaining and the worst starter anymore
def stratPickMaxValOverMeanRemainingAndWorstStarter( pos_values_dict, team_roster, **kwargs ):
    maxval = -1000
    bestpos = ''
    pos_picked_league_dict = kwargs['pos_picked_league']
    for pos,pos_vals in pos_values_dict.items():
        if len(team_roster[pos]) >= n_roster_per_team[pos]: continue
        npicked = pos_picked_league_dict[pos]
        best_in_pos = pos_vals[0]
        num_avg = n_starters[pos]-npicked
        ## these strats are inferior until they deal with flex starters properly
        # if num_avg <= 1: ## this can happen if someone has already taken a flex and needs to be dealt with
        #     print pos, n_starters[pos], npicked
        weights = np.exp( np.arange(num_avg) * np.log(num_avg)/(num_avg-1.0) )/num_avg # exponential does just as fine, possibly better than linear
        # weights = [(1.0+i)/num_avg for i in range(num_avg)] # linear weights seems to do comparable to only comparing to worst. # TODO: try different slopes? most seem worse
        sum_weights = sum(weights)
        weighted_vals = (w*(best_in_pos - pv) for w,pv in zip(weights,pos_vals[:num_avg]))
        val = sum(weighted_vals)/sum_weights if sum_weights > 0 else 0
        if val > maxval:
            bestpos,maxval = pos,val
    if hasExtraFlexPicks( team_roster ):
        n_flex_starters = sum( (n_starters[pos] for pos in flex_pos) ) + n_starters['FLEX']
        n_flex_picked = sum( (pos_picked_league_dict[pos] for pos in flex_pos) )
        flex_list = np.sort( list( itertools.chain.from_iterable( (pos_values_dict[pos] for pos in flex_pos) ) ) )
        num_avg = n_flex_starters - n_flex_picked
        weights = np.exp( np.arange(num_avg) * np.log(num_avg)/(num_avg-1.0) )/num_avg # exponential does just as fine, possibly better than linear
        sum_weights = sum( weights )
        weighted_vals = (w*(flex_list[0] - pv) for w,pv in zip(weights,flex_list[:num_avg]))
        flexval = sum( weighted_vals)/sum_weights if sum_weights > 0 else 0
        if flexval > maxval:
            maxflex = -1000 # we need to pick the maximum value flex player
            for pos in flex_pos:
                posval = pos_values_dict[pos]
                if posval > maxflex:
                    bestpos,maxflex = pos,posval
    return bestpos

def stratPickMaxValOverMeanRemainingAndWorstStarterBasic( pos_values_dict, team_roster, **kwargs ):
    """
    this one does pretty good just by averaging value over worse and value over mean remaining
    begs question of whether a more sophisticated weighting function would do even better
    """
    maxval = -1000
    bestpos = ''
    worst_weight = 0.5 # this one is better, so we should rate it higher. weight of 0.5 usually still does better.
    mean_weight = 1-worst_weight
    worst_starter_dict = kwargs['worst_starters']
    pos_picked_league_dict = kwargs['pos_picked_league']
    for pos,pos_vals in pos_values_dict.items():
        npicked = pos_picked_league_dict[pos]
        nstarters = n_starters[pos]
        best_in_pos = pos_vals[0]
        mean_remaining = np.mean(pos_vals[:nstarters-npicked]) if nstarters > npicked else 0
        mean_val = best_in_pos - mean_remaining
        worst_val = best_in_pos - worst_starter_dict[pos]
        val = mean_weight*mean_val + worst_weight*worst_val
        if len(team_roster[pos]) < n_roster_per_team[pos] and val > maxval:
            bestpos,maxval = pos,val
    if hasExtraFlexPicks( team_roster ):
        n_flex_starters = sum( (n_starters[pos] for pos in flex_pos) ) + n_starters['FLEX']
        n_flex_picked = sum( (pos_picked_league_dict[pos] for pos in flex_pos) )
        flex_list = np.sort( list( itertools.chain.from_iterable( (pos_values_dict[pos] for pos in flex_pos) ) ) )
        mean_remaining = np.mean( flex_list[:n_flex_starters-n_flex_picked] ) if n_flex_starters > n_flex_picked else 0 # TODO: why is this protection necessary? indexing error? -- because some strats pick flex before this one picks a starter. need new definition of "starter" to include FLEX players, perhaps per-round.
        best_flex = flex_list[0]
        mean_val = best_flex - mean_remaining
        worst_val = best_flex - worst_starter_dict['FLEX']
        flexval = mean_weight*mean_val + worst_weight*worst_val
        if flexval > maxval:
            maxflex = -1000 # we need to pick the maximum value flex player
            for pos in flex_pos:
                posval = pos_values_dict[pos]
                if posval > maxflex:
                    bestpos,maxflex = pos,posval
    return bestpos


# def pointsOverWorstNextTurn( pos_values_dict, picks_til_next_turn ):
#     """
#     returns a dict for each position where the value is the value the best player has above the worst possible player available in this position in the next round for this team
#     """
#     result_dict = {}
#     for pos,vals in pos_values_dict.items():
#         # n_already_picked = pos_picked_dict[pos]
#         # print pos
#         pointsWorst = vals[picks_til_next_turn] if len(vals) > picks_til_next_turn else 0
#         pointsOverWorst = vals[0] - pointsWorst
#         result_dict[pos] = pointsOverWorst
#     return result_dict

def stratPickMaxValOverWorstNextTurn( pos_values_dict, team_roster, **kwargs ):
    n_picks_til_next_turn = kwargs['n_picks_til_next_turn']
    worst_starter_dict = kwargs['worst_starters']
    maxval = -1000
    bestpick = ''
    for pos,posvals in pos_values_dict.items():
        if len(team_roster[pos]) >= n_roster_per_team[pos]: continue
        posval_worst_case = posvals[ n_picks_til_next_turn ] if len(posvals) > n_picks_til_next_turn else 0
        ntval = posvals[0] - posval_worst_case
        wsval = posvals[0] - worst_starter_dict[pos]
        modval = min(ntval,wsval)
        if modval > maxval:
            bestpick,maxval = pos,modval
    if hasExtraFlexPicks( team_roster ):
        worst_flex_starter = worst_starter_dict['FLEX']
        flex_list = np.sort( list( itertools.chain.from_iterable( (pos_values_dict[pos] for pos in flex_pos) ) ) )
        best_flex_worst_case = flex_list[ n_picks_til_next_turn ]
        flex_baseline = max(worst_flex_starter, best_flex_worst_case)
        for pos in flex_pos:
            posval = pos_values_dict[pos][0] - flex_baseline
            if posval > maxval:
                bestpick,maxval = pos,posval
    return bestpick

# def stratPickMaxValOverWorstStarterAndWorstNextTurn( pos_values_dict, pos_picked_league_dict, team_roster, n_picks_til_next_turn ):
#     pointsOverWorstStarterDict = pointsOverWorstStarter( pos_values_dict, pos_picked_league_dict )
#     pointsOverWorstNextTurnDict = pointsOverWorstNextTurn( pos_values_dict, pos_picked_league_dict, n_picks_til_next_turn )
#     maxval = -1000
#     bestpick = ''
#     stweight,ntweight = 0.9,0.05 # possibly better; possibly no statistical significance
#     for pos,ntval in pointsOverWorstNextTurnDict.items():
#         stval = pointsOverWorstStarterDict[pos]
#          # combo of two best easy strats
#         # modval = stweight*stval/(n_roster_per_team[pos]) + (1-stweight)*min(ntval,stval)*1.0
#         modval = stweight*stval + ntweight*min(ntval,stval)
#         if len(team_roster[pos]) < n_roster_per_team[pos] and modval > maxval:
#             bestpick,maxval = pos,modval
#     return bestpick


# def stratPickMaxValOverNext( pos_values_dict, team_roster, **kwargs ):
#     """
#     picks the player with the largest dropoff. is not a good strategy.
#     """
#     maxval = -1000
#     bestpick = ''
#     for pos,posvals in pos_values_dict.items():
#         modval = posvals[0] - posvals[1]
#         if len(team_roster[pos]) < n_roster_per_team[pos] and modval > maxval:
#             bestpick,maxval = pos,modval
#     return bestpick

def stratPickRandom( pos_values_dict, team_roster, **kwargs ):
    """
    will pick any position (weighted by remaining slots)
    intended to be a stupid strategy for control
    """
    poss_pos = []
    for pos,players in team_roster.items():
        n_left = n_roster_per_team[pos] - len(players)
        poss_pos += [pos for _ in range(n_left)]
    if hasExtraFlexPicks( team_roster ):
        poss_pos.extend( flex_pos )
    pos_choice = random.choice(poss_pos)
    return pos_choice

def stratPickRandomSkill( pos_values_dict, team_roster, **kwargs ):
    """
    will pick skill players before kickers or defense, in order to follow common sense
    otherwise dumbly random
    """
    poss_pos = []
    for pos,players in team_roster.items():
        n_left = n_roster_per_team[pos] - len(players)
        poss_pos += [pos for _ in range(n_left)]
    if hasExtraFlexPicks( team_roster ):
        poss_pos.extend( flex_pos )
    filtered_poss_pos = [pos for pos in poss_pos if pos not in ['K', 'DST']]
    if len(filtered_poss_pos) > 0:
        poss_pos = filtered_poss_pos
    pos_choice = random.choice(poss_pos)
    pick = pos_choice
    return pick


stratlist = [
    # stratPickRandom, # this one is bad enough that we shouldn't include it cuz people should be smarter
    # stratPickRandomSkill, # consistently worse than the following; should leave it out so those get more trials ( can put it back in for a baseline comparison )
    # stratPickMaxVal, # this is worse than both of the following
    stratPickMaxValOverWorstStarter, # this is the best of the simple strats
    # stratPickMaxValOverWorstNextTurn, # this is the next-best after mean remaining, but definitely inferior to the better ones
    # stratPickMaxValOverWorstStarterAndWorstNextTurn, # combo of the best two strats. ignoring for now because it's somewhat arbitrary
    # stratPickMaxValOverMeanStarter, # little better than random
    # stratPickMaxValOverMeanRemainingStarter, # 2nd best simple option, to value over worst starter. clearly inferior though
    stratPickMaxValOverMeanRemainingAndWorstStarter, # better than either simple part on their own; not by a lot though (few points per season)
    stratPickMaxValOverMeanRemainingAndWorstStarterBasic # less sophisticated average (mean + worst) than previous option. they perform too similar to compare by eye (within stat errors)
    # stratPickMaxValOverWorstEndOfRound, # this one sucks
    # stratPickMaxValOverNext # and this one sucks worse
]
def getRandomStrats( n_teams ):
    randfuncs = [stratlist[random.randint(0,len(stratlist)-1)] for _ in range(n_teams)]
    return randfuncs

def getSumValue( team_roster ):
    tot = 0
    for _,vals in team_roster.items():
        tot += sum(vals)
    return tot

datafilename = 'season_points.csv'
if not os.path.isfile( datafilename ):
    print datafilename, ' does not exist. generate it with another script (season_points.py?)'
    exit(1)

df = pd.DataFrame.from_csv( datafilename )
# print df

n_players = {'QB':32,'RB':64,'WR':72,'TE':48,'K':48}
# a dict of lists, where each list is a randomized estimated set of values for the position,
#  taken from historical data
position_values = {}

strat_totals = [[] for _ in stratlist]
draft_position_totals = [[] for _ in range(n_teams)]

n_trials = 100

for i_trial in range(n_trials):
    if VERBOSE: print '\n**************** trial {} ****************'.format( i_trial )
    worst_starters_dict = {}
    for pos in n_roster_per_team.keys():
        if pos == 'FLEX': continue # deal with FLEX separately
        point_data = df[df.position==pos]['fantasy_points']
        value_dist = stats.gaussian_kde( point_data )
        unnormed_vals = list(itertools.chain.from_iterable(np.sort(value_dist.resample(n_players[pos]))))[::-1]
        worst_starters_dict[pos] = unnormed_vals[ n_starters[pos]-1 ]
        position_values[pos] = [int( round( x ) ) for x in unnormed_vals]
        if VERBOSE: print pos, ': \n', position_values[pos]
    n_total_flex_players = sum( (n_starters[pos] for pos in flex_pos) ) + n_starters['FLEX']
    flex_list = np.sort(list(itertools.chain.from_iterable( ( position_values[pos] for pos in flex_pos ) ) ) )[::-1]
    # print flex_list
    worst_starters_dict['FLEX'] = flex_list[n_total_flex_players-1] # should be worst than the worst starter of each position independently
    ## TODO: need to adjust worst starters by-position, including FLEX.
    # changing global n_starters is not pretty, but maybe pass in as keyword arg
    # could possibly do it by round.
    n_starters_up_to_round = {}
        
    team_rosters = [{'QB':[],'RB':[],'WR':[],'TE':[],'K':[]} for _ in range(n_teams)]
    team_strats = getRandomStrats( n_teams )
    # keep track of the number of each position drafted by the league
    pos_picked_league_dict = {'QB':0,'RB':0,'WR':0,'TE':0,'K':0}

    for i_round in range(n_draft_rounds):
        if VERBOSE: print 'round {}:'.format( i_round )
        rnteams = range(n_teams)
        rev_round = i_round % 2 == 1 # every other round in a snake draft is in reverse order
        team_is = rnteams[::-1] if rev_round else rnteams
        for i_team in team_is:
            team_roster = team_rosters[i_team]
            team_strat = team_strats[i_team]
            n_picks_til_next_turn = picksTilNextTurn( n_teams, i_team, i_round )
            newpos = team_strat( position_values, team_roster, worst_starters=worst_starters_dict, pos_picked_league=pos_picked_league_dict, n_picks_til_next_turn=n_picks_til_next_turn )
            # value = position_values[newpos][posrank]
            # print 'offending function is ', team_strat.__name__
            value = position_values[newpos].pop(0) # now take the top one and remove it from the list
            posrank = pos_picked_league_dict[newpos]+1
            if VERBOSE: print 'team {}\'s pick ({}): {}{} ({})'.format(i_team,team_strat.__name__[9:],newpos,posrank,value)
            pos_picked_league_dict[newpos] = pos_picked_league_dict[newpos] + 1 # increment the count of each position picked
            team_roster[newpos].append( value )

    # print team_rosters
    # print team_strats
    for i_team in range(n_teams):
        strat = team_strats[i_team]
        rost = team_rosters[i_team]
        roster_value = getSumValue( rost )
        draft_position_totals[i_team].append( roster_value )
        for check_st,sttot in zip(stratlist,strat_totals):
            if strat is check_st:
                sttot.append( roster_value )

for i_st,st in enumerate(strat_totals):
    st_name = stratlist[i_st].__name__
    print '{}: {}'.format(st_name, np.mean(st))
for i_tm,dpt in enumerate(draft_position_totals):
    print 'position {}: {}'.format( i_tm, np.mean(dpt) )
