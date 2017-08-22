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

def stratPickMaxVal( pos_values_dict, pos_picked_league_dict, team_roster, n_picks_til_next_turn=None ):
    maxval = -1000
    bestpos = ''
    for pos,vals in pos_values_dict.items():
        index = pos_picked_league_dict[pos]
        posval = vals[index]
        if len(team_roster[pos]) < n_roster_per_team[pos] and posval > maxval:
            bestpos,maxval = pos,posval
    # check flex option
    nflex = sum( (team_roster[pos] for pos in flex_pos) )
    baseflexallowed = sum( (n_roster_per_team[pos] for pos in flex_pos) )
    if nflex >= baseflexallowed and nflex < baseflexallowed + n_roster_per_team['FLEX']:
        # we can pick an additional flex player
        for pos in flex_pos:
            index = pos_picked_league_dict[pos]
            posval = pos_values_dict[pos][index]
            if posval > maxval:
                bestpos,maxval = pos,posval
    return bestpos
    
def pointsOverWorstStarter( pos_values_dict, pos_picked_league_dict, picks_til_next_turn=None ):
    """
    returns a dict for each position where the value is the value the best player has above the worst starter in that position.
    # the data may already be normalized for this, but if so it should still be safe to subtract off the zero
    """
    result_dict = {}
    for pos,vals in pos_values_dict.items():
        result_dict[pos] = vals[pos_picked_league_dict[pos]] - vals[n_starters[pos]]
    remaining_flex_list = np.sort( list(itertools.chain.from_iterable( (pos_values_dict[pos][pos_picked_league_dict[pos]:] for pos in flex_pos) ) ) )
    n_picked_flex = sum( (pos_picked_league_dict[pos] for pos in flex_pos) )
    last_flex_index = sum( (n_starters[pos] for pos in flex_pos) )
    worst_flex_starter = remaining_flex_list[last_flex_index - n_picked_flex]
    max_flex_val = -1000
    bestflex = ''
    for pos in flex_pos:
        flexposval = pos_values_dict[pos][pos_picked_league_dict[pos]]
        if flexposval > max_flex_val:
            bestflex,max_flex_val = pos,flexposval
    result_dict['FLEX'] = (bestflex, max_flex_val - worst_flex_starter)
    return result_dict

def stratPickMaxValOverWorstStarter( pos_values_dict, pos_picked_league_dict, team_roster, n_picks_til_next_turn=None ):
    # pointsOverWorstStarterDict = pointsOverWorstStarter( pos_values_dict, pos_picked_league_dict )
    maxval = -1000
    bestpos = ''
    for pos,pos_vals in pos_values_dict.items():
        npicked = pos_picked_league_dict[pos]
        val = pos_vals[npicked] - pos_vals[n_starters[pos]]
        if len(team_roster[pos]) < n_roster_per_team[pos] and val > maxval:
            bestpos,maxval = pos,val
    # TODO: flex # consider refactoring to pop off picked players from position_values_dict ?
    ## yes we should: will be necessary for dealing with sub-optimal choices
    return bestpos

# def stratPickMaxValOverMeanStarter( pos_values_dict, pos_picked_league_dict, team_roster, n_picks_til_next_turn=None ):
#     maxval = -1000
#     bestpos = ''
#     for pos,pos_vals in pos_values_dict.items():
#         npicked = pos_picked_league_dict[pos]
#         val = pos_vals[npicked] - np.mean(pos_vals[0:n_starters[pos]+1])
#         if len(team_roster[pos]) < n_roster_per_team[pos] and val > maxval:
#             bestpos,maxval = pos,val
#     return bestpos

def stratPickMaxValOverMeanRemainingStarter( pos_values_dict, pos_picked_league_dict, team_roster, n_picks_til_next_turn=None ):
    maxval = -1000
    bestpos = ''
    for pos,pos_vals in pos_values_dict.items():
        npicked = pos_picked_league_dict[pos]
        val = pos_vals[npicked] - np.mean(pos_vals[npicked:n_starters[pos]+1])
        if len(team_roster[pos]) < n_roster_per_team[pos] and val > maxval:
            bestpos,maxval = pos,val
    return bestpos

def stratPickMaxValOverMeanRemainingAndWorstStarter( pos_values_dict, pos_picked_league_dict, team_roster, n_picks_til_next_turn=None ):
    maxval = -1000
    bestpos = ''
    worst_weight = 0.5 # this one is better, so we should rate it higher. weight of 0.5 usually still does better.
    mean_weight = 1-worst_weight
    for pos,pos_vals in pos_values_dict.items():
        npicked = pos_picked_league_dict[pos]
        best_in_pos = pos_vals[npicked]
        # num_avg = n_starters[pos]+1-npicked
        # weights = [float(i) for i in range(num_avg)] # linear weights does worse than averaging mean and worst
        # sum_weights = sum(weights)
        # weighted_vals = (w*(best_in_pos - pv) for w,pv in zip(weights,pos_vals[npicked:n_starters[pos]+1]))
        # val = sum(weighted_vals)/sum_weights if sum_weights > 0 else 0
        mean_val = best_in_pos - np.mean(pos_vals[npicked:n_starters[pos]+1])
        worst_val = best_in_pos - pos_vals[n_starters[pos]]
        val = mean_weight*mean_val + worst_weight*worst_val
        if len(team_roster[pos]) < n_roster_per_team[pos] and val > maxval:
            bestpos,maxval = pos,val
    return bestpos


def pointsOverWorstNextTurn( pos_values_dict, pos_picked_dict, picks_til_next_turn ):
    """
    returns a dict for each position where the value is the value the best player has above the worst possible player available in this position in the next round for this team
    """
    result_dict = {}
    for pos,vals in pos_values_dict.items():
        n_already_picked = pos_picked_dict[pos]
        # print pos
        pointsWorst = vals[n_already_picked+picks_til_next_turn] if len(vals) > n_already_picked+picks_til_next_turn else 0
        pointsOverWorst = vals[n_already_picked] - pointsWorst
        result_dict[pos] = pointsOverWorst
    return result_dict

def stratPickMaxValOverWorstNextTurn( pos_values_dict, pos_picked_league_dict, team_roster, n_picks_til_next_turn ):
    pointsOverWorstStarterDict = pointsOverWorstStarter( pos_values_dict, pos_picked_league_dict )
    pointsOverWorstNextTurnDict = pointsOverWorstNextTurn( pos_values_dict, pos_picked_league_dict, n_picks_til_next_turn )
    maxval = -1000
    bestpick = ''
    for pos,val in pointsOverWorstNextTurnDict.items():
        modval = min(val,pointsOverWorstStarterDict[pos])
        if len(team_roster[pos]) < n_roster_per_team[pos] and modval > maxval:
            bestpick,maxval = pos,modval
    return bestpick

def stratPickMaxValOverWorstStarterAndWorstNextTurn( pos_values_dict, pos_picked_league_dict, team_roster, n_picks_til_next_turn ):
    pointsOverWorstStarterDict = pointsOverWorstStarter( pos_values_dict, pos_picked_league_dict )
    pointsOverWorstNextTurnDict = pointsOverWorstNextTurn( pos_values_dict, pos_picked_league_dict, n_picks_til_next_turn )
    maxval = -1000
    bestpick = ''
    stweight,ntweight = 0.9,0.05 # possibly better; possibly no statistical significance
    for pos,ntval in pointsOverWorstNextTurnDict.items():
        stval = pointsOverWorstStarterDict[pos]
         # combo of two best easy strats
        # modval = stweight*stval/(n_roster_per_team[pos]) + (1-stweight)*min(ntval,stval)*1.0
        modval = stweight*stval + ntweight*min(ntval,stval)
        if len(team_roster[pos]) < n_roster_per_team[pos] and modval > maxval:
            bestpick,maxval = pos,modval
    return bestpick


def pointsOverWorstEndOfRound( pos_values_dict, pos_picked_dict, picks_til_next_turn ):
    """
    returns a dict for each position where the value is the value the best player has above the worst possible player available in this position in the next round for this team
    """
    result_dict = {}
    for pos,vals in pos_values_dict.items():
        n_already_picked = pos_picked_dict[pos]
        pointsWorst = vals[n_already_picked+(picks_til_next_turn-1)/2] if len(vals) > n_already_picked+(picks_til_next_turn-1)/2 else vals[-1]
        pointsOverWorst = vals[n_already_picked] - pointsWorst
        result_dict[pos] = pointsOverWorst
    return result_dict

def stratPickMaxValOverWorstEndOfRound( pos_values_dict, pos_picked_league_dict, team_roster, n_picks_til_next_turn ):
    pointsOverWorstStarterDict = pointsOverWorstStarter( pos_values_dict, pos_picked_league_dict )
    pointsOverWorstEndOfRoundDict = pointsOverWorstEndOfRound( pos_values_dict, pos_picked_league_dict, n_picks_til_next_turn )
    maxval = -1000
    bestpick = ''
    for pos,val in pointsOverWorstEndOfRoundDict.items():
        modval = min(val,pointsOverWorstStarterDict[pos])
        if len(team_roster[pos]) < n_roster_per_team[pos] and modval > maxval:
            bestpick,maxval = pos,val
    return bestpick

def stratPickMaxValOverNext( pos_values_dict, pos_picked_league_dict, team_roster, n_picks_til_next_turn ):
    """
    picks the player with the largest dropoff. is not a good strategy.
    """
    maxval = -1000
    bestpick = ''
    for pos,posvals in pos_values_dict.items():
        i_posval = pos_picked_league_dict[pos]
        modval = posvals[i_posval] - posvals[i_posval+1]
        if len(team_roster[pos]) < n_roster_per_team[pos] and modval > maxval:
            bestpick,maxval = pos,modval
    return bestpick

def stratPickRandom( pos_values_dict, pos_picked_league_dict, team_roster, n_picks_til_next_turn ):
    """
    will pick any position (weighted by remaining slots)
    intended to be a stupid strategy for control
    """
    poss_pos = []
    for pos,players in team_roster.items():
        n_left = n_roster_per_team[pos] - len(players)
        poss_pos += [pos for _ in range(n_left)]
    pos_choice = random.choice(poss_pos)
    pick = pos_choice
    return pick

def stratPickRandomSkill( pos_values_dict, pos_picked_league_dict, team_roster, n_picks_til_next_turn ):
    """
    will pick skill players before kickers or defense, in order to follow common sense
    otherwise dumbly random
    """
    poss_pos = []
    for pos,players in team_roster.items():
        n_left = n_roster_per_team[pos] - len(players)
        poss_pos += [pos for _ in range(n_left)]
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
    stratPickMaxValOverWorstNextTurn, # this is the next-best after mean remaining
    # stratPickMaxValOverWorstStarterAndWorstNextTurn, # combo of the best two strats. ignoring for now because it's somewhat arbitrary
    # stratPickMaxValOverMeanStarter, # little better than random
    stratPickMaxValOverMeanRemainingStarter, # 2nd best simple option, to value over worst starter
    stratPickMaxValOverMeanRemainingAndWorstStarter # better than either simple part on their own; not by a lot though (few points per season)
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

n_players = {'QB':32,'RB':64,'WR':72,'TE':48,'K':48}
# a dict of lists, where each list is a randomized estimated set of values for the position,
#  taken from historical data
position_values = {}

strat_totals = [[] for _ in stratlist]
draft_position_totals = [[] for _ in range(n_teams)]

n_trials = 200

for i_trial in range(n_trials):
    if VERBOSE: print '\n**************** trial {} ****************'.format( i_trial )
    for pos in n_roster_per_team.keys():
        point_data = df[df.position==pos]['fantasy_points']
        value_dist = stats.gaussian_kde( point_data )
        unnormed_vals = list(itertools.chain.from_iterable(np.sort(value_dist.resample(n_players[pos]))))[::-1]
        worst_starter_value = unnormed_vals[ n_starters[pos] ]
        # round to integers for ease of use/visualization --
        #   projections aren't nearly accurate enough for the decimals to matter
        # position_values[pos] = [int( round( x - worst_starter_value ) ) for x in unnormed_vals]
        position_values[pos] = [int( round( x ) ) for x in unnormed_vals]
        if VERBOSE: print pos, ': \n', position_values[pos]

    team_rosters = [{'QB':[],'RB':[],'WR':[],'TE':[],'K':[]} for _ in range(n_teams)]
    team_strats = getRandomStrats( n_teams )
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
            newpos = team_strat( position_values, pos_picked_league_dict, team_roster, n_picks_til_next_turn )
            posrank = pos_picked_league_dict[newpos]
            value = position_values[newpos][posrank]
            if VERBOSE: print 'team {}\'s pick ({}): {}{} ({})'.format(i_team,team_strat.__name__[9:],newpos,posrank+1,value)
            pos_picked_league_dict[newpos] = pos_picked_league_dict[newpos] + 1
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
