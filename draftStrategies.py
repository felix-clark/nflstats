import numpy as np
import random
import itertools

# ideally we avoid this global variable and pass it in...
n_roster_per_team = {'QB':1,'RB':2,'WR':2,'TE':1,'K':1,'FLEX':1}
flex_pos = ['RB', 'WR', 'TE']


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
#         val = pos_vals[npicked] - np.mean(pos_vals[0:n_roster_per_league[pos]+1])
#         if len(team_roster[pos]) < n_roster_per_team[pos] and val > maxval:
#             bestpos,maxval = pos,val
#     return bestpos

def stratPickMaxValOverMeanRemainingStarter( pos_values_dict, team_roster, **kwargs ):
    maxval = -1000
    bestpos = ''
    pos_picked_league_dict = kwargs['pos_picked_league']
    n_starters_by_round = kwargs['n_starters_up_to_round']
    for pos,pos_vals in pos_values_dict.items():
        if len(team_roster[pos]) >= n_roster_per_team[pos]: continue;
        npicked = pos_picked_league_dict[pos]
        nstarters = n_starters_by_round[pos][-1]
        mean_remaining = np.mean(pos_vals[:nstarters-npicked]) if nstarters > npicked else 0
        posval = pos_vals[0] - mean_remaining
        if posval > maxval:
            bestpos,maxval = pos,posval
    if hasExtraFlexPicks( team_roster ):
        flex_list = np.sort( list( itertools.chain.from_iterable( (pos_values_dict[pos] for pos in flex_pos) ) ) )
        n_flex_picked = sum( (pos_picked_league_dict[pos] for pos in flex_pos) )
        n_flex_starters = sum( (n_starters_by_round[pos][-1] for pos in flex_pos) )
        mean_remaining_flex = np.mean( flex_list[:n_flex_starters-n_flex_picked] )
        for pos in flex_pos:
            posval = pos_values_dict[pos][0] - mean_remaining_flex
            if posval > maxval:
                bestpos,maxval = pos,posval
    if not bestpos:
        print 'error: unable to find best position to draft.'
        if hasExtraFlexPicks( team_roster ):
            print ' we got to the flex area.'
        else:
            'we did not get to the flex area.'
    return bestpos

def stratPickMaxGeometricWeightToWorstStarter( pos_values_dict, team_roster, **kwargs ):
    maxval = -1000
    bestpos = ''
    pos_picked_league_dict = kwargs['pos_picked_league']
    n_starters_up_to_round = kwargs['n_starters_up_to_round']
    for pos,pos_vals in pos_values_dict.items():
        if len(team_roster[pos]) >= n_roster_per_team[pos]: continue
        npicked = pos_picked_league_dict[pos]
        best_in_pos = pos_vals[0]
        num_avg = n_starters_up_to_round[pos][-1] - npicked #compare to last
        if num_avg <= 0: ## this can happen if someone has taken too many of the position
            print 'bad num_avg.pos={}, nstarters={}, npicked={}'.format( pos, n_starters_up_to_round[pos][-1], npicked)
            # this might not be a big problem
        weights = np.exp( np.arange(num_avg) * np.log(num_avg)/(num_avg-1.0) )/num_avg if num_avg > 1 else [1] # exponential does just as fine, possibly better than linear
        sum_weights = sum(weights)
        weighted_vals = (w*(best_in_pos - pv) for w,pv in zip(weights,pos_vals[:num_avg]))
        val = sum(weighted_vals)/sum_weights if sum_weights > 0 else 0
        if val > maxval:
            bestpos,maxval = pos,val
    if hasExtraFlexPicks( team_roster ):
        n_flex_starters = sum( (n_starters_up_to_round[pos][-1] for pos in flex_pos) )
        n_flex_picked = sum( (pos_picked_league_dict[pos] for pos in flex_pos) )
        flex_list = np.sort( list( itertools.chain.from_iterable( (pos_values_dict[pos] for pos in flex_pos) ) ) )
        num_avg = n_flex_starters - n_flex_picked
        if num_avg < 1:
            print 'bad num_avg in flex code'
        weights = np.exp( np.arange(num_avg) * np.log(num_avg)/(num_avg-1.0) )/num_avg if num_avg > 1 else [1] # exponential does just as fine, possibly better than linear
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
    n_starters_by_round = kwargs['n_starters_up_to_round']
    for pos,pos_vals in pos_values_dict.items():
        npicked = pos_picked_league_dict[pos]
        nstarters = n_starters_by_round[pos][-1]
        best_in_pos = pos_vals[0]
        mean_remaining = np.mean(pos_vals[:nstarters-npicked]) if nstarters > npicked else 0
        mean_val = best_in_pos - mean_remaining
        worst_val = best_in_pos - worst_starter_dict[pos]
        val = mean_weight*mean_val + worst_weight*worst_val
        if len(team_roster[pos]) < n_roster_per_team[pos] and val > maxval:
            bestpos,maxval = pos,val
    if hasExtraFlexPicks( team_roster ):
        n_flex_starters = sum( (n_starters_by_round[pos][-1] for pos in flex_pos) )
        n_flex_picked = sum( (pos_picked_league_dict[pos] for pos in flex_pos) )
        flex_list = np.sort( list( itertools.chain.from_iterable( (pos_values_dict[pos] for pos in flex_pos) ) ) )
        mean_remaining = np.mean( flex_list[:n_flex_starters-n_flex_picked] ) # if n_flex_starters > n_flex_picked else 0
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

## this "worst next turn" should be replaced by the geometric weight until next turn (?)
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

def stratQBTroll( pos_valued_dict, team_roster, **kwargs ):
    """
    will draft 3 QBs first to try to remove value for later trading
    will then follow reasonable strategy to keep from sucking too much
    """
    if len(team_roster['QB']) < 3: return 'QB'
    return stratPickMaxValOverWorstStarter( pos_valued_dict, team_roster, **kwargs )

stratlist = [
    # stratPickRandom, # this one is bad enough that we shouldn't include it cuz people should be smarter
    # stratPickRandomSkill, # consistently worse than the following; should leave it out so those get more trials ( can put it back in for a baseline comparison )
    # stratPickMaxVal, # this is worse than both of the following
    stratPickMaxValOverWorstStarter, # this is the best of the simple strats
    # stratPickMaxValOverWorstNextTurn, # this is the next-best after mean remaining, but definitely inferior to the better ones
    # stratPickMaxValOverWorstStarterAndWorstNextTurn, # combo of the best two strats. ignoring for now because it's somewhat arbitrary
    # stratPickMaxValOverMeanStarter, # little better than random
    stratPickMaxValOverMeanRemainingStarter, # 2nd best simple option, to value over worst starter. clearly inferior though
    stratPickMaxGeometricWeightToWorstStarter, # better than either simple part on their own; not by a lot though (few points per season)
    stratPickMaxValOverMeanRemainingAndWorstStarterBasic, # less sophisticated average (mean + worst) than previous option. they perform too similar to compare by eye (within stat errors)
    # stratPickMaxValOverWorstEndOfRound, # this one sucks
    stratQBTroll # a dumb strategy to test ours.
]

def getRandomStrats( n_teams ):
    randfuncs = [stratlist[random.randint(0,len(stratlist)-1)] for _ in range(n_teams)]
    return randfuncs

def getSumValue( team_roster ):
    tot = 0
    for _,vals in team_roster.items():
        tot += sum(vals)
    return tot

