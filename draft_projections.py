#!/usr/bin/python
import numpy as np
import pandas as pd
from scipy import stats
import os.path
import itertools
import random

from draftStrategies import *
from getPoints import *
from ruleset import bro_league, phys_league, dude_league

VERBOSE = True

random.seed() # no argument (or None) seeds from urandom if available

n_teams = 14
# these now must be in draftStrategies:
# n_roster_per_team = {'QB':1,'RB':2,'WR':2,'TE':1,'K':1,'FLEX':1}
# flex_pos = ['RB', 'WR', 'TE']
n_roster_per_league = {}
for pos,nper in n_roster_per_team.items():
    n_roster_per_league[pos] = nper * n_teams


dfs = {}
for pos in ['QB','RB','WR','TE','K']: # no DST right now
    datafilename = 'preseason_rankings/project_fp_{}s_pre2017.csv'.format( pos.lower() )
    dfs[pos] = pd.DataFrame.from_csv( datafilename )
    # decorateDataFrame( bro_league, dfs[pos], 'fantasy_points_bro' ) # need to generalize this

# would be MUCH simpler if we could just combine all these into one dataframe
## probably next TODO.
decorateQBDataFrame( bro_league, dfs['QB'], 'fantasy_points_bro' )
decorateRBDataFrame( bro_league, dfs['RB'], 'fantasy_points_bro' )
decorateWRDataFrame( bro_league, dfs['WR'], 'fantasy_points_bro' )
decorateTEDataFrame( bro_league, dfs['TE'], 'fantasy_points_bro' )
decorateKDataFrame( bro_league, dfs['K'], 'fantasy_points_bro' )

print 'TODO: will generate draft board using static value over worst starter as an organizer.'
print 'this strategy seems to do comparably to other simple ones, well within the difference typically driven by draft position.'
print 'This will deal with starters only. You may wish to delay e.g. drafting kickers until after getting bench players.'


# a dict of lists, where each list is a randomized estimated set of values for the position,
#  taken from historical data
position_values = {}

worst_starters_dict = {}
n_starters_up_to_round = {}
for pos in n_roster_per_team.keys():
    if pos == 'FLEX': continue # deal with FLEX separately
    df = dfs[pos]#['fantasy_points_bro']
    print df
    point_data = df['fantasy_points_bro'] # point data isn't all we want -- we want to list names and VOW!
    unnormed_vals = np.sort(point_data)[::-1]
    worst_starters_dict[pos] = unnormed_vals[ n_roster_per_league[pos]-1 ]
    position_values[pos] = [int( round( x ) ) for x in unnormed_vals]
    n_starters_up_to_round[pos] = [n_teams*(i+1) for i in range(n_roster_per_team[pos])]
flex_only_list = np.sort(list(itertools.chain.from_iterable( ( position_values[pos][n_roster_per_league[pos]:] for pos in flex_pos ) ) ) )[::-1] # values of players that aren't good enough to be an WR/RB 1 or 2 (up to number on roster for each)
# print 'flex only list: ', flex_only_list
worst_flex_value = flex_only_list[n_roster_per_league['FLEX']-1]
worst_starters_dict['FLEX'] = worst_flex_value # should be worst than the worst starter of each position independently
for pos in flex_pos:
    pos_vals = position_values[pos]
    n_pos_in_flex = len(list(itertools.takewhile(lambda n: n >= worst_flex_value, pos_vals)))
    pos_n_starters = n_starters_up_to_round[pos]
    # need to check that there are actually any players in this position that would work in FLEX.
    # in PPR for instance, often FLEX is all WR
    if n_pos_in_flex > pos_n_starters[-1]:
        pos_n_starters.append( n_pos_in_flex )
        worst_starters_dict[pos] = pos_vals[n_pos_in_flex-1]
if VERBOSE:
    for pos,posvals in position_values.items():
        print '\n{}:'.format(pos)
        index_breaks = [0]
        index_breaks.extend(n_starters_up_to_round[pos])
        index_breaks.append( None )
        for i_round in range(len(index_breaks)-1):
            i_low = index_breaks[i_round]
            i_up = index_breaks[i_round+1]
            if i_up: print 'round {}:'.format(i_round+1)
            else: print 'non-starters:'
            print posvals[i_low:i_up]
                
            # for i_posvals in range(starters_by_n_to_print):
        
team_rosters = [{'QB':[],'RB':[],'WR':[],'TE':[],'K':[]} for _ in range(n_teams)]
team_strats = getRandomStrats( n_teams )
# keep track of the number of each position drafted by the league
pos_picked_league_dict = {'QB':0,'RB':0,'WR':0,'TE':0,'K':0}
n_draft_rounds = sum( (n for (_,n) in n_roster_per_team.items()) )

for i_round in range(n_draft_rounds):
    if VERBOSE: print 'round {}:'.format( i_round )
    rnteams = range(n_teams)
    rev_round = i_round % 2 == 1 # every other round in a snake draft is in reverse order
    team_is = rnteams[::-1] if rev_round else rnteams
    for i_team in team_is:
        team_roster = team_rosters[i_team]
        team_strat = team_strats[i_team]
        n_picks_til_next_turn = picksTilNextTurn( n_teams, i_team, i_round )
        newpos = team_strat( position_values, team_roster,
                             worst_starters=worst_starters_dict,
                             n_starters_up_to_round=n_starters_up_to_round,
                             pos_picked_league=pos_picked_league_dict,
                             n_picks_til_next_turn=n_picks_til_next_turn )
        # value = position_values[newpos][posrank]
        try:
            value = position_values[newpos].pop(0) # now take the top one and remove it from the list
        except KeyError:
            print 'offending function is ', team_strat.__name__
            exit(1)
                
        posrank = pos_picked_league_dict[newpos]+1
        if VERBOSE: print 'team {}\'s pick ({}): {}{} ({})'.format(i_team,team_strat.__name__[9:],newpos,posrank,value)
        pos_picked_league_dict[newpos] = pos_picked_league_dict[newpos] + 1 # increment the count of each position picked
        team_roster[newpos].append( value )
