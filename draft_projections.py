#!/usr/bin/python
import numpy as np
import pandas as pd
from scipy import stats
import os.path
import itertools

from draftStrategies import *
from getPoints import *
from ruleset import bro_league, phys_league, dude_league

n_teams = 14
n_roster_per_league = {}
for pos,nper in n_roster_per_team.items():
    n_roster_per_league[pos] = nper * n_teams


ruleset = bro_league
posdfs = []
main_positions = ['QB','RB','WR','TE','K']
for pos in main_positions:
    filename = 'preseason_rankings/project_fp_{}s_pre2017.csv'.format( pos.lower() )
    # posdf = pd.DataFrame.from_csv( filename ) # this uses the first column as an index
    posdf = pd.read_csv( filename )
    ## TODO (low priority): try using a multi-indexed dataframe instead of decorating every entry with the position
    posdf['position'] = pos
    posdfs.append( posdf )
df = pd.concat( posdfs, ignore_index=True )

# for these purposes, if they have no stats listed we can assume they had zero of that stat
df.fillna(0, inplace=True)

# fill in zeros for the additional stats that aren't included in FP projections
# this will make computing the points for the ruleset simpler
# print 'note: no projections for two-point conversions'
zeroed_stats = ['passing_twoptm', 'rushing_twoptm', 'receiving_twoptm']
for st in zeroed_stats:
    if st not in df:
        df[st] = 0
    else:
        print '{} already in data frame!'.format( st )

# decorate the dataframe with projections for our ruleset
df['projection'] = getPointsFromDataFrame( ruleset, df )


# print 'generates draft board using static \"value above worst starter\" quantity.'
# print 'this strategy seems to do comparably to other simple ones, well within the larger difference determined by draft position.'
# print 'This will deal with starters only. You may wish to delay e.g. drafting kickers until after getting bench players.'


# a dict of lists, where each list is a randomized estimated set of values for the position,
#  taken from historical data
position_values = {}
worst_starters_dict = {}
n_starters_up_to_round = {}
for pos in n_roster_per_team.keys():
    if pos == 'FLEX': continue # deal with FLEX separately
    posdf = df[df.position==pos]
    point_data = posdf['projection'] # use this simple list of projections to get the values for the worst starters
    unnormed_vals = np.sort(point_data)[::-1]
    worst_starters_dict[pos] = unnormed_vals[ n_roster_per_league[pos]-1 ]
    position_values[pos] = [int( round( x ) ) for x in unnormed_vals]
    n_starters_up_to_round[pos] = [n_teams*(i+1) for i in range(n_roster_per_team[pos])]
flex_only_list = np.sort(list(itertools.chain.from_iterable( ( position_values[pos][n_roster_per_league[pos]:] for pos in flex_pos ) ) ) )[::-1] # values of players that aren't good enough to be an WR/RB 1 or 2 (up to number on roster for each)
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
        # change the worst starter threshold (often just for WR in PPR) if there are starters in the flex category.
        worst_starters_dict[pos] = pos_vals[n_pos_in_flex-1]
        
# if VERBOSE:
#     for pos,posvals in position_values.items():
#         print '\n{}:'.format(pos)
#         index_breaks = [0]
#         index_breaks.extend(n_starters_up_to_round[pos])
#         index_breaks.append( None )
#         for i_round in range(len(index_breaks)-1):
#             i_low = index_breaks[i_round]
#             i_up = index_breaks[i_round+1]
#             if i_up: print 'round {}:'.format(i_round+1)
#             else: print 'non-starters:'
#             print posvals[i_low:i_up]

## decorate dataframe with value above worst starter
for pos in worst_starters_dict.keys():
    worst_value = worst_starters_dict[pos]
    # df[df.position==pos]['vaws'] = ... # this uses a copy; will not assign
    df.loc[df.position==pos,'vaws'] = df['projection'] - worst_value

df = df[['name','team','position','projection','vaws']].sort_values('vaws', ascending=False)
df.reset_index(drop=True,inplace=True) # will re-number our list to sort by vaws

# this method will be our main output
def printTopChoices(df, ntop=8, npos=3):
    with pd.option_context('display.max_rows', None):
        print df.head(ntop)
    for pos in main_positions:
        print df[df.position==pos].head(npos)

while(True):
    ## run the main program loop.
    # this will prompt us to remove players from the available list, and update the display.
    # TODO: implement saving the chosen players, and the option to return them to the main list in case of an input error.
    # print out the most likely chosen
    printTopChoices(df)
    ## use "input" command to interpret the results of the input
    user_in = raw_input('\nEnter a command (type help for assistance):\n $$ ').lower().strip()
    if user_in in ['h','help']:
        print '[h]elp: print this message'
        print '[q]uit: exit program'
        print 'pop N: remove player with index N from available'
        print 'enter a position (qb,rb,wr,te,flex,k) to list more top players'
        continue
    if user_in in ['q','quit']:
        user_verify = raw_input('Are you sure you want to quit and lose all progress [y/N]? ')
        if user_verify.strip() == 'y':
            print 'Goodbye, make sure you beat Russell!'
            exit(0)
        elif user_verify.lower().strip() == 'n':
            print 'Will not quit after all.'
        else:
            print 'Did not recognize confirmation. Will not quit.'
    if user_in[:4] == 'pop ':
        index = int(user_in[4:])
        # print 'attempting to remove player with index {}...'.format( index )
        player = df.iloc[index]
        # player = df.pop(index) # DataFrame.pop pops a column, not a row
        name = player['name']
        pos = player['position']
        team = player['team']
        print 'removing {} -- {} ({})'.format( name, pos, team )
        df.drop(index, inplace=True)
    if user_in.upper() in main_positions:
        with pd.option_context('display.max_rows', None):
            print df[df.position==user_in.upper()].head( n_teams*2 )
    if user_in == 'flex':
        # with pd.option_context('display.max_rows', None):
        print df.loc[df['position'].isin(['RB','WR','TE'])].head( n_teams*2 )
            
    if user_in:
        ## if we just pressed enter or entered whitespace -- don't make us pause again.
        ## otherwise, don't move on until another return in case we had something to display.
        raw_input('\nPress enter to continue...\n')
    
exit(0)
    
    
