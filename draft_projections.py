#!/usr/bin/python
import numpy as np
import pandas as pd
from scipy import stats
import os.path
import sys
import itertools
import argparse

from draftStrategies import *
from getPoints import *
from ruleset import bro_league, phys_league, dude_league

# this method will be our main output
def printTopChoices(df, ntop=8, npos=3):
    with pd.option_context('display.max_rows', None):
        print df.head(ntop)
    for pos in main_positions:
        print df[df.position==pos].head(npos)

def printHelp():
    print '[h]elp:\tprint this message'
    print '[q]uit:\texit program'
    print 'list:\tprint summary'
    print 'pick N [N1 N2 ...]: remove player with index N (... from available'
    print 'pickme: pick but do other features like save positions and handcuffs (unimplemented)'
    print 'enter a position (qb,rb,wr,te,flex,k) to list more top players'
    print 'unpick N [N1 N2 ...]: move player from picked list to available (not implemented)'
    print 'find <player>: find player (not implemented)'
        
def verifyAndQuit():
    user_verify = raw_input('Are you sure you want to quit and lose all progress [y/N]? ')
    if user_verify.strip() == 'y':
        print 'Goodbye, make sure you beat Russell!'
        return True
    elif user_verify.lower().strip() == 'n':
        print 'OK then, will not quit after all.'
    else:
        print 'Did not recognize confirmation. Will not quit.'
    return False

def popFromPlayerList( index, available, picked=None ):
    """
    index: index of player to be removed from available
    """
    player = available.iloc[index]
    # player = df.pop(index) # DataFrame.pop pops a column, not a row
    name = player['name']
    pos = player['position']
    team = player['team']
    print 'removing {} -- {} ({})'.format( name, pos, team )
    df.drop(index, inplace=True)

def printTopPosition(df, pos, ntop=24):
    if pos.upper() == 'FLEX':
        with pd.option_context('display.max_rows', None):
            print df.loc[df['position'].isin(['RB','WR','TE'])].head( ntop )
    else:
        with pd.option_context('display.max_rows', None):
            print df[df.position==pos.upper()].head( ntop )

def doMainLoop(ap, unpicked=None):
    """
    ap: dataframe with available player data
    returns: True unless loop should be exited.
    """
    # TODO: implement saving the chosen players, and the option to return them to the main list in case of an input error.
    ## use "input" command to interpret the results of the input
    user_in = ''
    while not user_in:
        user_in = raw_input('\nEnter a command (type help for assistance):\n $$ ').lower().strip()
    insplit = [word for word in user_in.split(' ') if word]
    user_com = insplit[0]
    user_args = insplit[1:]
    if user_com in ['h','help']:
        printHelp()
        return
    elif user_com in ['q','quit']:
        if verifyAndQuit(): return False # quit looping
    elif user_com in ['ls','list']:
        ntop = int(user_args[0]) if user_args else 8
        npos = int(user_args[1]) if user_args[1:] else 3
        printTopChoices(ap, ntop, npos)
    elif user_com == 'pick':
        indices = [int(i) for i in user_args]
        print indices
        for i in indices:
            popFromPlayerList( i, ap, picked=None )
    elif user_com == 'unpick':
        print 'unpicking is unimplemented'
    elif user_com.upper() in main_positions + ['FLEX']:
        ntop = int(user_args[0]) if user_args else 3
        printTopPosition(ap, user_com, ntop)
    else:
        if user_in:
            print 'Unrecognized command \"{}\".'.format(user_in)
            print 'type \"help\" for a list of commands.'
    return True # continue looping

if __name__=='__main__':
## use argument parser
    parser = argparse.ArgumentParser(description='Script to aid in real-time fantasy draft')
    parser.add_argument('--ruleset',type=str,choices=['phys','dude','bro'],default='bro',help='which ruleset to use of the leagues I am in')
    parser.add_argument('--n-teams',type=int,default=14,help='number of teams in the league')
    parser.add_argument('--n-qb',type=int,default=1,help='number of QB per team')
    parser.add_argument('--n-rb',type=int,default=2,help='number of RB per team')
    parser.add_argument('--n-wr',type=int,default=2,help='number of WR per team')
    parser.add_argument('--n-te',type=int,default=1,help='number of TE per team')
    parser.add_argument('--n-flex',type=int,default=1,help='number of FLEX per team')
    parser.add_argument('--n-k',type=int,default=1,help='number of K per team')
    parser.add_argument('--n-dst',type=int,default=1,help='number of D/ST per team')

    args = parser.parse_args()
    n_teams = args.n_teams
    n_roster_per_team['QB'] = args.n_qb
    n_roster_per_team['RB'] = args.n_rb
    n_roster_per_team['WR'] = args.n_wr
    n_roster_per_team['TE'] = args.n_te
    n_roster_per_team['FLEX'] = args.n_flex
    n_roster_per_team['K'] = args.n_k
    # n_roster_per_team['DST'] = args.n_dst
    n_roster_per_league = {}
    for pos,nper in n_roster_per_team.items():
        n_roster_per_league[pos] = nper * n_teams

    if args.ruleset == 'phys': ruleset = phys_league
    if args.ruleset == 'dude': ruleset = dude_league
    if args.ruleset == 'bro': ruleset = bro_league
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
        
    ## decorate dataframe with value above worst starter
    for pos in worst_starters_dict.keys():
        worst_value = worst_starters_dict[pos]
        # df[df.position==pos]['vaws'] = ... # this uses a copy; will not assign
        df.loc[df.position==pos,'vaws'] = df['projection'] - worst_value

    df = df[['name','team','position','projection','vaws']].sort_values('vaws', ascending=False)
    df.reset_index(drop=True,inplace=True) # will re-number our list to sort by vaws

    ## print table once first ?? -- no, just wait for 'list'
    # printTopChoices(df)
    loop=True
    while(loop):
        ## run the main program loop.
        try:
            loop=doMainLoop( df )
        except:
            backup_fname = 'draft_backup.csv'
            print 'Error: {}\nSaving as {}.'.format( sys.exc_info(), backup_fname )
            df.to_csv( backup_fname )
            raise
            
    exit(0) # return from main
    
    
