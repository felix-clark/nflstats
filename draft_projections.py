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
    print '\nhelp (h):\t\tprint this message'
    print 'quit (q):\t\texit program'
    print 'list (ls) [N M] :\tprint summary of N top available players and M top players at each position'
    print '<pos> [N]:\t\tenter a position (qb,rb,wr,te,flex,k) to list N top players'
    print 'pick (pop) i [i1 ...]:\tremove player(s) with index i from available player list'
    print 'lspick:\t\t\tprint summary of picked players'
    print 'unpick (push, unpop) i [i1 ...]:\tmove player(s) from picked list to available'
    print 'save (s) <output>:\tsave player lists to output'
    print 'load <output>:\t\tload from output'
    print '  coming soon...:'
    print 'find <player>:\t\tfind player (not implemented)'
    print 'pickme (?):\t\tpick but do other features e.g. save positions and handcuffs (not implemented)'
        
def verifyAndQuit():
    print 'for debugging I want to quit immediately. Come here to the code when you are ready to use this for real'
    return True ## delete up to here
    user_verify = raw_input('Are you sure you want to quit and lose all progress [y/N]? ')
    if user_verify.strip() == 'y':
        print 'Goodbye, make sure you beat Russell!'
        return True
    elif user_verify.lower().strip() == 'n':
        print 'OK then, will not quit after all.'
    else:
        print 'Did not recognize confirmation. Will not quit.'
    return False

def popFromPlayerList(index, ap, pp=None):
    """
    index: index of player to be removed from available
    """
    if index not in ap.index:
        print 'Error: The index ({}) does not indicate an available player!'.format(index)
        return
    player = ap.loc[index] # a dictionary of the entry
    # were using iloc, but the data may get re-organized so this should be safer
    if pp is not None:
        if len(pp[pp.index==index]) > 0:
            print 'It seems like the index of the player is already in the picked player list. Someone needs to clean up the logic...'
            print 'DEBUG: picked players w/index:', pp.loc[index]
            print 'DEBUG: available players w/index:', ap.loc[index]
        pp.loc[index] = player
    # player = df.pop(index) # DataFrame.pop pops a column, not a row
    name = player['name']
    pos = player['position']
    team = player['team']
    print 'removing {} - {} ({})'.format( name, pos, team )
    ap.drop(index, inplace=True)

def pushToPlayerList(index, ap, pp):
    """
    index: index of player to be removed from available
    """
    if index not in pp.index:
        print 'Error: The index ({}) does not indicate a picked player!'.format(index)
        return
    player = pp.loc[index]
    if len(ap[ap.index==index]) > 0:
        print 'It seems like the index of the picked player is already in the available player list. Someone needs to clean up the logic...'
        print 'DEBUG: picked players w/index:', pp.loc[index]
        print 'DEBUG: available players w/index:', ap.loc[index]
    # must use loc, not iloc, since positions may move
    ap.loc[index] = player
    # ap = ap.sort_values('vaws', ascending=False) # re-sort
    ap.sort_values('vaws', ascending=False, inplace=True) # re-sort
    name = player['name']
    pos = player['position']
    team = player['team']
    print 'replacing {} - {} ({})'.format( name, pos, team )
    pp.drop(index, inplace=True)

    
def printTopPosition(df, pos, ntop=24):
    if pos.upper() == 'FLEX':
        with pd.option_context('display.max_rows', None):
            print df.loc[df['position'].isin(['RB','WR','TE'])].head( ntop )
    else:
        with pd.option_context('display.max_rows', None):
            print df[df.position==pos.upper()].head( ntop )

def savePlayerList(outname, ap, pp=None):
    print 'Saving with label {}.'.format(outname)
    ap.to_csv(outname+'.csv')
    if pp is not None: pp.to_csv(outname+'_picked.csv')

def loadPlayerList(outname, ap, pp):
    print 'Loading with label {}.'.format(outname)
    if os.path.isfile(outname+'.csv'):
         # we saved it directly we could use from_csv, but read_csv is encouraged
        ap = pd.DataFrame.from_csv(outname+'.csv')
        # ap = pd.read_csv(outname+'.csv')
    else:
        print 'Could not find file {}.csv!'.format(outname)
    if os.path.isfile(outname+'_picked.csv'):
        pp = pd.DataFrame.from_csv(outname+'_picked.csv')
        print pp
    else:
        print 'Could not find file {}_picked.csv!'.format(outname)
    return ap,pp
    
def doMainLoop(ap, pp=None):
    """
    ap: dataframe with available player data
    returns: ap,pp to update the main dataframes
    """
    # TODO: implement saving the chosen players, and the option to return them to the main list in case of an input error.
    ## use "input" function to interpret the results of the input (e.g. return an object in memory with the given name)
    user_in = ''
    while not user_in:
        user_in = raw_input('\n $$ ').lower().strip()    
    insplit = [word for word in user_in.split(' ') if word]
    user_com = insplit[0]
    user_args = insplit[1:]
    if user_com in ['h','help']:
        printHelp()
    elif user_com in ['q', 'quit', 'exit']:
        if verifyAndQuit(): exit(0) # quit looping
    elif user_com in ['ls', 'list']:
        ntop = int(user_args[0]) if user_args else 8
        npos = int(user_args[1]) if user_args[1:] else 3    
        printTopChoices(ap, ntop, npos)
    elif user_com in ['lspick']:
        print 'picked players (TODO: we can stand to improve this output):'
        print pp
    elif user_com in ['pick', 'pop']:
        indices = [int(i) for i in user_args]
        for i in indices:
            popFromPlayerList( i, ap, pp )
    elif user_com in ['unpick','push','unpop']:
        indices = [int(i) for i in user_args]
        for i in indices:
            pushToPlayerList( i, ap, pp )
    elif user_com.upper() in main_positions + ['FLEX']:
        ntop = int(user_args[0]) if user_args else 8
        printTopPosition(ap, user_com, ntop)
    elif user_com in ['s', 'save']:
        outname = user_args[0] if user_args else 'draft_players'
        savePlayerList(outname, ap, pp)
    elif user_com == 'load':
        outname = user_args[0] if user_args else 'draft_players'
        ap,pp = loadPlayerList(outname,ap,pp)
    else:
        if user_in:
            print 'Unrecognized command \"{}\".'.format(user_in)
            print 'Type \"help\" for a list of available commands.'
    return ap,pp # continue looping

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

    # give some output to verify the ruleset
    print ' {} team, {} PPR'.format(n_teams, ruleset.ppREC)
    rosterstr = ''
    for pos in ['QB', 'RB', 'WR', 'TE', 'FLEX']: # there's always just 1 DST and K, right?
        nper = n_roster_per_team[pos]
        rosterstr += ' {}{} /'.format(nper, pos)
    print rosterstr[:-2]
    
    main_positions = ['QB','RB','WR','TE','K']

    posdfs = []
    for pos in main_positions:
        # TODO: don't hardcode in the source file names.
        # also, make a script to automatically clean up csv's from FP.
        filename = 'preseason_rankings/project_fp_{}s_pre2017.csv'.format(pos.lower())
        posdf = pd.read_csv(filename)
        ## TODO (low priority): try using a multi-indexed dataframe instead of decorating every entry with the position
        posdf['position'] = pos
        posdfs.append(posdf)
    # create dataframe of all available players
    ap = pd.concat(posdfs, ignore_index=True)
    # if they have no stats listed (NaN) we can treat that as a zero
    ap.fillna(0, inplace=True)

    # fill in zeros for the additional stats that aren't included in FP projections
    # this will make computing the points for the ruleset simpler
    # print 'note: no projections for two-point conversions'
    zeroed_stats = ['passing_twoptm', 'rushing_twoptm', 'receiving_twoptm']
    for st in zeroed_stats:
        if st not in ap:
            ap[st] = 0
        else:
            print '{} already in data frame!'.format( st )

    # decorate the dataframe with projections for our ruleset
    ap['projection'] = getPointsFromDataFrame(ruleset, ap)

    # print 'generates draft board using static \"value above worst starter\" quantity.'
    # print 'this strategy seems to do comparably to other simple ones, well within the larger difference determined by draft position.'
    # print 'This will deal with starters only. You may wish to delay e.g. drafting kickers until after getting bench players.'


    position_values = {}
    worst_starters_dict = {}
    n_starters_up_to_round = {}
    for pos in n_roster_per_team.keys():
        if pos == 'FLEX': continue # deal with FLEX separately
        posdf = ap[ap.position==pos]
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
        # ap[ap.position==pos]['vaws'] = ... # this uses a copy; will not assign
        ap.loc[ap.position==pos,'vaws'] = ap['projection'] - worst_value

    ap = ap[['name','team','position','projection','vaws']].sort_values('vaws', ascending=False)
    ap.reset_index(drop=True,inplace=True) # will re-number our list to sort by vaws
    
    # make an empty dataframe with these reduces columns to store the picked players
    # this might be better as another level index in the dataframe, or simply as an additional variable in the dataframe.
    # In the latter case we'd need to explicitly exclude it from print statements.
    # it's a bit kludgey to move between (and pass in-and-out of functions), so maybe unifying these two dataframes is a worthy goal.
    pp = pd.DataFrame(columns=ap.columns)

    ## print table once first ?? -- no, just wait for 'list'
    # printTopChoices(ap)
    loop=True
    while(loop):
        ## run the main program loop.
        try:
            ap, pp = doMainLoop(ap, pp)
        # handle exit(), Ctrl-C, and Ctrl-D:
        except (SystemExit, KeyboardInterrupt, EOFError):
            print 'Looks like a clean exit, so we will not create an emergency backup.'
            loop = False
            # raise # we don't want to re-raise the exception, just print our nice message.
        except:
            backup_fname = 'draft_backup'
            print 'Error: {}\nBackup save with label \"{}\".'.format( sys.exc_info(), backup_fname )
            savePlayerList(backup_fname, ap, pp)
            raise
            
    exit(0) # return from main
    
