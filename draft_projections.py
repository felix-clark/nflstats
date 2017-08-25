#!/usr/bin/python
import numpy as np
import pandas as pd
from scipy import stats
import os.path
import sys
import itertools
import argparse
from cmd import Cmd

from draftStrategies import *
from getPoints import *
from ruleset import bro_league, phys_league, dude_league

# this method will be our main output
def printTopChoices(df, ntop=8, npos=3):
    with pd.option_context('display.max_rows', None):
        print df.head(ntop)
    if npos > 0:
        for pos in main_positions:
            print df[df.position==pos].head(npos)

def verifyAndQuit():
    user_verify = raw_input('Are you sure you want to quit and lose all progress [y/N]? ')
    if user_verify.strip() == 'y':
        print 'Make sure you beat Russell.'
        exit(0)
    elif user_verify.lower().strip() == 'n':
        print 'OK then, will not quit after all.'
    else:
        print 'Did not recognize confirmation. Will not quit.'

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
    print 'removing {} - {} ({})'.format(name, pos, team)
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
    print 'replacing {} - {} ({})'.format(name, pos, team)
    pp.drop(index, inplace=True)

    
def printTopPosition(df, pos, ntop=24):
    if pos.upper() == 'FLEX':
        with pd.option_context('display.max_rows', None):
            print df.loc[df['position'].isin(['RB','WR','TE'])].head(ntop)
    else:
        with pd.option_context('display.max_rows', None):
            print df[df.position==pos.upper()].head(ntop)

def savePlayerList(outname, ap, pp=None):
    print 'Saving with label {}.'.format(outname)
    ap.to_csv(outname+'.csv')
    if pp is not None: pp.to_csv(outname+'_picked.csv')

def loadPlayerList(outname):
    print 'Loading with label {}.'.format(outname)
    if os.path.isfile(outname+'.csv'):
        # we saved it directly we could use from_csv, but read_csv is encouraged
        ap = pd.DataFrame.from_csv(outname+'.csv')
        # ap = pd.read_csv(outname+'.csv')
    else:
        print 'Could not find file {}.csv!'.format(outname)
    if os.path.isfile(outname+'_picked.csv'):
        pp = pd.DataFrame.from_csv(outname+'_picked.csv')
    else:
        print 'Could not find file {}_picked.csv!'.format(outname)
    return ap,pp

# adding features to search by team name/city/abbreviation might be nice,
#   but probably not worth the time for the additional usefulness.
#   It could also complicate the logic and create edge cases.
def findPlayer(search_words, ap, pp):
    """
    prints the players with one of the words in search_words in their name.
    useful for finding which index certain players are if they are not in the top when drafted.
    search_words: list of words to look for
    ap: dataframe of available players
    pp: dataframe of picked players
    """
    # check if any of the search words are in either of the first or last names
    # checkfunc = lamda name: [[sw in fl for fl in name.lower().strip().split(' ')].any() for sw in search_words].any()
    # we shouldn't actually need to split up into first and last names for this to work, though
    checkfunc = lambda name: any([sw in name.strip().lower() for sw in search_words])
    filtered_ap = ap[ ap['name'].map(checkfunc) ] # map creates a boolean mask
    if len(filtered_ap) == 0:
        print '\n  Could not find any available players.'
    else:
        print '\n  Available players:'
        print filtered_ap
    filtered_pp = pp[ pp['name'].map(checkfunc) ]
    if len(filtered_pp) > 0:
        # print '  Could not find any picked players.'
    # else:
        print '\n  Picked players:'
        print filtered_pp

def findHandcuff(index, ap, pp):
    """
    prints a list of players with the same team and position as the indexed player.
    ap: dataframe of available players
    pp: dataframe of picked players
    """
    player = pd.concat([ap, pp]).loc[index]
    ## the "name" attribute is the index, so need dictionary syntax to grab actual name
    name, pos, team = player['name'], player.position, player.team
    print 'Looking for handcuffs for {} - {} ({})...\n'.format(name, pos, team)
    # print ap # still full
    ah = ap[(ap.position == pos) & (ap.team == team) & (ap.name != name)]
    if len(ah) > 0:
        print 'The following handcuffs are available:'
        print ah
    ph = pp[(pp.position == pos) & (pp.team == team) & (pp.name != name)]
    if len(ph) > 0:
        print 'The following handcuffs have already been picked:'
        print ph
    print # end on a newline

def printTeams(ap, pp):
    """
    prints a list of teams in both the available and picked player lists
    """
    teams = pd.concat([ap,pp])['team']
    # unique() does not sort, but assumes a sorted list
    print teams.sort_values().unique()

def findByTeam(team, ap, pp):
    """
    prints players on the given team
    """
    available = ap[ap.team == team.upper()]
    if len(available) > 0:
        print 'Available players:'
        print available
    else:
        print 'No available players found on team {}'.format(team)
    picked = pp[pp.team == team.upper()]
    if len(picked) > 0:
        print 'Picked players:'
        print picked
    
# note that this class can be used with tab-autocompletion...
# something to play with in the further future
class MainPrompt(Cmd):
    # overriding default member variable
    prompt = ' $$ '

    # member variables to have access to the player dataframes
    ap = pd.DataFrame()
    pp = pd.DataFrame()
    
    def set_dataframes(self, ap, pp):
        self.ap = ap
        self.pp = pp
        
    def do_quit(self, _):
        """
        exit the program
        """
        verifyAndQuit()
    def do_q(self, args):
        """alias for `quit`"""
        self.do_quit(args)
    def do_exit(self, args):
        """alias for `quit`"""
        self.do_quit(args)
    def do_ls(self, args):
        """
        usage: ls [N [M]]
        prints N top available players and M top players at each position
        """
        spl_args = [w for w in args.split(' ') if w]
        ntop,npos = 8,3
        try:
            if spl_args: ntop = int(spl_args[0])
            if spl_args[1:]: npos = int(spl_args[1])
        except ValueError as e:
            print '`ls` requires integer arguments.'
            print e
        printTopChoices(self.ap, ntop, npos)
    def do_list(self, args):
        """alias for `ls`"""
        self.do_ls(args)
    def do_lspos(self, args):
        """
        usage: lspos POS [N]
        prints N top available players at POS where POS is one of (qb|rb|wr|te|flex|k)
        """
        spl_args = [w for w in args.split(' ') if w]
        if not spl_args:
            print 'Provide a position (qb|rb|wr|te|flex|k) to the `lspos` command.'
            return
        pos = spl_args[0]
        ntop = 8
        try:
            ntop = int(spl_args[1])
        except ValueError:
            print '`lspos` requires an integer second argument.'
        printTopPosition(self.ap, pos, ntop)
        
    def do_find(self, args):
        """
        usage: find NAME...
        finds and prints players with the string(s) NAME in their name.
        """
        search_words = [word for word in args.split(' ') if word]
        findPlayer(search_words, self.ap, self.pp)

    def do_handcuff(self, args):
        """
        usage: handcuff I...
        find potential handcuffs for player with index(ces) I
        """
        indices = []
        try:
            indices = [int(i) for i in args.split(' ') if i]
        except ValueError as e:
            print '`handcuff` requires integer indices.'
            print e
        for i in indices:
            findHandcuff(i, self.ap, self.pp)

    def do_pick(self, args):
        """
        usage: pick I...
        remove player(s) with index(ces) I from available player list
        """
        indices = []
        try:
            indices = [int(i) for i in args.split(' ') if i]
        except ValueError as e:
            print '`pick` requires integer indices.'
            print e
        for i in indices:
            popFromPlayerList(i, self.ap, self.pp)
    def do_pop(self, args):
        """alias for `pick`"""
        self.do_pick(args)
            
    def do_lspick(self, args):
        """prints summary of players that have already been picked"""
        ## (TODO: we can probably stand to improve this output):'
        with pd.option_context('display.max_rows', None):
            print self.pp
        
    def do_unpick(self, args):
        """move player(s) from picked list to available"""
        indices = []
        try:
            indices = [int(i) for i in args.split(' ') if i]
        except ValueError as e:
            print '`unpick` requires integer indices.'
            print e
        for i in indices:
            pushToPlayerList(i, self.ap, self.pp)
    def do_unpop(self, args):
        """alias for unpick"""
        self.do_unpick(args)

    def do_team(self, args):
        """
        usage: team [TEAM]
        Lists players by TEAM abbreviation or, if no argument is provided, lists the available teams.
        """
        if args:
            findByTeam(args, self.ap, self.pp)
        else:
            printTeams(self.ap, self.pp)
        
    def do_save(self, args):
        """
        usage: save [OUTPUT]
        saves player lists to OUTPUT.csv (default OUTPUT is draft_players)
        """
        outname = args if args else 'draft_players'
        savePlayerList(outname, self.ap, self.pp)

    def do_load(self, args):
        """
        usage load [OUTPUT]
        loads player lists from OUTPUT.csv (default OUTPUT is draft_players)
        """        
        outname = args if args else 'draft_players'
        self.ap, self.pp = loadPlayerList(outname)

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

    print 'Initializing with ruleset:'
    # print some output to verify the ruleset we are working with
    print '  {} team, {} PPR'.format(n_teams, ruleset.ppREC)
    rosterstr = ' '
    for pos in ['QB', 'RB', 'WR', 'TE', 'FLEX']: # there's always just 1 DST and K, right?
        nper = n_roster_per_team[pos]
        rosterstr += ' {}{} /'.format(nper, pos)
    print rosterstr[:-2]
    
    main_positions = ['QB','RB','WR','TE','K']

    posdfs = []
    for pos in main_positions:
        # TODO: don't hardcode in the source file names.
        # also, make a script to automatically clean up csv's from FP.
        # TODO: better yet, use Beautiful Soup to grab the latest projections.
        filename = 'preseason_rankings/project_fp_{}s_pre2017.csv'.format(pos.lower())
        posdf = pd.read_csv(filename)
        ## TODO (low priority): try using a multi-indexed dataframe instead of decorating every entry with the position
        posdf['position'] = pos
        posdfs.append(posdf)
    # create dataframe of all available players
    availdf = pd.concat(posdfs, ignore_index=True)
    # if they have no stats listed (NaN) we can treat that as a zero
    availdf.fillna(0, inplace=True)

    # fill in zeros for the additional stats that aren't included in FP projections
    # this will make computing the points for the ruleset simpler
    # print 'note: no projections for two-point conversions'
    zeroed_stats = ['passing_twoptm', 'rushing_twoptm', 'receiving_twoptm']
    for st in zeroed_stats:
        if st not in availdf:
            availdf[st] = 0
        else:
            print '{} already in data frame!'.format(st)

    # decorate the dataframe with projections for our ruleset
    availdf['projection'] = getPointsFromDataFrame(ruleset, availdf)

    # print 'generates draft board using static \"value above worst starter\" quantity.'
    # print 'this strategy seems to do comparably to other simple ones, well within the larger difference determined by draft position.'
    # print 'This will deal with starters only. You may wish to delay e.g. drafting kickers until after getting bench players.'


    position_values = {}
    worst_starters_dict = {}
    n_starters_up_to_round = {}
    for pos in n_roster_per_team.keys():
        if pos == 'FLEX': continue # deal with FLEX separately
        posdf = availdf[availdf.position==pos]
        point_data = posdf['projection'] # use this simple list of projections to get the values for the worst starters
        unnormed_vals = np.sort(point_data)[::-1]
        worst_starters_dict[pos] = unnormed_vals[ n_roster_per_league[pos]-1 ]
        position_values[pos] = [int(round(x)) for x in unnormed_vals]
        n_starters_up_to_round[pos] = [n_teams*(i+1) for i in range(n_roster_per_team[pos])]
    flex_only_list = np.sort(list(itertools.chain.from_iterable((position_values[pos][n_roster_per_league[pos]:] for pos in flex_pos))))[::-1] # values of players that aren't good enough to be an WR/RB 1 or 2 (up to number on roster for each)
    worst_flex_value = flex_only_list[n_roster_per_league['FLEX']-1]
    worst_starters_dict['FLEX'] = worst_flex_value # should be worst than the worst starter of each position independently
    for pos in flex_pos:
        pos_vals = position_values[pos]
        n_pos_in_flex = len(list(itertools.takewhile(lambda n: n >= worst_flex_value, pos_vals)))
        pos_n_starters = n_starters_up_to_round[pos]
        # need to check that there are actually any players in this position that would work in FLEX.
        # in PPR for instance, often FLEX is all WR
        if n_pos_in_flex > pos_n_starters[-1]:
            pos_n_starters.append(n_pos_in_flex)
            # change the worst starter threshold (often just for WR in PPR) if there are starters in the flex category.
            worst_starters_dict[pos] = pos_vals[n_pos_in_flex-1]
        
    ## decorate dataframe with value above worst starter
    for pos in worst_starters_dict.keys():
        worst_value = worst_starters_dict[pos]
        # ap[ap.position==pos]['vaws'] = ... # this uses a copy; will not assign
        availdf.loc[availdf.position==pos,'vaws'] = availdf['projection'] - worst_value

    availdf = availdf[['name','team','position','projection','vaws']].sort_values('vaws', ascending=False)
    availdf.reset_index(drop=True,inplace=True) # will re-number our list to sort by vaws
    
    # make an empty dataframe with these reduces columns to store the picked players
    # this might be better as another level index in the dataframe, or simply as an additional variable in the dataframe.
    # In the latter case we'd need to explicitly exclude it from print statements.
    # it's a bit kludgey to move between (and pass in-and-out of functions), so maybe unifying these two dataframes is a worthy goal.
    pickdf = pd.DataFrame(columns=availdf.columns)

    prompt = MainPrompt()
    prompt.set_dataframes(availdf, pickdf)
    try:
        prompt.cmdloop()
    except (SystemExit, KeyboardInterrupt, EOFError):
        # a system exit, Ctrl-C, or Ctrl-D can be treated as a clean exit.
        #  will not create an emergency backup.
        print 'Goodbye!'
    except:
        backup_fname = 'draft_backup'
        print 'Error: {}'.format(sys.exc_info())
        print 'Backup save with label \"{}\".'.format(backup_fname)
        savePlayerList(backup_fname, prompt.ap, prompt.pp)
        raise
        
