#!/usr/bin/python
import numpy as np
import pandas as pd
from scipy import stats
import os.path
import sys
import itertools
import argparse
from cmd import Cmd

# from draftStrategies import *
from getPoints import *
from ruleset import bro_league, phys_league, dude_league

# this method will be our main output
def printTopChoices(df, ntop=8, npos=3):
    with pd.option_context('display.max_rows', None):
        print df.head(ntop)
    if npos > 0:
        for pos in main_positions:
            print df[df.position == pos].head(npos)

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
        if len(pp[pp.index == index]) > 0:
            print 'It seems like the index of the player is already in the picked player list.'
            print 'Someone needs to clean up the logic...'
            print 'DEBUG: picked players w/index:', pp.loc[index]
            print 'DEBUG: available players w/index:', ap.loc[index]
        pp.loc[index] = player
    # player = df.pop(index) # DataFrame.pop pops a column, not a row
    name = player['name']
    pos = player['position']
    team = player['team']
    print 'selecting {} ({}) - {}'.format(name, team, pos)
    ap.drop(index, inplace=True)

def pushToPlayerList(index, ap, pp):
    """
    index: index of player to be removed from available
    """
    if index not in pp.index:
        print 'Error: The index ({}) does not indicate a picked player!'.format(index)
        return
    player = pp.loc[index]
    if len(ap[ap.index == index]) > 0:
        print 'It seems like the index of the picked player is already in the available player list.'
        print 'Someone needs to clean up the logic...'
        print 'DEBUG: picked players w/index:', pp.loc[index]
        print 'DEBUG: available players w/index:', ap.loc[index]
    # must use loc, not iloc, since positions may move
    ap.loc[index] = player
    # ap = ap.sort_values('vaws', ascending=False) # re-sort
    ap.sort_values('vaws', ascending=False, inplace=True) # re-sort
    name = player['name']
    pos = player['position']
    team = player['team']
    print 'replacing {} ({}) - {}'.format(name, team, pos)
    pp.drop(index, inplace=True)
    
def printTopPosition(df, pos, ntop=24):
    if pos.upper() == 'FLEX':
        with pd.option_context('display.max_rows', None):
            print df.loc[df['position'].isin(['RB', 'WR', 'TE'])].head(ntop)
    else:
        with pd.option_context('display.max_rows', None):
            print df[df.position == pos.upper()].head(ntop)

def printPickedPlayers(df):
    if df.shape[0] == 0:
        print 'No players have been picked yet.'
    else:
        with pd.option_context('display.max_rows', None):
            ## TODO: we can probably stand to improve this output:'
            print df
        print '\nPlayers picked by position:'
        # TODO: minor annoyance... this prints out an additional line with the Name and dtype.
        # would be ideal to remove it.
        print df.position.value_counts()
    
            
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
    print 'Looking for handcuffs for {} ({}) - {}...\n'.format(name, team, pos)
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
    prompt = ' $_$ '

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
        ntop,npos = 10,3
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
        ntop = 16
        if spl_args[1:]:
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
        printPickedPlayers(self.pp)
        
    def do_unpick(self, args):
        """
        usage: unpick [I]...
        moves player(s) with index(ces) I from picked list to available.
        if no index is provided, then the last player picked will be returned.
        """
        indices = []
        try:
            indices = [int(i) for i in args.split(' ') if i]
        except ValueError as e:
            print '`unpick` requires integer indices.'
            print e
        for i in indices:
            pushToPlayerList(i, self.ap, self.pp)
        if not args:
            # empty argument: unpick the last player picked
            npicked = self.pp.shape[0]
            if npicked > 0:
                lasti = self.pp.index[npicked-1]
                pushToPlayerList(lasti, self.ap, self.pp)
            else:
                print 'No players have been picked.'
            
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

    def do_sort(self, args):
        """choose a stat to sort by (unimplemented)"""
        return
            
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

if __name__ == '__main__':
## use argument parser
    parser = argparse.ArgumentParser(description='Script to aid in real-time fantasy draft')
    parser.add_argument('--ruleset', type=str, choices=['phys','dude','bro'], default='bro', help='which ruleset to use of the leagues I am in')
    parser.add_argument('--n-teams', type=int, default=14, help='number of teams in the league')
    parser.add_argument('--n-qb', type=int, default=1, help='number of QB per team')
    parser.add_argument('--n-rb', type=int, default=2, help='number of RB per team')
    parser.add_argument('--n-wr', type=int, default=2, help='number of WR per team')
    parser.add_argument('--n-te', type=int, default=1, help='number of TE per team')
    parser.add_argument('--n-flex', type=int, default=1, help='number of FLEX per team')
    parser.add_argument('--n-dst', type=int, default=1, help='number of D/ST per team')
    parser.add_argument('--n-k', type=int, default=1, help='number of K per team')
    parser.add_argument('--n-bench', type=int, default=6, help='number of bench slots per team')

    args = parser.parse_args()
    n_teams = args.n_teams
    n_roster_per_team = {
        'QB':args.n_qb,
        'RB':args.n_rb,
        'WR':args.n_wr,
        'TE':args.n_te,
        'FLEX':args.n_flex,
        'K':args.n_k,
        'BENCH':args.n_bench
    }
    # n_roster_per_team['DST'] = args.n_dst
    n_roster_per_league = {}
    for pos,nper in n_roster_per_team.items():
        n_roster_per_league[pos] = nper * n_teams

    # in principle FLEX can be defined in a different way,
    # so we'll leave this definition local so that we might change it later.
    flex_pos = ['RB', 'WR', 'TE']

    if args.ruleset == 'phys': ruleset = phys_league
    if args.ruleset == 'dude': ruleset = dude_league
    if args.ruleset == 'bro': ruleset = bro_league

    print 'Initializing with ruleset:'
    # print some output to verify the ruleset we are working with
    rulestr = '  {} team, {} PPR'.format(n_teams, ruleset.ppREC)
    if ruleset.ppPC != 0 or ruleset.ppINC != 0:
        rulestr += ', {}/{} PPC/I'.format(ruleset.ppPC, ruleset.ppINC)
    print rulestr
    rosterstr = ' '
    for pos in ['QB', 'RB', 'WR', 'TE', 'FLEX']: # there's always just 1 DST and K, right?
        nper = n_roster_per_team[pos]
        rosterstr += ' {}{} /'.format(nper, pos)
    print rosterstr[:-2]
    
    main_positions = ['QB','RB','WR','TE','K']

    posdfs = []
    for pos in main_positions:
        # TODO: don't hardcode in the source file names.
        # TODO: better yet, use e.g. Beautiful Soup to grab the latest projections from the web.
        filename = 'preseason_rankings/project_fp_{}s_pre2017.csv'.format(pos.lower())
        posdf = pd.read_csv(filename)
        ## TODO (low priority): try using a multi-indexed dataframe instead of decorating every entry with the position?
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
    # can go ahead and filter out stats once we have projections
    availdf = availdf[['name','team','position','projection']]    

    # availdf['level'] = np.nan
    # label nominal (non-flex) starters by their class
    for pos in main_positions:
        # sort the players in each position so we can grab the top indices
        availpos = availdf.loc[availdf.position==pos,:].sort_values('projection', ascending=False)
        for i_class in range(n_roster_per_team[pos]):
            ia,ib = i_class*n_teams, (i_class+1)*n_teams
            itoppos = availpos.index[ia:ib]
            icls = availdf.index.isin(itoppos)
            availdf.loc[icls,'level'] = '{}{}'.format(pos,i_class+1)
            # print availdf.loc[icls,:]
    availflex = availdf.loc[(availdf.position.isin(flex_pos)) & (availdf['level'].isnull()),:].sort_values('projection', ascending=False)
    for i_class in range(n_roster_per_team['FLEX']):
        ia,ib = i_class*n_teams, (i_class+1)*n_teams
        itoppos = availflex.index[ia:ib]
        icls = availdf.index.isin(itoppos)
        availdf.loc[icls,'level'] = 'FLEX{}'.format(i_class+1)
        
    # players that have been assigned a class so far are starters
    # use this to find the worst value of each starter and subtract it from the projection to get the "VOLS" (value over last starter)
    # this is just a static calculation right now.
    # in the future we could adjust this for draft position and dynamically update in the case of other teams making "mistakes".
    starter_mask = availdf['level'].notnull()
    starterdf = availdf.loc[starter_mask]
    for pos in main_positions:
        worst_starter_value = starterdf[starterdf.position==pos]['projection'].min()
        availdf.loc[availdf.position==pos,'vols'] = availdf['projection'] - worst_starter_value
    # print starterdf['level'].value_counts()

    # define an "absolute" bench by collecting the top projections of all players that can fit on benches (this is probably dumb, but it's a check) -- yeah it's actually too dumb
    total_bench_positions = n_roster_per_league['BENCH']
    nonsuck_pos = ['QB', 'RB', 'WR', 'TE']
    total_nonsuck_positions = sum([n_roster_per_team[pos] for pos in nonsuck_pos])
    nonstarter_mask = availdf.level.isnull()
    nonstarterdf = availdf.loc[nonstarter_mask]
    for pos in nonsuck_pos:
        # a totally-not-rigorous estimate how how many bench spots will be taken up by each position.
        # assumes K and D/ST will not be drafted multiply (not unreasonable)
        n_pos_bench = n_roster_per_team[pos] * total_bench_positions / total_nonsuck_positions
        # print pos, n_pos_bench
        pos_nsdf = nonstarterdf.loc[nonstarterdf.position==pos].sort_values('projection',ascending=False)
        ipos = pos_nsdf.index[:n_pos_bench]
        ibnch_mask = availdf.index.isin(ipos)
        availdf.loc[ibnch_mask,'level'] = 'BU'

    # print 'this many backup:'
    # print len(availdf[availdf['level'] == 'BU'])

    # now we've given the backups a class, the worst projection at each position is the worst bench value.
    # we will define this as the VORP (value over replacement player)
    # this is also a static calculation right now, but in principle it could be dynamically updated like VOLS. this might be a lot of re-computation and/or more complicated code.
    # doing this here instead of trying to grab the value from the loop above is less-than-optimized, but is more vulnerable to programmer error and edge cases.
    draftable_mask = availdf.level.notnull()
    draftable_df = availdf.loc[draftable_mask]
    for pos in main_positions:
        worst_draftable_value = draftable_df[draftable_df.position==pos]['projection'].min()
        availdf.loc[availdf.position==pos,'vorp'] = availdf['projection'] - worst_draftable_value

    ## now label remaining players as waiver wire material
    availdf.loc[availdf.level.isnull(),'level'] = 'WAIV'
    

    ## finally sort by our stat of choice for display
    availdf = availdf.sort_values('vols', ascending=False)
    availdf.reset_index(drop=True,inplace=True) # will re-number our list to sort by vols
    
    # make an empty dataframe with these reduces columns to store the picked players
    # this might be better as another level index in the dataframe, or simply as an additional variable in the dataframe.
    # In the latter case we'd need to explicitly exclude it from print statements.
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
        
    # # should this be used for bench distribution?
    # # 13 games in regular FF season
    # bye_factor = 1.0/13
    # # this is the approximate fraction of the time that each position spends on the field.
    # # this should be useful for analyzing bench value.
    # # from sportinjurypredictor.net, based on average games missed assuming 17 game season
    # # obviously rough, but captures trend and follows intuition
    # pos_injury_factor = {
    #     'QB':0.94,
    #     'RB':0.85,
    #     'WR':0.89,
    #     'TE':0.89,
    #     'DST':1.0,
    #     'K':1.0 # place kickers are rarely injured. whatever injury factor they have will be overshadowed by their bye week.
    # }
    # ## This block of code is useless right here right now. It should be used in a function that both evaluates starting projections and guesses bench value.
        
