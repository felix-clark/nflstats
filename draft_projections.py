#!/usr/bin/python
# import numpy as np
import sys
import os.path
import argparse
from cmd import Cmd
from itertools import takewhile
import pandas as pd

# from draftStrategies import *
from getPoints import *
from ruleset import bro_league, phys_league, dude_league

## TODO: make this compatible with python 3 :,(
##   (it's probably 95% print statement parentheses)

def evaluate_roster(rosdf, n_roster_per_team, flex_pos):
    """
    applies projection for season points, with an approximation for bench value
    returns tuple of starter, bench value
    """

    numplayers = len(rosdf)
    numroster = sum([n_roster_per_team[pos] for pos in n_roster_per_team])
    if numplayers < numroster:
        print 'This roster is not full.'
    if numplayers > numroster:
        print 'This roster has too many players.'
    
    starterval, benchval = 0, 0
    i_st = [] # the indices of the players we have counted so far
    for pos in ['QB', 'RB', 'WR', 'TE', 'K']:
        n_starters = n_roster_per_team[pos]
        rospos = rosdf[rosdf.position == pos].sort_values('projection', ascending=False)
        i_stpos = rospos.index[:n_starters]
        val = rospos[rospos.index.isin(i_stpos)]['projection'].sum()
        starterval = starterval + val
        i_st.extend(i_stpos)

    n_flex = n_roster_per_team['FLEX']
    rosflex = rosdf[(~rosdf.index.isin(i_st)) & (rosdf.position.isin(flex_pos))].sort_values('projection', ascending=False)
    i_flex = rosflex.index[:n_flex]
    starterval = starterval + rosflex[rosflex.index.isin(i_flex)]['projection'].sum()
    i_st.extend(i_flex)
    
    print '  starting lineup:'
    print rosdf[rosdf.index.isin(i_st)].drop(['vols','volb'], axis=1)

    benchdf = rosdf[~rosdf.index.isin(i_st)].drop(['vols','volb'], axis=1)
    if len(benchdf) > 0:
        print '  bench:'
        print benchdf

    ## we're gonna do a really dumb estimation for bench value
    # and pretend that the chance of a bench player being used
    # is the same as that of a starter being out.
    # we'll call this an "index" to avoid commiting to a meaning right now :)

    # 13 games in regular FF season. we'll pretend they're independent.
    bye_factor = (13.0-1.0)/13.0

    # this is the approximate fraction of the time that a player in
    #  each position spends on the field uninjured.
    # from sportinjurypredictor.net, based on average games missed assuming a 17 game season
    # obviously rough, but captures trend and follows intuition
    pos_injury_factor = {'QB':0.94, 'RB':0.85, 'WR':0.89, 'TE':0.89, 'DST':1.0, 'K':1.0}
    for _,row in benchdf.iterrows():
        pos = row.position
        benchval = benchval + (1 - bye_factor*pos_injury_factor[pos])*row.projection
        
    # round values to whole numbers for josh, who doesn't like fractions :)
    print '\nprojected starter points:\t{}'.format(int(round(starterval)))
    print 'estimated bench value:\t{}\n'.format(int(round(benchval)))
    return starterval, benchval

def find_by_team(team, ap, pp):
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
    
def find_handcuff(index, ap, pp):
    """
    prints a list of players with the same team and position as the indexed player.
    ap: dataframe of available players
    pp: dataframe of picked players
    """
    player = pd.concat([ap, pp]).loc[index]
    ## the "name" attribute is the index, so need dictionary syntax to grab actual name
    name, pos, team = player['name'], player.position, player.team
    print 'Looking for handcuffs for {} ({}) - {}...\n'.format(name, team, pos)
    ah = ap[(ap.position == pos) & (ap.team == team) & (ap.name != name)]
    if len(ah) > 0:
        print 'The following potential handcuffs are available:'
        print ah
    ph = pp[(pp.position == pos) & (pp.team == team) & (pp.name != name)]
    if len(ph) > 0:
        print 'The following handcuffs have already been picked:'
        print ph
    print # end on a newline

# adding features to search by team name/city/abbreviation might be nice,
#   but probably not worth the time for the additional usefulness.
#   It could also complicate the logic and create edge cases.
def find_player(search_words, ap, pp):
    """
    prints the players with one of the words in search_words in their name.
    useful for finding which index certain players are if they are not in the top when drafted.
    search_words: list of words to look for
    ap: dataframe of available players
    pp: dataframe of picked players
    """
    # check if any of the search words are in the full name
    checkfunc = lambda name: any([sw in name.strip().lower() for sw in search_words])
    filtered_ap = ap[ap['name'].map(checkfunc)] # map creates a boolean mask
    if len(filtered_ap) == 0:
        print '\n  Could not find any available players.'
    else:
        print '\n  Available players:'
        print filtered_ap
    filtered_pp = pp[pp['name'].map(checkfunc)]
    if len(filtered_pp) > 0:
        # print '  Could not find any picked players.'
    # else:
        print '\n  Picked players:'
        print filtered_pp

def load_player_list(outname):
    """loads the available and picked player data from the label \"outname\""""
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
    return ap, pp

def pop_from_player_list(index, ap, pp=None, manager=None, pickno=None):
    """
    index: index of player to be removed from available
    """
    if index not in ap.index:
        raise IndexError('The index ({}) does not indicate an available player!'.format(index))
    player = ap.loc[index] # a dictionary of the entry
    # were using iloc, but the data may get re-organized so this should be safer
    if pp is not None:
        if len(pp[pp.index == index]) > 0:
            print 'It seems like the index of the player is already in the picked player list.'
            print 'Someone needs to clean up the logic...'
            print 'DEBUG: picked players w/index:', pp.loc[index]
            print 'DEBUG: available players w/index:', ap.loc[index]
        pp.loc[index] = player
        if manager is not None:
            pp.loc[index, 'manager'] = manager
             # this method of making the variable an integer is ugly and over time redundant.
            pp.manager = pp.manager.astype(int)
        if pickno is not None:
            pp.loc[index, 'pick'] = pickno
            pp.pick = pp.pick.astype(int)
        # player = df.pop(index) # DataFrame.pop pops a column, not a row
    name = player['name']
    pos = player['position']
    team = player['team']
    print 'selecting {} ({}) - {}'.format(name, team, pos)
    ap.drop(index, inplace=True)

def print_picked_players(df):
    """prints the players in dataframe df as if they have been selected"""
    if df.shape[0] == 0:
        print 'No players have been picked yet.'
    else:
        with pd.option_context('display.max_rows', None):
            ## TODO: we can probably still stand to improve this output:'
            print df.drop([col for col in ['manager', 'pick'] if col in df], axis=1)
        print '\nPlayers picked by position:'
        # TODO: minor annoyance... this prints out an additional line with the Name and dtype.
        # would be ideal to remove it.
        print df.position.value_counts()

def print_teams(ap, pp):
    """
    prints a list of teams in both the available and picked player lists
    """
    teams = pd.concat([ap, pp])['team']
    # unique() does not sort, but assumes a sorted list
    print teams.sort_values().unique()

# this method will be our main output
def print_top_choices(df, ntop=8, npos=3):
    print '   DRAFT BOARD   '.center( 80, '*')
    with pd.option_context('display.max_rows', None):
        print df.head(ntop)
    if npos > 0:
        # ... let's just leave out the kickers, they can be requested specifically with e.g. lspos k
        main_positions = ['QB', 'RB', 'WR', 'TE']
        for pos in main_positions:
            print df[df.position == pos].head(npos)

def print_top_position(df, pos, ntop=24):
    """prints the top `ntop` players in the position in dataframe df"""
    if pos.upper() == 'FLEX':
        with pd.option_context('display.max_rows', None):
            print df.loc[df['position'].isin(['RB', 'WR', 'TE'])].head(ntop)
    else:
        with pd.option_context('display.max_rows', None):
            print df[df.position == pos.upper()].head(ntop)

def push_to_player_list(index, ap, pp):
    """
    index: index of player to be removed from available
    ap: dataframe of available players
    pp: dataframe of picked players
    """
    if index not in pp.index:
        raise IndexError('The index ({}) does not indicate a picked player!'.format(index))
    player = pp.loc[index]
    if len(ap[ap.index == index]) > 0:
        print 'The index of the picked player is already in the available player list.'
        print 'Someone needs to clean up the logic...'
        print 'DEBUG: picked players w/index:', pp.loc[index]
        print 'DEBUG: available players w/index:', ap.loc[index]
    # must use loc, not iloc, since positions may move
    ap.loc[index] = player
    ap.sort_values('vols', ascending=False, inplace=True) # re-sort
    name = player['name']
    pos = player['position']
    team = player['team']
    print 'replacing {} ({}) - {}'.format(name, team, pos)
    pp.drop(index, inplace=True)

# maybe not necessary rn
# could give it to AI managers as alternative strategy
# def pick_vona(ap, pp, coach, coaches_til_next):
#     #TODO
#     return

# instead of passing managers_til_next, could pass "forward" or "backward"
def print_vona(ap, pp, manager, managers_til_next):
    """prints VONA at each position, assuming each manager picks with VOLS"""
    # TODO
    # need a function that walks through and predicts other managers' picks:
    # predict_next_board(ap, pp, manager, managers_til_next) (or predict_next_available)
    return

def save_player_list(outname, ap, pp=None):
    """saves the available and picked player sets with label "outname"."""
    print 'Saving with label {}.'.format(outname)
    ap.to_csv(outname+'.csv')
    if pp is not None:
        pp.to_csv(outname+'_picked.csv')

def verify_and_quit():
    user_verify = raw_input('Are you sure you want to quit and lose all progress [y/N]? ')
    if user_verify.strip() == 'y':
        print 'Make sure you beat Russell.'
        exit(0)
    elif user_verify.lower().strip() == 'n':
        print 'OK then, will not quit after all.'
    else:
        print 'Did not recognize confirmation. Will not quit.'

# note that this class can be used with tab-autocompletion...
# can we give it more words in its dictionary? (e.g. player names)
class MainPrompt(Cmd):
    """
    This is the main command loop for the program.
    The important data will mostly be copied into member variables so it has easy access.
    """
    # overriding default member variable
    prompt = ' $$ '

    # member variables to have access to the player dataframes
    ap = pd.DataFrame()
    pp = pd.DataFrame()

    flex_pos = ['RB', 'WR', 'TE']

    # member variables for DRAFT MODE !!!
    draft_mode = False
    i_manager_turn = None
    manager_picks = [] # when initialized, looks like [1,2,3,...,11,12,12,11,10,...]
    user_manager = None
    manager_names = {}
    n_teams = None
    n_roster_per_team = {}

    # this is a member function we are overriding
    def emptyline(self):
        """
        do nothing when an empty line is entered.
        (without this definition, the last command is repeated)
        """
        pass

    def _advance_snake(self):
        """move up one step in the snake draft"""
        self.i_manager_turn = self.i_manager_turn + 1
        if self.i_manager_turn >= len(self.manager_picks):
            print 'Draft is over!'
            conf = raw_input('Are you done [y/N]? ')
            if conf != 'y':
                print 'Undoing last pick'
                return self._regress_snake()
            self.draft_mode = False
            i_manager_turn = None
            manager_picks = []
            print 'You\'re done! Type `evaluate all` to see a summary for each team.'
        self._set_prompt()

    def _regress_snake(self):
        """move up one step in the snake draft"""
        self.i_manager_turn = self.i_manager_turn - 1
        self._set_prompt()

    def _get_current_manager(self):
        """returns number of current manager"""
        return self.manager_picks[self.i_manager_turn] if self.manager_picks else None

    def _get_manager_name(self, num=None):
        """
        returns name of manager number num
        if num is None then uses current manager
        """
        if num is None:
            num = self._get_current_manager()
        return self.manager_names[num] if num in self.manager_names else 'manager {}'.format(num)
        
    def _set_prompt(self):
        if self.draft_mode:
            manno = self._get_current_manager()
            managername = self._get_manager_name()
            # vols = self.pick_vols(manno)
            
            if self.user_manager is not None and manno == self.user_manager:
                self.prompt = ' s~({},{})s~  your pick! $$ '.format(manno, self.i_manager_turn+1)
            else:
                self.prompt = ' s~({},{})s~  {}\'s pick $$ '.format(manno, self.i_manager_turn+1, managername)
        else:
            self.prompt = ' $$ '

    def _get_manager_roster(self, manager):
        """returns dataframe of manager's roster"""
        # this will return a small copy w/ the manager index removed
        if len(self.pp) == 0:
            return self.pp # there isn't anything in here yet, and we haven't added the "manager" branch
        return self.pp[self.pp.manager == manager].drop('manager', inplace=False, axis=1)

    def _update_vorp(self):
        """
        updates the VORP values in the available players dataframe
        based on how many players in that position have been picked.
        a replacement for a 1-st round pick comes from the top of the bench,
        while a replacement for a bottom bench player comes from the waivers.
        """

        # not the smoothest or safest way to get the positions...
        positions = [pos for pos in self.n_roster_per_team.keys() if pos not in ['FLEX', 'BENCH']]
        
        for pos in positions:
            # also implement: maximum players on roster
            # TODO: implement that max in the WAIV designation as well (elsewhere)
            # maximum probably won't matter for most drafts, so de-prioritize it
            # while your draft is in an hour :E
            
            pos_picked = self.pp[self.pp.position == pos]
            n_pos_picked = len(pos_picked.index)
            n_waiv_picked = len(pos_picked[pos_picked.level == 'WAIV'].index)
            # if people picked waiver-level players, then we can shed the next-worst bench player from our calculations
            # we can still shed all WAIV players since this raises the value of the threshold
            ## list of possible baseline players -- they will mostly be in the bench
            # in theory the first baseline should be the -worst- starter, not a bench player
            # but that's a small correction to implement later
            pos_draftable = self.ap[(self.ap.position == pos) & (self.ap.level != 'WAIV')]
            n_pos_draftable = len(pos_draftable.index) - n_waiv_picked
            vorp_baseline = 0
            if n_pos_draftable <= 0:
                # no more "draftable" players -- vorp should be zero for top
                vorp_baseline = pos_draftable['projection'].max()
            else:
                frac_through_bench = n_pos_picked * 1.0 / (n_pos_picked + n_pos_draftable)
                pos_baseline = pos_draftable[pos_draftable.level == 'BU']
                n_pos_baseline = len(pos_baseline.index)
                if n_pos_baseline == 0:
                    # this can happen, e.g. with kickers who have no "backup" level players
                    self.ap.loc[self.ap.position == pos, 'vorp'] = self.ap['vols']
                    continue
                index = int(frac_through_bench * n_pos_baseline)
                if index >= len(pos_baseline):
                    print 'warning: check index here later'
                    index = len(pos_baseline-1)
                vorp_baseline = pos_baseline['projection'].sort_values( ascending=False ).iloc[index]
            self.ap.loc[self.ap.position == pos, 'vorp'] = self.ap['projection'] - vorp_baseline

    def do_evaluate(self, args):
        """evaluate one or more rosters"""
        if not self.draft_mode:
            print 'roster from selected players:'
            evaluate_roster(self.pp,
                            self.n_roster_per_team,
                            self.flex_pos)
            return
        indices = []
        if not args:
            indices = [self._get_current_manager()]
        elif args.lower() == 'all':
            indices = range(1,self.n_teams+1)
        else:
            try:
                indices = [int(a) for a in args.split(' ')]
            except ValueError as e:
                print 'Could not interpret managers to evaluate.'
                print e
        for i in indices:
            print '{}\'s roster:'.format(self._get_manager_name(i))
            evaluate_roster(self._get_manager_roster(i),
                            self.n_roster_per_team,
                            self.flex_pos)
            
    def do_exit(self, args):
        """alias for `quit`"""
        self.do_quit(args)

    def do_find(self, args):
        """
        usage: find NAME...
        finds and prints players with the string(s) NAME in their name.
        """
        search_words = [word for word in args.split(' ') if word]
        find_player(search_words, self.ap, self.pp)

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
            find_handcuff(i, self.ap, self.pp)

    def do_list(self, args):
        """alias for `ls`"""
        self.do_ls(args)

    def do_load(self, args):
        """
        usage load [OUTPUT]
        loads player lists from OUTPUT.csv (default OUTPUT is draft_players)
        """
        print 'if you quit from draft mode, then that state will not be saved.'
        print 'in principle this can be extracted from the manager of the picked players,'
        print 'but that is not yet implemented.'
        outname = args if args else 'draft_players'
        self.ap, self.pp = load_player_list(outname)

    def do_ls(self, args):
        """
        usage: ls [N [M]]
        prints N top available players and M top players at each position
        """
        spl_args = [w for w in args.split(' ') if w]
        ntop, npos = 10, 3
        try:
            if spl_args:
                ntop = int(spl_args[0])
            if spl_args[1:]:
                npos = int(spl_args[1])
        except ValueError as e:
            print '`ls` requires integer arguments.'
            print e
        self._update_vorp()
        print_top_choices(self.ap, ntop, npos)

    def do_lspick(self, args):
        """prints summary of players that have already been picked"""
        # we already have the `roster` command to look at a roster of a manager;
        # TODO: let optional argument select by e.g. position?
        print_picked_players(self.pp)

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
        print_top_position(self.ap, pos, ntop)

    def do_name(self, args):
        """
        name [N] <manager name>
        names the current manager if first argument is not an integer
        """
        if not args:
            print 'need to pass more than that to `name`'
            return
        mannum = self._get_current_manager()
        splitargs = args.split(' ')
        firstarg = splitargs[0]
        manname = args
        try:
            mannum = int(firstarg)
            manname = ' '.join(splitargs[1:])
        except ValueError as e:
            if mannum is None:
                print e
                print 'Could not figure out a valid manager to assign name to.'
                print 'If not in draft mode, enter a number manually.'
        self.manager_names[mannum] = manname
        self._set_prompt()

    def do_next_managers(self, _):
        """
        usage: next_managers
        prints the remaining open starting lineup positions of the managers
        that will have (two) picks before this one's next turn.
        """
        if not self.draft_mode:
            print 'this command is only available in draft mode.'
            return
        # first we get the list of managers the will go before our next turn
        i_man = self.i_manager_turn
        current_team = self.manager_picks[i_man]
        comp_mans = []
        for man in self.manager_picks[i_man:]:
            if man not in comp_mans:
                comp_mans.append(man)
            else:
                break
        comp_mans.remove(current_team) # don't include our own roster

        # here we loop through the managers and see how many starting spots they have
        starting_pos = [pos for pos,numpos in self.n_roster_per_team.items()
                        if numpos > 0 and pos not in ['FLEX', 'BENCH']]
        pos_totals = {key:0 for key in starting_pos}
        pos_totals['FLEX'] = 0
        for man in comp_mans:
            print '{} needs starters at:'.format(self._get_manager_name(man))
            roster = self._get_manager_roster(man)
            for pos in starting_pos:
                hasleft = self.n_roster_per_team[pos] - len(roster[roster.position == pos])
                if hasleft > 0:
                    print '{}: {}'.format(pos, hasleft)
                    pos_totals[pos] = pos_totals[pos] + hasleft
            flexused = sum([min(0, len(roster[roster.position == pos])
                            - self.n_roster_per_team[pos])
                            for pos in self.flex_pos])
            flexleft = self.n_roster_per_team['FLEX'] - flexused
            if flexleft > 0:
                print 'FLEX: {}'.format(flexleft)
                pos_totals['FLEX'] = pos_totals['FLEX'] + flexleft
        if sum(pos_totals.values()) > 0:
            print '\ntotal open starting roster spots ({} picks):'.format(2*len(comp_mans))
            for pos,n in pos_totals.items():
                if n > 0:
                    print '{}: {}'.format(pos, n)
                

    def do_pick(self, args):
        """
        usage: pick I...
               pick vols  (in snake draft mode)
        remove player(s) with index(ces) I from available player list
        in snake draft mode, `pick vols` can be used to pick the VOLS recommended player.
        """
        # it may not be worth supporting removing multiple indices at once.
        manager = self._get_current_manager()
        indices = []
        if self.draft_mode and args.lower() == 'vols':
            indices = [self.pick_vols(manager)]
        try:
            if not indices:
                indices = [int(i) for i in args.split(' ') if i]
        except ValueError as e:
            print '`pick` requires integer indices.'
            print e
        if self.draft_mode and len(indices) > 1:
            print 'Picking multiple indices at once is not supported in draft mode.'
            return
        for i in indices:
            try:
                pickno = self.i_manager_turn + 1 if self.draft_mode else None
                pop_from_player_list(i, self.ap, self.pp, manager=manager, pickno=pickno)
                self._update_vorp()
                if self.draft_mode:
                    self._advance_snake()
            except IndexError as e:
                print e
                print 'could not pick player from list.'

    def do_pop(self, args):
        """alias for `pick`"""
        self.do_pick(args)

    def do_q(self, args):
        """alias for `quit`"""
        self.do_quit(args)

    def do_quit(self, _):
        """
        exit the program
        """
        verify_and_quit()

    def do_roster(self, args):
        """
        usage: roster [N]...
               roster all
        prints the roster of the current manager so far
        can take a number or series of numbers to print only those manager's
        if "all" is passed then it will output all rosters
        """
        if not self.draft_mode:
            print 'The `roster` command is only available in draft mode.'
            return
        if args.lower() == 'all':
            for i_man in range(1, 1+self.n_teams):
                manname = self._get_manager_name(i_man)
                print '\n {}:'.format(manname)
                theroster = self._get_manager_roster(i_man)
                if len(theroster) > 0:
                    print theroster
                else:
                    print 'No players on this team yet.\n'
            print
            return
        if not args:
            print '\n {}:'.format( self._get_manager_name() )
            theroster = self._get_manager_roster(self._get_current_manager())
            if len(theroster) > 0:
                print theroster
                print
            else:
                print 'No players on this team yet.\n'
            return
        indices = []
        try:
            indices = [int(i) for i in args.split(' ')]
        except ValueError as e:
            print '`roster` requires integer arguments'
            print e
            return
        for i in indices:
            manname = self._get_manager_name(i)
            print '\n {}:'.format(manname)
            theroster = self._get_manager_roster(i)
            if len(theroster) > 0:
                print theroster
            else:
                print 'No players on this team yet.'
        print # newline

    def do_save(self, args):
        """
        usage: save [OUTPUT]
        saves player lists to OUTPUT.csv (default OUTPUT is draft_players)
        """
        outname = args if args else 'draft_players'
        save_player_list(outname, self.ap, self.pp)

    def do_snake(self, args):
        """
        usage: snake [N]
        initiate snake draft mode, with the user in draft position N
        """
        if self.draft_mode:
            print 'You are already in draft mode!'
            return
        if len(self.pp) > 0:
            print 'There are already picked players. This is not starting a draft from scratch.'
            print 'It is recommended you just quit and start fresh. Draft command will be canceled.'
            return
        numprompt = 'Enter your position in the snake draft [1,...,{}]: '.format(self.n_teams)
        numstr = args if args else raw_input(numprompt)
        try:
            self.user_manager = int(numstr)
            if self.user_manager not in range(1,self.n_teams+1):
                raise ValueError('Argument not in range.')
        except ValueError as e:
            print e
            print 'Could not cast argument to draft.'
            print 'Use a single number from 1 to {}'.format(self.n_teams)        
            return
        # perhaps there is a proxy we can use for this to reduce the number of variables
        self.draft_mode = True
        n_rounds = sum([self.n_roster_per_team[pos] for pos in self.n_roster_per_team])
        print '{} rounds of drafting commencing.'.format(n_rounds)
        self.manager_picks = []
        for i in range(n_rounds):
            if i % 2 == 0:
                self.manager_picks.extend(range(1,self.n_teams+1))
            else:
                self.manager_picks.extend(range(self.n_teams,0,-1))
        self.i_manager_turn = 0
        self._set_prompt()

    def do_sort(self, args):
        """choose a stat to sort by (unimplemented - will need to add member to class for consistency)"""
        return

    def do_team(self, args):
        """
        usage: team [TEAM]
        lists players by TEAM abbreviation.
        if no argument is provided, lists the available teams.
        """
        if args:
            find_by_team(args, self.ap, self.pp)
        else:
            print_teams(self.ap, self.pp)

    def do_unpick(self, args):
        """
        usage: unpick
        moves player(s) with index(ces) I from picked list to available.
        if no index is provided, then the last player picked will be returned.
        """
        # we used to allow indices, and multiple indices, but this gets too complicated w/ draft mode.
        if len(self.pp) > 0:
            lasti = self.pp.index[-1]
            try:
                push_to_player_list(lasti, self.ap, self.pp)
                self._update_vorp()
                if self.draft_mode:
                    self._regress_snake()
            except IndexError as e:
                print e
                print 'could not put player ({}) back in available list.'.format(lasti)
        else:
            print 'No players have been picked.'

    def do_unpop(self, args):
        """alias for unpick"""
        self.do_unpick(args)

    def do_vona(self, args):
        """print out VONA for each position, probably assuming VOLS strategy for others"""

        return
        
    def do_test_vols(self, args):
        """quick test for pick_vols"""
        manager = self._get_current_manager()
        pick = self.pick_vols(manager)
        # print self.ap.loc[pick] # instead of printing the whole player, print the line of the df:
        print ' VOLS recommended:'
        print self.ap[self.ap.index == pick]
        print
                
    def pick_vols(self, manager):
        """
        picks the player with the highest value over lowest starter in that position
        returns the index of that player? (should be able to get everything else from self.ap.loc[index])
        """
        if not self.draft_mode:
            print 'WARNING: i think you will need to be in draft mode for this, or manager will not be defined'
        roster = self._get_manager_roster(manager)
        total_roster_spots = sum([self.n_roster_per_team[pos] for pos in self.n_roster_per_team])
        if len(roster) >= total_roster_spots:
            manname = self._get_manager_name()
            print '{}\'s roster has no available spots left'.format(manname)
            return None
        
        starting_roster_spots = sum([self.n_roster_per_team[pos]
                                     for pos in self.n_roster_per_team
                                     if pos.upper() is not 'BENCH'])
        crap_positions = ['K'] # add DST when (or if) we bother
        crap_starting_roster_spots = sum([self.n_roster_per_team[pos] for pos in crap_positions])
        # key_starting_roster_spots = starting_roster_spots - crap_starting_roster_spots

        key_positions = ['QB', 'RB', 'WR', 'TE'] # this concept includes FLEX so don't count it
        # realistically this will just be QBs but let's keep it flexible
        key_nonflex_positions = [pos for pos in key_positions if pos not in self.flex_pos]
        needed_key_starter_positions = []
        needed_key_starter_positions.extend([pos for pos in key_nonflex_positions
                                             if len(roster[roster.position == pos])
                                             < self.n_roster_per_team[pos]])
        # print [len(roster[roster.position == pos])
        #        > self.n_roster_per_team[pos] for pos in self.flex_pos]
        used_flex_spot = any([len(roster[roster.position == pos]) > self.n_roster_per_team[pos]
                                      for pos in self.flex_pos])
        flex_mult = 0 if used_flex_spot else 1
        print 'allowing flex: ', flex_mult
        needed_key_starter_positions.extend([pos for pos in self.flex_pos
                                             if len(roster[roster.position == pos])
                                             < self.n_roster_per_team[pos]
                                             + flex_mult*self.n_roster_per_team['FLEX']])
        # needed_starter_backups = [...] #work on getting starter limit +1 of everything. then an extra flex
        print 'needed key starter positions:', needed_key_starter_positions
        
        if needed_key_starter_positions:
            topstart = self.ap[self.ap.position.isin(needed_key_starter_positions)].sort_values('vols', ascending=False)
            # player = topstart.iloc[0] # this is the player
            player_index = topstart.index[0]
            return player_index
        elif len(roster) < starting_roster_spots - crap_starting_roster_spots:
            print 'LITERALLY A DISASTER'
            ## make sure to grab one level of backup for each position before getting more
            pass
            # fill up bench with more starters
        else:
            print 'YEAH YOU WILL PROLLY NEED TO DEBUG IN HERE'
            needed_crap_positions = [pos for pos in crap_positions
                                     if len(roster[roster.position == pos])
                                     < self.n_roster_per_team[pos]]
            print needed_crap_positions
            topcrap = self.ap[self.ap.position.isin(needed_crap_positions)].sort_values('vols', ascending=False)
            player = topcrap.iloc[0] # out of bounds error, could happen if people grab kickers too soon. need to check for it
            print player
            player_index = topcrap.index[0]
            print player_index
            return player_index

        return


                
def main():
    """main function that runs upon execution"""
    ## use argument parser
    parser = argparse.ArgumentParser(description='Script to aid in real-time fantasy draft')
    parser.add_argument('--ruleset', type=str, choices=['phys', 'dude', 'bro'], default='bro', help='which ruleset to use of the leagues I am in')
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
        # ignoring defense for the draft
        #'DST':args.n_dst,
        'K':args.n_k,
        'BENCH':args.n_bench
    }
    n_roster_per_league = {}
    for pos, nper in n_roster_per_team.items():
        n_roster_per_league[pos] = nper * n_teams

    # in principle FLEX can be defined in a different way,
    # so we'll leave this definition local so that we might change it later.
    flex_pos = ['RB', 'WR', 'TE']

    if args.ruleset == 'phys':
        rules = phys_league
    if args.ruleset == 'dude':
        rules = dude_league
    if args.ruleset == 'bro':
        rules = bro_league

    print 'Initializing with ruleset:'
    # print some output to verify the ruleset we are working with
    rulestr = '  {} team, {} PPR'.format(n_teams, rules.ppREC)
    if rules.ppPC != 0 or rules.ppINC != 0:
        rulestr += ', {}/{} PPC/I'.format(rules.ppPC, rules.ppINC)
    print rulestr
    rosterstr = ' '
    for pos in ['QB', 'RB', 'WR', 'TE', 'FLEX']: # there's always just 1 DST and K, right?
        nper = n_roster_per_team[pos]
        rosterstr += ' {}{} /'.format(nper, pos)
    print rosterstr[:-2]
    
    main_positions = ['QB', 'RB', 'WR', 'TE', 'K']

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
    availdf['projection'] = get_points_from_data_frame(rules, availdf)
    # can go ahead and filter out stats once we have projections
    availdf = availdf[['name', 'team', 'position', 'projection']]    

    # label nominal (non-flex) starters by their class
    for pos in main_positions:
        # sort the players in each position so we can grab the top indices
        availpos = availdf.loc[availdf.position == pos, :].sort_values('projection', ascending=False)
        for i_class in range(n_roster_per_team[pos]):
            ia, ib = i_class*n_teams, (i_class+1)*n_teams
            itoppos = availpos.index[ia:ib]
            icls = availdf.index.isin(itoppos)
            availdf.loc[icls, 'level'] = '{}{}'.format(pos, i_class+1)
    availflex = availdf.loc[(availdf.position.isin(flex_pos)) & (availdf['level'].isnull()), :].sort_values('projection', ascending=False)
    for i_class in range(n_roster_per_team['FLEX']):
        ia, ib = i_class*n_teams, (i_class+1)*n_teams
        itoppos = availflex.index[ia:ib]
        icls = availdf.index.isin(itoppos)
        availdf.loc[icls, 'level'] = 'FLEX{}'.format(i_class+1)

    # players that have been assigned a class so far are starters
    # use this to find the worst value of each starter and subtract it
    #  from the projection to get the "VOLS" (value over last starter).
    # this is just a static calculation right now.
    # in the future we could adjust this for draft position and dynamically
    #  update in the case of other teams making "mistakes".
    starter_mask = availdf['level'].notnull()
    starterdf = availdf.loc[starter_mask]
    for pos in main_positions:
        worst_starter_value = starterdf[starterdf.position == pos]['projection'].min()
        availdf.loc[availdf.position == pos, 'vols'] = availdf['projection'] - worst_starter_value

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
        pos_nsdf = nonstarterdf.loc[nonstarterdf.position == pos].sort_values('projection', ascending=False)
        ipos = pos_nsdf.index[:n_pos_bench]
        ibnch_mask = availdf.index.isin(ipos)
        availdf.loc[ibnch_mask, 'level'] = 'BU'

    # now we've given the backups a class, the worst projection at each position is the worst bench value.
    # we will define this as the VOLB (value over replacement player)
    # this is also a static calculation right now, but in principle it could be dynamically updated like VOLS. this might be a lot of re-computation and/or more complicated code.
    # doing this here instead of trying to grab the value from the loop above is less-than-optimized, but is more vulnerable to programmer error and edge cases.
    draftable_mask = availdf.level.notnull()
    draftable_df = availdf.loc[draftable_mask]
    for pos in main_positions:
        worst_draftable_value = draftable_df[draftable_df.position == pos]['projection'].min()
        availdf.loc[availdf.position == pos, 'volb'] = availdf['projection'] - worst_draftable_value

    ## now label remaining players as waiver wire material
    availdf.loc[availdf.level.isnull(), 'level'] = 'WAIV'

    ## TODO: should think about defining an additional, dynamic quantity which interpolates somehow between VOLS and VOLB as the draft goes on. call this VORP?

    ## finally sort by our stat of choice for display
    availdf = availdf.sort_values('vols', ascending=False)
    availdf.reset_index(drop=True, inplace=True) # will re-number our list to sort by vols
    
    # make an empty dataframe with these reduces columns to store the picked players
    # this might be better as another level index in the dataframe, or simply as an additional variable in the dataframe.
    # In the latter case we'd need to explicitly exclude it from print statements.
    pickdf = pd.DataFrame(columns=availdf.columns)

    prompt = MainPrompt()
    prompt.ap = availdf
    prompt.pp = pickdf
    prompt.n_teams = n_teams
    prompt.n_roster_per_team = n_roster_per_team
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
        save_player_list(backup_fname, prompt.ap, prompt.pp)
        raise
        
        
if __name__ == '__main__':
    main()
