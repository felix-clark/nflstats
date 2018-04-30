#!/usr/bin/env python
import argparse

def getElems(line):
    # remove '"' and whitespace from each end of every element
    # splitline = (line.strip(' \n\r"').split('","'))
    splitline = line.split('","')
    # remove commas which are placed into strings representing 4+ digit numbers
    number_commas_removed = [e.replace(',','').strip(' \n\r"') for e in splitline]
    return number_commas_removed

def replaceTitleLine(line):
    """change names from those used online to those used in our code (e.g. 'INTS' to 'passing_int')"""
    # check https://github.com/BurntSushi/nfldb/wiki/The-data-model for list of full possible stats
    line = line.replace('"Player"', 'name')
    line = line.replace('"Team"', 'team')
    # the QB pattern is unambiguous, with CMP and INTS
    line = line.replace('"ATT","CMP","YDS","TDS","INTS"', 'passing_att,passing_cmp,passing_yds,passing_td,passing_int')
    # then check the receiver one since it has REC
    line = line.replace('"REC","YDS","TDS"', 'receiving_rec,receiving_yds,receiving_td')
    # the rushing one is ambiguous on its own, so we have to fix it last
    line = line.replace('"ATT","YDS","TDS"', 'rushing_att,rushing_yds,rushing_td')
    # then kickers
    line = line.replace('"FG","FGA","XPT"', 'kicking_fgm,kicking_fga,kicking_xpmade')
    line = line.replace('"FL"', 'fumbles_lost')
    # defense stats
    line = line.replace('"SACK","INT","FR","FF","TD","ASSIST","SAFETY","PA","YDS_AGN",',
                        'defense_sk,defense_int,defense_frec,defense_ffum,defense_td,defense_ast,defense_safe,defense_pa,defense_lost_yds,')
    # more of a stylistic choice to replace this
    line = line.replace('"FPTS"', 'fp_projection')

    # add options for ECP files (which include ADP). We'll use the half-PPR rankings.
    line = line.replace('"Rank"', 'rank')
    line = line.replace('"Bye"', 'bye')
    line = line.replace('"Pos"', 'pos')
    line = line.replace('"Best","Worst","Avg","Std Dev","ADP","vs. ADP"', 'best,worst,ecp,ecp_std_dev,adp,ecp_vs_adp')
    line = line.replace('"Overall (Team)"', 'name,team')

    return line
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='transform .csv files scraped in a certain weird format and turn them into a reasonable one that can be easily read as a DataFrame')
    parser.add_argument('input',type=str,help='name of .csv to clean')
    parser.add_argument('-o','--output',type=str,default='',help='name of output .csv')

    args = parser.parse_args()

    infile = open(args.input, 'r')
    outfile = open(args.output, 'w') if args.output else None
    for i_line,line in enumerate(infile):
        if not any([c.isalnum() for c in line]):
            continue # strip out garbage lines
        if i_line == 0:
            # in the 1st (title) line, transform the keys into our convention.
            line = replaceTitleLine(line)
        else:
            elems = getElems(line)
            if len(elems) <= 1:
                print 'This does not appear to be the type of .csv for which this script is intended.'
                print elems
                exit(1)
            # rejoin with commas
            line = ','.join( elems )
        if not outfile:
            print line
        else:
            outfile.write(line+'\n')

    infile.close()
    if outfile: outfile.close()
