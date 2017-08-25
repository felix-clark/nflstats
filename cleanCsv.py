#!/usr/bin/env python
import argparse

def getElems(line):
    # remove '"' and whitespace from each end of every element
    splitline = (line.strip(' \n\r"').split('","'))
    # remove commas which are placed into strings representing 4+ digit numbers
    return [e.replace(',','') for e in splitline]

def replaceTitleLine(line):
    """change names from those used online to those used in our code (e.g. 'INTS' to 'passing_ints')"""
    line = line.replace('"Player"', 'name')
    line = line.replace('"Team"', 'team')
    # the QB pattern is unambiguous, with CMP and INTS
    line = line.replace('"ATT","CMP","YDS","TDS","INTS"', 'passing_att,passing_cmp,passing_yds,passing_tds,passing_ints')
    # then check the receiver one since it has REC
    line = line.replace('"REC","YDS","TDS"', 'receiving_rec,receiving_yds,receiving_tds')
    # the rushing one is ambiguous on its own, so we have to fix it last
    line = line.replace('"ATT","YDS","TDS"', 'rushing_att,rushing_yds,rushing_tds')
    # then kickers
    line = line.replace('"FG","FGA","XPT"', 'kicking_fgm,kicking_fga,kicking_xpmade')
    # at some point we'll add defense. if we even care about checking those stats. probably not.
    line = line.replace('"FL"', 'fumbles_lost')
    # more of a stylistic choice to replace this
    line = line.replace('"FPTS"', 'fp_projection')
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
