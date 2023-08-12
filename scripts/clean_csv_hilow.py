#!/usr/bin/env python3
"""
Similar to clean_csv.py, but operates on CSVs with high and low projections also included.
"""
import argparse
import logging


def get_line_elems(line):
    # remove '"' and whitespace from each end of every element
    # splitline = (line.strip(' \n\r"').split('","'))
    splitline = line.split('","')
    # remove commas which are placed into strings representing 4+ digit numbers
    number_commas_removed = [e.replace(',', '').strip(' \n\r"') for e in splitline]
    return number_commas_removed


def replace_title_line(line):
    """
    change names from those used online to those used in the PFR tables (e.g. 'INTS' to 'pass_int')
    """
    # check https://github.com/BurntSushi/nfldb/wiki/The-data-model for list of full possible stats
    line = line.replace('"Player"', 'player')
    line = line.replace('"Team"', 'team')
    # the QB pattern is unambiguous, with CMP and INTS
    line = line.replace('"ATT","CMP","YDS","TDS","INTS"',
                        'pass_att,pass_cmp,pass_yds,pass_td,pass_int')
    # then check the receiver one since it has REC
    line = line.replace('"REC","YDS","TDS"', 'rec,rec_yds,rec_td')
    # the rushing one is ambiguous on its own, so we have to fix it last
    line = line.replace('"ATT","YDS","TDS"', 'rush_att,rush_yds,rush_td')
    # then kickers
    line = line.replace('"FG","FGA","XPT"', 'fgm,fga,xpm')
    line = line.replace('"FL"', 'fumbles_lost')
    # defense stats
    line = line.replace('"SACK","INT","FR","FF","TD"',
                        'defense_sk,defense_int,defense_frec,defense_ffum,defense_td')
    line = line.replace('"ASSIST"',
                        'defense_ast')
    line = line.replace('"SAFETY","PA","YDS_AGN",',
                        'defense_safe,defense_pa,defense_lost_yds,')

    # more of a stylistic choice to replace this
    line = line.replace('"FPTS"', 'fp_projection')

    # add options for ECP files (which include ADP). We'll use the half-PPR rankings.
    line = line.replace('"Rank"', 'rank')
    line = line.replace('"Bye"', 'bye')
    line = line.replace('"Pos"', 'pos')
    line = line.replace('"POS"', 'pos')
    line = line.replace('"Best","Worst","Avg","Std Dev","ADP","vs. ADP"',
                        'best,worst,ecp,ecp_std_dev,adp,ecp_vs_adp')
    line = line.replace('"Overall"', 'player')
    line = line.replace('"Team"', 'team')
    # line = line.replace(',"WSID",', ',')

    return line


def get_type(elems):
    """
    returns 'ev' for the middle, 'high' for high and 'low' for low
    """
    if len(elems) < 3:
        logging.error('Error determining line type: %s', elems)
        raise RuntimeError('Invalid line')
    # for high/low lines, the first element is empty (hi and low take up the "team" field)
    if elems[0]:
        return 'ev'
    # type should be high or low
    linetype = elems[1]
    if linetype not in ['high', 'low']:
        logging.error('Unrecognized type: %s', linetype)
        raise RuntimeError('Invalid line')
    return linetype


def main():
    parser = argparse.ArgumentParser(
        description='transform downloaded CSV files with high/low lines into three output CSVs')
    parser.add_argument('input', type=str, help='name of .csv to clean')
    parser.add_argument('output', type=str, help='name of output .csv')

    args = parser.parse_args()

    infile = open(args.input, 'r')
    if '.csv' not in args.output:
        logging.error('Output file must have extension .csv')
        return 1

    outfile = open(args.output, 'w')
    outfile_high = open(args.output.replace('.csv', '_high.csv'), 'w')
    outfile_low = open(args.output.replace('.csv', '_low.csv'), 'w')
    # returns which file to write to based on the line type
    target_files = {
        'ev': outfile,
        'high': outfile_high,
        'low': outfile_low
    }
    outfiles = (outfile, outfile_high, outfile_low)

    it_infile = iter(infile)

    # process the header line first, writing it to every output
    line = next(it_infile)
    for fout in outfiles:
        fout.write(replace_title_line(line))

    # the high and low lines don't list the player, so we have to keep track of the last one
    player, team = None, None

    # loop through the remaining lines
    for line in it_infile:
        if not any([c.isalnum() for c in line]):
            continue  # strip out garbage lines
        elems = get_line_elems(line)
        linetype = get_type(elems)
        if linetype == 'ev':
            player, team = elems[:2]
        else:
            # the high and low lines need to use the player and team from the last ev line
            elems[0] = player
            elems[1] = team
        line = ','.join(elems)
        # some empty fields may need removing (like WSID in ecp/adp)
        # line = line.replace(',,', ',') # actually fine to just have empty field

        target_file = target_files[linetype]
        target_file.write(line+'\n')

    infile.close()
    for fout in outfiles:
        if fout:
            fout.close()


if __name__ == '__main__':
    main()
