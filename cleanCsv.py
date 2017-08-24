#!/usr/bin/env python
import argparse

def getElems(line):
    # remove '"' and whitespace from each end of every element
    splitline = (line.strip(' \n"').split('","'))
    # remove commas which are placed into strings representing 4+ digit numbers
    return [e.replace(',','') for e in splitline]

replacement_dict = {
    'INTS':'passing_ints'
}
def replaceTitle(name):
    """change names from those used online to those used in our code (e.g. 'INTS' to 'passing_ints')"""
    try:
        name = replacement_dict[name]
    except KeyError:
        pass
    return name
    

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='transform .csv files scraped in a certain weird format and turn them into a reasonable one that can be easily read as a DataFrame')
    parser.add_argument('input',type=str,help='name of .csv to clean')
    parser.add_argument('-o','--output',type=str,default='',help='name of output .csv')

    args = parser.parse_args()

    infile = open(args.input, 'r')
    outfile = open(args.output, 'w') if args.output else None
    for i_line,line in enumerate(infile):
        print i_line
        elems = getElems(line)
        if len(elems) <= 1:
            print 'This does not appear to be the type of .csv for which this script is intended.'
            exit(1)
        if i_line == 0:
            # in the 1st (title) line, transform the keys into our convention.
            elems = replaceTitles(elems)
        # rejoin with commas
        modline = ','.join( elems )
        if not outfile:
            print modline
        else:
            outfile.write(modline+'\n')

    infile.close()
    if outfile: outfile.close()
