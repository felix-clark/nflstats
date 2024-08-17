#!/usr/bin/env python
import argparse

# We don't have to deal with the comma issue in ADP/ECP it seems


def getElems(line: str) -> list[str]:
    # remove '"' and whitespace from each end of every element
    # splitline = (line.strip(' \n\r"').split('","'))
    splitline = line.split(",")
    # remove commas which are placed into strings representing 4+ digit numbers
    number_commas_removed = [e.replace(",", "").strip(' \n\r"') for e in splitline]
    return number_commas_removed


def replaceTitleLine(line: str) -> str:
    """change names from those used online to those used in the PFR tables (e.g. 'INTS' to 'pass_int')"""
    # check https://github.com/BurntSushi/nfldb/wiki/The-data-model for list of full possible stats
    line = line.replace('"Player"', "player")
    line = line.replace('"Team"', "team")
    # the QB pattern is unambiguous, with CMP and INTS
    line = line.replace(
        '"ATT","CMP","YDS","TDS","INTS"', "pass_att,pass_cmp,pass_yds,pass_td,pass_int"
    )
    # then check the receiver one since it has REC
    line = line.replace('"REC","YDS","TDS"', "rec,rec_yds,rec_td")
    # the rushing one is ambiguous on its own, so we have to fix it last
    line = line.replace('"ATT","YDS","TDS"', "rush_att,rush_yds,rush_td")
    # then kickers
    # line = line.replace('"FG","FGA","XPT"', 'kick_fgm,kick_fga,kick_xpmade')
    line = line.replace('"FG","FGA","XPT"', "fgm,fga,xpm")
    line = line.replace('"FL"', "fumbles_lost")
    # defense stats
    line = line.replace(
        '"SACK","INT","FR","FF","TD"',
        "defense_sk,defense_int,defense_frec,defense_ffum,defense_td",
    )
    line = line.replace('"ASSIST"', "defense_ast")
    line = line.replace(
        '"SAFETY","PA","YDS_AGN",', "defense_safe,defense_pa,defense_lost_yds,"
    )

    # more of a stylistic choice to replace this
    line = line.replace('"FPTS"', "fp_projection")

    # # add options for ECP files (which include ADP). We'll use the half-PPR rankings.
    # line = line.replace('"Rank"', 'rank')
    # line = line.replace('"Bye"', 'bye')
    # line = line.replace('"Pos"', 'pos')
    # line = line.replace('"Best","Worst","Avg","Std Dev","ADP","vs. ADP"', 'best,worst,ecp,ecp_std_dev,adp,ecp_vs_adp')
    # # line = line.replace('"Overall (Team)"', 'name,team') # old format
    # line = line.replace('"Overall"', 'player')
    # line = line.replace('"Team"', 'team')
    # # line = line.replace(',"WSID",', ',')

    # The ADP and ECP files are split now
    # It's a bit tricker to infer from the field names alone, so replace the whole line now.
    # ADP (...<year>_Overall_ADP_Rankings.csv)
    # This was form 2023:
    # line = line.replace(
    #     '"Rank",player,team,"Bye","POS","ESPN","Sleeper","NFL","RTSports","FFC","AVG"',
    #     "rank,player,team,bye,pos,espn_adp,sleeper_adp,nfl_adp,rtsports_adp,ffc_adp,adp",
    # )
    # in 2024 it looks like this:
    line = line.replace(
        '"Rank",player,team,"Bye","POS","Yahoo","Sleeper","RTSports","AVG"',
        "rank,player,team,bye,pos,yahoo_adp,sleeper_adp,rtsports_adp,adp",
    )
    # ECP (...<year>_Draft_ALL_Rankings.csv)
    line = line.replace(
        '"RK",TIERS,"PLAYER NAME",TEAM,"POS","BYE WEEK","SOS SEASON","ECR VS. ADP"',
        "ecp,ecp_tier,player,team,pos,bye,sos,ecp_vs_adp",
    )

    return line


def main():
    parser = argparse.ArgumentParser(
        description="transform .csv files scraped in a certain weird format and turn them into a reasonable one that can be easily read as a DataFrame"
    )
    parser.add_argument("input", type=str, help="name of .csv to clean")
    parser.add_argument(
        "-o", "--output", type=str, default="", help="name of output .csv"
    )

    args = parser.parse_args()

    infile = open(args.input, "r")
    outfile = open(args.output, "w") if args.output else None
    for i_line, line in enumerate(infile):
        if not any([c.isalnum() for c in line]):
            continue  # strip out garbage lines
        if i_line == 0:
            # in the 1st (title) line, transform the keys into our convention.
            line = replaceTitleLine(line)
        else:
            elems = getElems(line)
            if len(elems) <= 1:
                print(
                    "This does not appear to be the type of .csv for which this script is intended."
                )
                print(elems)
                exit(1)
            # rejoin with commas
            line = ",".join(elems)
            # some empty fields may need removing (like WSID in ecp/adp)
            # line = line.replace(',,', ',') # actually fine to just have empty field
        if not outfile:
            print(line)
        else:
            outfile.write(line + "\n")

    infile.close()
    if outfile:
        outfile.close()


if __name__ == "__main__":
    main()
