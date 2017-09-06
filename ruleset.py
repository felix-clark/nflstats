from collections import namedtuple


points_per = [
    'ppPY', # points per passing yard
    'ppPY25', # points per 25 passing yards
    'ppPC',  # points per completion
    'ppINC',  # points per incompletion
    'ppPTD', # points per touchdown pass
    'ppPTD40', # points per 40+ yard TD pass
    'ppPTD50',  # points per 50+ yard TD pass
    'ppINT',  # points per interception
    'pp2PC', # points per 2pt passing conversion
    'ppP300', # bonus for 300-399 yard passing game
    'ppP400', # bonus for 400+ yard passing game
    'ppRY', # points per rushing yard
    'ppRY10', # points per 10 rushing yards
    'ppRTD',  # points per rushing touchdown
    'ppRTD40', # points per 40+ yard rushing TD
    'ppRTD50', # points per 50+ yard rushing TD
    'pp2PR', # points per 2pt rushing conversion
    'ppRY100', # bonus for 100-199 yard rushing game
    'ppRY200', # bonus for 200+ yard rushing game
    'ppREY', # points per receiving yard
    'ppREY10', # points per 10 receiving yards
    'ppREC', # points per reception
    'ppRETD', # points per receiving touchdown
    'ppRETD40', # points per 40+ yard receiving TD
    'ppRETD50', # points per 50+ yard receiving TD
    'pp2PRE', # points per 2pt receiving conversion
    'ppREY100', # bonus for 100-199 yard receiving game
    'ppREY200', # bonus for 200+ yard receiving game
    'ppFUML', # points per fumble lost
    'ppPAT', # points per PAT made
    'ppPATM', # points per PAT missed
    'ppFGM', # points per missed FG
    'ppFG0', # points for 0-39 yard FG
    'ppFG40', # points for 40-49 yard FG
    'ppFG50' # points for 50+ yard FG
]

# namedtuples are immutable
Ruleset = namedtuple('Ruleset', ' '.join(points_per) )
## set all point rewards to zero by default
Ruleset.__new__.__defaults__ = (0,) * len(Ruleset._fields)


bro_league = Ruleset(
    ppPY25=1,
    ppPC=1,
    ppINC=-1,
    ppPTD=4,
    ppPTD40=2,
    ppPTD50=1,
    ppINT=-2,
    pp2PC=2,
    ppRY10=1,
    ppRTD=6,
    ppRTD40=2,
    ppRTD50=1,
    pp2PR=2,
    ppREY10=1,
    ppREC=1,
    ppRETD=6,
    ppRETD40=2,
    ppRETD50=1,
    pp2PRE=2,
    ppFUML=-2,
    ppPAT=1,
    ppFGM=-1,
    ppFG0=3,
    ppFG40=4,
    ppFG50=5
)

phys_league = Ruleset(
    ppPY=0.04,
    ppPTD=4,
    ppINT=-2,
    pp2PC=2,
    ppP400=1,
    ppRY=0.1,
    ppRTD=6,
    pp2PR=2,
    ppRY100=1,
    ppRY200=2,
    ppREY=0.1,
    ppREC=0.5,
    ppRETD=6,
    pp2PRE=2,
    ppREY100=1,
    ppREY200=2,
    ppFUML=-2,
    ppPAT=1,
    ppFGM=-1,
    ppFG0=3,
    ppFG40=4,
    ppFG50=5
    )

dude_league = Ruleset(
    ppPY=0.04,
    ppPTD=4,
    ppINT=-2,
    pp2PC=2,
    ppRY=0.1,
    ppRTD=6,
    pp2PR=2,
    ppREY=0.1,
    ppREC=0.5,
    ppRETD=6,
    pp2PRE=2,
    ppFUML=-2,
    ppPAT=1,
    ppFGM=-1, # yahoo's point deduction depends on the distance
    ppFG0=2,
    # there is another 30-39 yd category here
    ppFG40=4,
    ppFG50=5
)

nycfc_league = Ruleset(
    ppPY=0.04,
    ppPC=0.2,
    ppINC=-0.2,
    ppPTD=4,
    # ppPTD50=1,
    ppINT=-2,
    pp2PC=2,
    ppP300=1,
    ppP400=2,
    ppRY=0.1,
    ppRTD=6,
    # ppRTD50=1,
    pp2PR=2,
    ppRY100=1,
    ppRY200=2,
    ppREY=0.1,
    ppREC=0.5,
    ppRETD=6,
    # ppRETD50=1,
    pp2PRE=2,
    ppREY100=1,
    ppREY200=2,
    ppFUML=-2,
    ppPAT=1,
    ppPATM=-1,
    ppFGM=-1,
    ppFG0=3,
    ppFG40=4,
    ppFG50=5
    )
