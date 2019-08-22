# nflstats

## Description

statistical analysis of NFL players for fantasy football

bayesian models are implemented to update player predictions week-to-week (not done or optimized yet)

there is also a command-line tool to interactively track players as a draft is occuring. It has a few different metrics for evaluating draft value.

no instructions right now because if you're in one of my leagues this I don't want to make it too easy for you (especially you, Russell).

## Requirements

Dependencies are now handled with pipenv. It is recommended to use pyenv to manage virtual environment. Install pipenv with pip, then `pipenv install` in this directory.

Matplotlib has some requirements when being installed in a virtual environment. See the following link for instructions on how to install prerequisites for the GTK backend.

https://matplotlib.org/3.1.1/faq/virtualenv_faq.html

## Instructions

To scrape for historic data, run `get_yearly_stats.py`. This will enable some of the other analyses.

other scripts to get data are `get_weekly_stats.py` and `get_draft_data.py`. while there is some attempt at automatically scraping and caching data when needed, these may need to be run manually as the process is not robust.

`draft_app.py` can be used to provide useful metrics while drafting. it updates a file `draft_board.html` that can be opened in a browser, ideally with an auto-updating feature.
