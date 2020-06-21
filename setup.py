from glob import glob

from setuptools import find_packages, setup

setup(
    name="nflstats",
    version="0.4.0",
    packages=find_packages(),
    scripts=glob("scripts/*.py"),
)
