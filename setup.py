import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "paper-rl",
    version = "0.0.1",
    author = "Stone Tao",
    description = "An RL library that might be cutting edge",
    license = "MIT",
    keywords = "reinforcement-learning machine-learning ai",
    url = "http://packages.python.org/paper-rl",
    packages=['paper_rl', 'tests'],
    long_description=read('README.md'),
)