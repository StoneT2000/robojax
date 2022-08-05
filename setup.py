import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "robojax",
    version = "0.0.1",
    author = "Stone Tao",
    description = "An RL library in Jax built for robotic learning",
    license = "MIT",
    keywords = ["reinforcement-learning", "machine-learning", "ai"],
    url = "http://packages.python.org/robojax",
    packages=['robojax', 'tests'],
    long_description=read('README.md'),
)