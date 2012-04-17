import os
from setuptools import setup

# Utility function to read the README file.  
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "rbm.py",
    version = "1.0",
    author = "Yann N. Dauphin",
    author_email = "dhaemon@gmail.com",
    description = ("Pain-free Restricted Boltzmann Machines."),
    license = "BSD",
    keywords = "machine learning restricted boltzmann machine",
    url = "http://ynd.github.com/rbm.py/",
    packages=['rbm'],
    long_description=read('README'),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: BSD License",
    ],
)
