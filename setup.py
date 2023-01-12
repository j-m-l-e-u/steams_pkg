from setuptools import setup
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

library_folder = os.path.dirname(os.path.realpath(__file__))

requirementPath = f'{library_folder}/requirements.txt'
install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

setup(
    name='steams',
    version='0.17',
    author="Jean-Marie Lepioufle",
    author_email="jm@jeanmarie.eu",
    packages=[
        'steams',
        'steams.data',
        "steams.models",
        "steams.tepe",
        "steams.utils"],
    license='MIT + Copyright NILU',
    description='Space-time prediction with sparse and irregular space-time multi-timeserie.',
    long_description = long_description,
    url="https://git.nilu.no/aqdl/steams_pkg",
    python_requires='>=3.8',
    install_requires=install_requires,
    extras_require={
        "examples": ["ipython", "jupyter", "os", "matplotlib"],
    })
