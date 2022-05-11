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

dev_requirements = []

setup(
    name='steams',
    version='0.02dev',
    author="Jean-Marie Lepioufle",
    author_email="jml@nilu.no",
    packages=[
        'steams',
        'steams.classes',
        "steams.dictionnary",
        "steams.krig",
        "steams.models",
        "steams.train_eval_pred",
        "steams.utils"],
    license='MIT + Copyright NILU',
    description='A package for testing space-time prediction with multi-timeserie located at sparse locations.',
    long_description = long_description,
    url="https://git.nilu.no/aqdl/steams",
    python_requires='>=3.8',
    install_requires=install_requires,
    extras_require={
        'dev': dev_requirements})
