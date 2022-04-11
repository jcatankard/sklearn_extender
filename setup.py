from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, 'README.md'), encoding='utf-8') as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.1'
DESCRIPTION = 'add short description'

# Setting up
setup(
    name='sklearn_extender',
    version=VERSION,
    author='https://github.com/jcatankard',
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=['add package dependencies'],
    keywords=[],
    classifiers=[
        'Programming Language :: Python :: 3'
    ]
)