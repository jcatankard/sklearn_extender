from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, 'README.md'), encoding='utf-8') as fh:
    long_description = '\n' + fh.read()

VERSION = '0.0.1'
DESCRIPTION = 'add useful functionality to sci-kit learn'

# Setting up
setup(
    name='sklearn_extender',
    version=VERSION,
    author='https://github.com/jcatankard',
    description=DESCRIPTION,
    long_description_content_type='text/markdown',
    long_description=long_description,
    packages=find_packages(),
    install_requires=['numpy', 'matplotlib', 'sklearn'],
    setup_requires=['wheel'],
    keywords=[],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License'
    ]
)