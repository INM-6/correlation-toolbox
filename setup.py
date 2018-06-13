# encoding: utf8
from setuptools import setup

setup(
    name='correlation-toolbox',
    version='0.0.1',
    author='Jakob Jordan, David Dahmen, Hannah Bos, Maximilian Schmidt',
    author_email='j.jordan@fz-juelich.de',
    description=('Collection of functions to investigate correlations in '
                 'spike trains and membrane potentials.'),
    license='MIT',
    url='https://github.com/INM-6/correlation-toolbox',
    python_requires='>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*, !=3.4.*, <4',
    install_requires=['future', 'quantities'],
    packages=['correlation_toolbox'],
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Utilities',
    ],
)
