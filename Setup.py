# -*- coding: utf-8 -*-
"""
Created on Wed May 10 09:25:55 2023

@author: Mertcan
"""

import os
from setuptools import setup, find_packages
 
classifiers = [
  'Development Status :: 5 - Production/Stable',
  'Intended Audience :: Science/Research',
  'Intended Audience :: Developers',
  'Operating System :: Microsoft :: Windows :: Windows 10',
  'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
  'Programming Language :: Python :: 3',
  'Topic :: Scientific/Engineering :: GIS',
  'Topic :: Scientific/Engineering :: Information Analysis',
  'Topic :: Scientific/Engineering :: Image Recognition',
  'Topic :: Scientific/Engineering :: Image Processing',
]

with open('README.md') as f:
    long_description = f.read()

# only specify install_requires if not in RTD environment
if os.getenv("READTHEDOCS") == "True":
    INSTALL_REQUIRES = []
else:
    with open("requirements.txt") as f:
        INSTALL_REQUIRES = [line.strip() for line in f.readlines()]
        
setup(
  name='LasBuildSeg',
  version='0.0.10',
  description='Building Footrprint Extraction from Aerial LiDAR data',
  long_description=long_description,
  long_description_content_type='text/markdown',  
  url='',  
  author='MertcanErdem',
  author_email='merak908@gmail.com',
  license='GNU General Public License v2.0', 
  classifiers=classifiers,
  keywords=['Building', 'Lidar'], 
  packages=["LasBuildSeg"],
  install_requires=INSTALL_REQUIRES
)

