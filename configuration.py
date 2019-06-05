# -*- coding: utf-8 -*-
"""configuration.py: Config file for Lagrangian_CSET. Set projectwide parameters and file locations."""

__author__ = "Johannes Mohrmann"
__version__ = "0.1"


import os

# data locations
CSET_data_dir =       r'/home/disk/eos4/jkcm/Data/CSET'
trajectory_dir =      os.path.join(CSET_data_dir, 'Lagrangian_project', 'Trajectories')
GOES_netcdf_dir =     os.path.join(CSET_data_dir, 'GOES/VISST_pixel')
dropsonde_dir =       os.path.join(CSET_data_dir, 'AVAPS/NETCDF')
HYSPLIT_working_dir = '/home/disk/eos4/jkcm/Data/HYSPLIT/working'  # storing CONTROL
HYSPLIT_source_dir =  '/home/disk/eos4/jkcm/Data/HYSPLIT/source'

# system calls
HYSPLIT_call =        '/home/disk/p/jkcm/hysplit/trunk/exec/hyts_std'  # to run HYSPLIT

# geoconfigs
latlon_range = {'lat': (15, 50), 'lon': (-160, -110)}