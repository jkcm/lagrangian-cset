#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon July  30 2018

@author: jkcm
"""

import datetime as dt
from subprocess import check_output
import os

date_range = [dt.datetime(2015, 6, 20) + dt.timedelta(days=i) for i in range(73)]

for date in date_range:
    aer_file = r'/home/disk/eos4/jkcm/Data/CSET/MERRA/aer_Nv/svc_MERRA2_400.inst3_3d_aer_Nv.{:%Y%m%d}.nc4'.format(date)
    chm_file = r'/home/disk/eos4/jkcm/Data/CSET/MERRA/chm_Nv/MERRA2_400.inst3_3d_chm_Nv.{:%Y%m%d}.SUB.nc'.format(date)
    asm_file = r'/home/disk/eos4/jkcm/Data/CSET/MERRA/asm_Nv/MERRA2_400.inst3_3d_asm_Nv.{:%Y%m%d}.SUB.nc'.format(date)
    unified_file = r'/home/disk/eos4/jkcm/Data/CSET/MERRA/unified/MERRA.unified.{:%Y%m%d}.nc4'.format(date)
    asm_file_2 = r'/home/disk/eos4/jkcm/Data/CSET/MERRA/asm_NV_2/MERRA2_400.inst3_3d_asm_Nv.{:%Y%m%d}.SUB.nc'.format(date)
    
    new_save_location = r'/home/disk/eos4/jkcm/Data/CSET/MERRA/unified_2'
    
    unified_file_2 = os.path.join(new_save_location, 'MERRA.unified.{:%Y%m%d}.nc4'.format(date))
    
    for f in [aer_file, chm_file, asm_file]:
        if not os.path.exists(f):
            print(f)
    # cdo_call = 'cdo merge {} {} {} {}'.format(aer_file, chm_file, asm_file, unified_file)
    cdo_call = 'cdo merge {} {} {} {}'.format(asm_file_2, unified_file, unified_file_2)

    check_output(cdo_call, shell=True, cwd = '/home/disk/p/jkcm')

