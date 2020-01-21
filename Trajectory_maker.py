# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 12:21:49 2016
Code for making all the many many trajectories for CSET Lagrangian analysis
@author: jkcm
"""
import numpy as np
import matplotlib.pyplot as plt
import utils
import datetime as dt
from LoopTimer import LoopTimer as lt
from subprocess import call
import os

hours = 72
init_height = 500
vertical_type = 1


def run_trajectories(run_date, location_list):

    hysplit_file_list = utils.get_hysplit_files(run_date=run_date, run_hours=hours)

    utils.write_control_file(start_time=run_date,
                             coords=location_list,
                             hyfile_list=hysplit_file_list,
                             hours=hours,
                             vertical_type=vertical_type,
                             init_height=init_height,
                             tdumpdir=utils.trajectory_dir)
    print 'calling HYSPLIT'
    call(utils.HYSPLIT_call, shell=False, cwd=utils.HYSPLIT_workdir)


if __name__ == "__main__":
    grid_coords = utils.gridder([27, -130], [36, -137], [41, -128], [32, -121],
                                numlats=5, numlons=6)
#    utils.plot_gridpoints(grid_coords)
    date_list = [dt.datetime(2015, 7, 01) + dt.timedelta(x) for x in range(0, 60)]
    loop = lt(len(date_list))

    fig, ax, m_ax = utils.make_map_plot()

#    date_list = [dt.datetime(2015, 8, 19) + dt.timedelta(x) for x in range(0, 2)]

    for i, date in enumerate(date_list):
        loop.update(i)
        if False:
            run_trajectories(date, grid_coords)
        if True:
            t_name = os.path.join(utils.trajectory_dir,
                                  'tdump'+date.strftime('%Y%m%dH%H%M'))
            print t_name
            utils.add_tdump_to_plot(m_ax, t_name)

    ax.set_title('CSET grand trajectory set, {} hours, {}m'.format(hours, init_height),
                 y=1.08)
    outfile = r'/home/disk/eos4/jkcm/Data/CSET/Lagrangian_project/all_trajectories.png'
    fig.savefig(outfile, dpi=300, bbox_inches='tight')