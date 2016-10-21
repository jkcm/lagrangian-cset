# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 21:10:57 2016

@author: jkcm
Goal: filter out all bad trajectories, return a list of only the ones that
go the right way (i.e. southwest)

"""
#import matplotlib
#matplotlib.use('Qt4Agg')
import utils
import glob
import os
from math import atan2, degrees
import numpy as np
from scipy.io import savemat, loadmat
from matplotlib import cm


def angle(origin, point):
    """origin is (lat, lon), point is (lat, lon)
    """
    rise = point[0] - origin[0]
    run = point[1] - origin[1]
    angle = (90 - degrees(atan2(rise, run))) % 360
    return angle


def trajectory_24_hour_check(traj):
    """
    Calculates the vector between every successive 24-hour point on the
    trajectory. If it's north or east, discard
    """
    traj_24 = traj[traj.age % 24 == 0.0]
    for i in range(len(traj_24)-1):
        origin = (traj_24.lat[i], traj_24.lon[i])
        dest = (traj_24.lat[i+1], traj_24.lon[i+1])
        if angle(origin, dest) < 180 or angle(origin, dest) > 270:
            return False
    else:
        return True


def trajectory_12_hour_check(traj):
    """
    Calculates the vector between every successive 24-hour point on the
    trajectory. If it's north or east, discard
    """
    traj_12 = traj[traj.age % 12 == 0.0].append(traj.tail(1))
    for i in range(len(traj_12)-1):
        origin = (traj_12.lat[i], traj_12.lon[i])
        dest = (traj_12.lat[i+1], traj_12.lon[i+1])
        if angle(origin, dest) < 140 or angle(origin, dest) > 290:
            return False
    else:
        return True


def trajectory_direction_check(traj):
    """
    Calculates the vector from the trajectory origin to n, where n is the
    trajectory location every 12 hours (so there are 3 values for n in a
    48-hour trajectory). If any vector goes in the north or east directions,
    trajectory is not valid, return false.
    """
    traj_12 = traj[traj.age % 12 == 0.0]
    assert traj_12.age[0] == 0.0
    origin = (traj_12.lat.values[0], traj_12.lon.values[0])
    coords = zip(traj_12.lat[1:], traj_12.lon[1:])
    angles = np.array([angle(origin, i) for i in coords])
#    print angles
    if np.any(np.logical_or(angles > 270, angles < 180)):
        print "BAD TRAJ!"
        return False
    else:
        return True


def DEP_make_list_of_trajectories(testfunction, output=True):
    trajectory_folder = utils.trajectory_dir
    flist = glob.glob(os.path.join(trajectory_folder, 'tdump*'))

    good_traj, bad_traj = 0, 0

    save_dict = {}
    for f in flist:
        good_list = []
        tdump = utils.read_tdump(f)
        t_group = tdump.groupby('tnum')

        for key in t_group.groups.keys():
            traj = t_group.get_group(key)
            if testfunction(traj):
                good_list.append(key)
                good_traj += 1
            else:
                bad_traj += 1

        save_dict[os.path.basename(f)] = good_list
    if output:
        savemat('/home/disk/eos4/jkcm/Data/CSET/Lagrangian_project/good_trajectories_12.mat',
                {'good_trajectories': save_dict})
    return save_dict


def tdump_name_2_date(tname):
    tname = os.path.basename(tname)


def make_list_of_trajectories(testfunction, output=True):
    trajectory_folder = utils.trajectory_dir
    flist = glob.glob(os.path.join(trajectory_folder, 'tdump*'))

    good_traj, bad_traj = 0, 0

    save_dict = {}

    file_list = []
    good_traj_list = []
    for f in flist:
        tdump = utils.read_tdump(f)
        t_group = tdump.groupby('tnum')
        for key in t_group.groups.keys():
            traj = t_group.get_group(key)
            fname = "{}-{:02}-2deg.nc".format(os.path.basename(f)[5:13], key)
            file_list.append(fname)
            if testfunction(traj):
                save_dict[fname] = '1'
                good_traj_list.append(1)
                good_traj += 1
            else:
                bad_traj += 1
                save_dict[fname] = '0'
                good_traj_list.append(0)

    if output:
        file_list = np.array(file_list, dtype=object)
        savemat(output,
               {'file_list': file_list, 'good_traj': good_traj_list})
#                {'good_trajectories': save_dict})
    return save_dict


def plot_good_bad_trajectories(matfile):
    trajectory_folder = utils.trajectory_dir
    flist = glob.glob(os.path.join(trajectory_folder, 'tdump*'))

    good_traj, bad_traj = 0, 0
    file_list = loadmat(matfile, squeeze_me=True)['file_list']
    good_list = loadmat(matfile, squeeze_me=True)['good_traj']
#    oct_struct['tdump20150825H0000']

    fig_good, ax_g, m_ax_g = utils.make_map_plot()
    fig_bad, ax_b, m_ax_b = utils.make_map_plot()

    for f in flist:
        tdump = utils.read_tdump(f)
        t_group = tdump.groupby('tnum')
#        good_list = np.array(mat_cont[os.path.basename(f)].tolist())
        for key in t_group.groups.keys():

            ncfile = r'{}-{:02}-2deg.nc'.format(os.path.basename(f)[5:13], key)
            indx = np.argwhere(file_list == ncfile).squeeze().item()
            good = good_list[indx]
            traj = t_group.get_group(key)
            color = cm.rainbow(np.random.uniform(0,1))
            if good:
                good_traj += 1
                utils.plot_single(t=traj, m=m_ax_g, c=color)
            else:
                bad_traj += 1
                utils.plot_single(t=traj, m=m_ax_b, c=color)
    ax_g.set_title('CSET good trajectories, {} total'.format(good_traj),
                   y=1.08)
    ax_b.set_title('CSET bad trajectories, {} total'.format(bad_traj),
                   y=1.08)

    fig_good.savefig('/home/disk/p/jkcm/good.png')
    fig_bad.savefig('/home/disk/p/jkcm/bad.png')

if __name__ == "__main__":

#    matfile = '/home/disk/eos4/jkcm/Data/CSET/Lagrangian_project/good_trajectories_12.mat'

    savename = '/home/disk/eos4/jkcm/Data/CSET/Lagrangian_project/good_trajectories_morewest.mat'
    save_dict = make_list_of_trajectories(testfunction=trajectory_12_hour_check,
                                          output=savename)


    plot_good_bad_trajectories(savename)


