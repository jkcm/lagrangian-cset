# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 13:24:12 2016

@author: jkcm
"""
import os
import sys
import glob
import re
import pandas as pd
import netCDF4 as nc4
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from geopy.distance import great_circle as gdist
import urllib2 as ul
from ftplib import FTP
from ftplib import error_perm
from mpl_toolkits.basemap import Basemap
import smtplib
from email.mime.text import MIMEText
import pickle
from subprocess import call
from time import sleep
import urllib2 as ul


latlon_range = {'lat': (15, 50), 'lon': (-160, -110)}
# HYSPLIT-related parameters
HYSPLIT_workdir = '/home/disk/eos4/jkcm/Data/HYSPLIT/working'  # storing CONTROL
HYSPLIT_call = '/home/disk/p/jkcm/hysplit/trunk/exec/hyts_std'  # to run HYSPLIT
HYSPLIT_source = '/home/disk/eos4/jkcm/Data/HYSPLIT/source'
trajectory_dir = r'/home/disk/eos4/jkcm/Data/CSET/Lagrangian_project/Trajectories'
GOES_source = '/home/disk/eos4/jkcm/Data/CSET/GOES/VISST_pixel'


def varcheck(fname, attr):
    with nc4.Dataset(fname) as dataset:
        if attr in dataset.variables.keys():
#            print 'okay'
            return True
        else:
            print fname
            return False

def get_hysplit_files(run_date, run_hours):
    """Get HYSPLIT files required to run trajectories, return as list of files
    run_date: date of trajectory initialization
    run_hours: hours of trajectory. negative number means backward trajectory

    """
    today = dt.datetime.today()
    start_date = min(run_date, run_date + dt.timedelta(hours=run_hours))
    end_date = max(run_date, run_date + dt.timedelta(hours=run_hours))

    days_since_start = (today.date() - start_date.date()).days
    days_since_end = (today.date() - end_date.date()).days

    file_list = []

    while days_since_start > 0:  # add all analysis files from previous days
        date_to_add = today - dt.timedelta(days=days_since_start)
        if date_to_add > end_date:
            break
        try:
            f, d = get_hysplit_analysis(date_to_add)
            file_list.append(f)
        except ValueError:
            print 'could not find analysis for {}'.format(date_to_add)
        days_since_start -= 1

    if days_since_end < 1:  # trajectory either ends today or in future
        f, d = get_hysplit_appended_files(today)
        file_list.append(f)
        f, d = get_hysplit_forecast_files(today)
        file_list.append(f)

    return file_list


def get_hysplit_analysis(date):
    """
    gets hysplit analysis file for day in date.
    if the file is already acquired, will not download it again.
    if the file does not exist yet raises error.
    """
    ftp = FTP('arlftp.arlhq.noaa.gov')
    ftp.login()
    ftp.cwd('/archives/gdas0p5')
    rx = re.compile('{:%Y%m%d}_gdas0p5\Z'.format(date))
    files = sorted(filter(rx.match, ftp.nlst()))
    if len(files) == 0:
        raise ValueError("ARL: No analysis available for {:%Y%m%d} yet...".format(date))
    newest = files[-1]
    savedir = os.path.join(HYSPLIT_source, 'analysis')
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    print("ARL: Attempting to find analysis file {} locally...".format(newest))
    if os.path.isfile(os.path.join(savedir, newest)):
        print "ARL: File already acquired, not downloading it again."
    else:
        print "ARL: File not found, will grab it from archives."
        try:
            ftp.retrbinary("RETR " + newest,
                           open(os.path.join(savedir, newest), 'wb').write)
        except:
            print "ARL: Error in ftp transfer."
            raise
        print('ARL: Analysis file successfully downloaded')

    savedfile = os.path.join(savedir, newest)
    print('ARL: {}'.format(savedfile))
    return savedfile, date


def get_hysplit_appended_files(date=None):
    """
    Gets most recent HYSPLIT appended files on date.
    Returns file location and initialization time (in the appended
    case that means the end of the file, so gfsa for 18Z on the 12th
    is relevant from 18Z on the 10th through the 12th, for instance)
    """
    f, d = get_hysplit_forecast_files(date, model='gfsa')
    return f, d


def get_hysplit_forecast_files(date=None, model='gfsf'):
    """
    Gets most recent HYSPLIT forecast files on date.
    Finds most recent file on ARL server. If it already exists on disk,
    does nothing and returns location on disk and initialization date.
    If it does not exist on disk, downloads and then returns the same.
    """
    def try_FTP_connect(ftpname):
        counter = 0
        while True:
            try:
                ftp = FTP(ftpname)
                return ftp
            except Exception as e:
                counter += 1
                sleep(1)
                if counter > 20:
                    raise e

    if date is None:
        date = dt.datetime.utcnow()

    ftp = try_FTP_connect('arlftp.arlhq.noaa.gov')
    ftp.login()
    ftp.cwd('/forecast/{:%Y%m%d/}'.format(date))
    rx = re.compile('hysplit.*.{}\Z'.format(model))
    files = filter(rx.match, ftp.nlst())
    if len(files) == 0:  # too early in the day
        print('ARL: no recent {} matches, looking at yesterday instead'.format(model))
        date = date - dt.timedelta(days=1)
        ftp.cwd('/forecast/{:%Y%m%d/}'.format(date))
        files = filter(rx.match, ftp.nlst())
    newest = files[-1]

    savedir = os.path.join(HYSPLIT_source, 'forecast',
                           '{:%Y%m%d}'.format(date))
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    print("ARL: Attempting to find {} for {:%Y-%m-%d}...".format(newest, date))
    if os.path.isfile(os.path.join(savedir, newest)):
        print "ARL: File already acquired, not downloading it again."
    else:
        print "ARL: File not found, will grab it from server."
        try:
            ftp.retrbinary("RETR " + newest,
                           open(os.path.join(savedir, newest), 'wb').write)
        except:
            print "AR:L Error in ftp transfer."
            raise
        print('ARL: File successfully downloaded')

    inittime = int(newest.split('.')[-2][1:3])
    initdate = date.replace(hour=inittime, minute=00, second=00,
                            microsecond=00)
    savedfile = os.path.join(savedir, newest)
    print("ARL: file saves as {}".format(savedfile))
    return(savedfile, initdate)


def write_control_file(start_time, coords, hyfile_list, hours, vertical_type, init_height,
                       tdumpdir):
    """
    This file generates the CONTROL files used for running the trajectories.
    start_time - the datetime object of when the trajectory should start
    coords - list of decimal [lat, lon] pairs. N and E are positive.
    hyfile_list - list of HYSPLIT source files on which to run model
    hours- negative hours means backwards run
    vertical_type:
        0 'data' ie vertical velocity fields
        1 isobaric
        2 isentropic
        3 constant density
        4 constant internal sigma coord
        5 from velocity divergence
        6 something wacky to convert from msl to HYSPLIT's above ground level
        7 spatially averaged vertical velocity
    """

    fl = os.path.join(HYSPLIT_workdir, 'CONTROL')
    f = open(fl, 'w')

    f.write(start_time.strftime('%y %m %d %H\n'))
    f.writelines([str(len(coords)), '\n'])
    for j in coords:
        f.write('{} {} {}\n'.format(str(j[0]), str(j[1]), init_height))
    f.writelines([str(hours), '\n'])

    f.writelines([str(vertical_type), '\n', '10000.0\n'])

    f.write('{}\n'.format(len(hyfile_list)))
    for hyfile in hyfile_list:
        f.writelines([
            os.path.dirname(hyfile), os.sep, '\n',
            os.path.basename(hyfile), '\n'])

    f.writelines([tdumpdir, os.sep, '\n', 'tdump',
                  start_time.strftime('%Y%m%dH%H%M'), '\n'])
    f.close()
    return os.path.join(tdumpdir, 'tdump'+start_time.strftime('%Y%m%dH%H%M'))


def read_tdump(tdump):
    """
    Read a tdump file as output by the HYSPLIT Trajectory Model
        Returns a pandas DataFrame object.
    """
    def parseFunc(y, m, d, H, M):
        return dt.datetime(int('20'+y), int(m), int(d), int(H), int(M))
    columns = ['tnum', 'gnum', 'y', 'm', 'd', 'H', 'M', 'fhour', 'age', 'lat',
               'lon', 'height', 'pres']

    tmp = pd.read_table(tdump, nrows=100, header=None)
    l = [len(i[0]) for i in tmp.values]
    skiprows = l.index(max(l))
    D = pd.read_table(tdump, names=columns,
                      skiprows=skiprows,
                      engine='python',
                      sep=r'\s*',
                      parse_dates={'dtime': ['y', 'm', 'd', 'H', 'M']},
                      date_parser=parseFunc,
                      index_col='dtime')
    return D


def bmap(ax=None, proj='cyl', drawlines=True):

    if ax is None:
        fig, ax = plt.subplots()

    lat_range = latlon_range['lat']
    lon_range = latlon_range['lon']

    m = Basemap(llcrnrlon=lon_range[0], llcrnrlat=lat_range[0],
                urcrnrlon=lon_range[1],  urcrnrlat=lat_range[1],
                rsphere=(6378137.00, 6356752.3142),
                projection=proj, ax=ax, resolution='i')
    if drawlines:
        m.drawparallels(np.arange(-90., 99., 10.), labels=[1, 1, 1, 1])
        m.drawmeridians(np.arange(-180., 180., 10.), labels=[1, 1, 1, 1])
    m.drawcoastlines()

    return m


def gridder(SW, NW, NE, SE, numlats=6, numlons=6):
    """each point is a [lat lon] corner of the desired area"""
    lat_starts = np.linspace(SW[0], NW[0], numlats)
    lon_starts = np.linspace(SW[1], SE[1], numlons)
    lat_ends = np.linspace(SE[0], NE[0], numlats)
    lon_ends = np.linspace(NW[1], NE[1], numlons)
    lat_weight = np.linspace(0., 1., numlats)
    lon_weight = np.linspace(0., 1., numlons)
    lat = (1. - lon_weight[:, None])*lat_starts[None, :] +\
        lon_weight[:, None]*lat_ends[None, :]
    lon = (1. - lat_weight[:, None])*lon_starts[None, :] +\
        lat_weight[:, None]*lon_ends[None, :]
    l = []
    for i in range(numlats):
        for j in range(numlons):
            l.append((lat[j, i], lon[i, j]))
    return(l)


def plot_gridpoints(coords, outfile=None):

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    m = bmap(ax=ax, proj='cyl', drawlines=True)

    m.drawgreatcircle(-121.3, 38.6, -156, 19.8, linestyle='--', c='black')
    colors = cm.rainbow(np.linspace(0, 1, len(coords)))
    for i, crd in enumerate(coords):
        m.plot(crd[1], crd[0], '*', c=colors[i], latlon=True, ms=12, label=i)
        x, y = m(crd[1]+.5, crd[0]+.5)
        ax.annotate(str(i), xy=(x, y), xytext=(x, y), xycoords='data',
                    textcoords='data', fontsize=6)

    if outfile is not None:
        ax.patch.set_visible(False)
        fig.savefig(outfile, dpi=300, transparent=True, bbox_inches='tight',
                    pad_inches=0)


def plot_trajectory(date=None, filename=None):
    if date is None and filename is None:
        print('give me a date (YYYY-MM-DD) or a file, dummy')
        return
    elif date:
        datet = dt.datetime.strptime(date, '%Y-%m-%d')
        filename = os.path.join(trajectory_dir, 'tdump'+datet.strftime('%Y%m%dH%H%M'))

    fig, ax, m_ax = make_map_plot()
    add_tdump_to_plot(m_ax, filename)
    return


def make_map_plot(fig=None):
    if fig is None:
        fig = plt.figure(figsize=(7, 8))
    ax = fig.add_axes([0, 0, 1, 1])
    m_ax = bmap(ax=ax)
    m_ax.drawgreatcircle(-121.3, 38.6, -156, 19.8, linestyle='--', c='black')
    m_ax.plot(-121.3, 38.6, 's', ms=8, c='black', latlon=True)
    m_ax.plot(-156, 19.8, '*', ms=12, c='black', latlon=True)
    m_ax.plot(-118.2, 33.77, 's', ms=8, c='red', latlon=True)
    return fig, ax, m_ax


def plot_single(t, m=None, c=None, i=None):
    m.plot(t.lon.values, t.lat.values, c=c, latlon=True, label=t.tnum[0])
    m.plot(t.lon.values[::6], t.lat.values[::6], '.', c=c, latlon=True)
    m.plot(t.lon.values[0], t.lat.values[0], '*', c=c, latlon=True, ms=12)
    m.plot(t.lon.values[-1], t.lat.values[-1], 's', c=c, latlon=True, ms=8)
    if i is not None:
        plt.annotate(str(i), xy=(t.lon.values[0]+.5, t.lat.values[0]+.5))

    return m


def add_tdump_to_plot(m_ax, tdump):

    T = read_tdump(tdump)
    t = T.groupby('tnum')
    colors = cm.rainbow(np.linspace(0, 1, len(t.groups.keys())))
    for i, k in enumerate(t.groups.keys()):
        m_ax = plot_single(t.get_group(k), m=m_ax, c=colors[i], i=i)

    return


def get_pesky_GOES_files():
    badfiles = []
    with open(r'/home/disk/p/jkcm/Code/Lagrangian_CSET/GOES_Extractor.log', 'r') as f:
            for line in f:
                if r'/home/disk/eos4/mcgibbon/nobackup/GOES' in line:
                    if line not in badfiles:
                        badfiles.append(line)

    with open(r'/home/disk/p/jkcm/Code/Lagrangian_CSET/flawed_GOES.log', 'w') as g:
        for line in sorted(badfiles):
            if os.path.exists(line[:-1]):
                size = '{:3.0f}'.format(os.path.getsize(line[:-1])/1024)
#                print size
            else:
                size = 'NA '
            replace_GOES_file(line[:-1])
            g.writelines(size + '    ' + line)


def replace_GOES_file(filename, savedir=None):
    oldfilename = os.path.basename(filename)
    year = int(oldfilename[12:16])
    date = dt.datetime(year, 01, 01) + dt.timedelta(days=int(oldfilename[16:19]) - 1)
    newfilename = 'prod.goes-west.visst-pixel-netcdf.{:%Y%m%d}.{}'.format(
        date, oldfilename)
    floc = 'prod/goes-west/visst-pixel-netcdf/{:%Y/%m/%d}/'.format(date)
    server = r'http://cloudsgate2.larc.nasa.gov/'
    url = server + floc + newfilename

    try:
        response = ul.urlopen(url)
    except ul.HTTPError:
        print('could not find file!')
        return
    print('file found, downloading')
    if savedir is None:
        savedir = GOES_source
    print ('old size is {}KB'.format(os.path.getsize(filename)/1024.))
    if os.path.dirname(filename) == savedir:
        print('moving old file')
        if not os.path.exists(os.path.join(savedir, 'old')):
            os.makedirs(os.path.join(savedir, 'old'))
        os.rename(filename, os.path.join(savedir, 'old', oldfilename))

    save_file = os.path.join(savedir, oldfilename)
    with open(save_file, 'wb') as fp:
        while True:
            chunk = response.read(16384)
            if not chunk:
                break
            fp.write(chunk)

    print('new size = {}KB'.format(os.path.getsize(save_file)/1024.))


#get_pesky_GOES_files()