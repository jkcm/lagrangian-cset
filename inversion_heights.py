#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 15:22:07 2017

@author: jkcm
"""



import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import numpy as np
import os
import glob
import xarray as xr
from importlib import reload
import pickle
import netCDF4 as nc
from scipy.stats import linregress
import warnings
import sys
sys.path.insert(0, '/home/disk/p/jkcm/Code')
from Lagrangian_CSET import utils
from Lagrangian_CSET import met_utils as mu
from Lagrangian_CSET.LoopTimer import LoopTimer

CSET_dir = r'/home/disk/eos4/jkcm/Data/CSET'
flight_dir = os.path.join(CSET_dir, 'flight_data')



"""
get all the profiles from CSET (upsoundings, downsoundings)
for each profile, estimate the inversion height using:
    RH 50%
    Chris' fancy one
        at least 80% of the time, one could 
        identify a 'RH inversion base' as the altitude of max RH for which RH(zi 
        + 300 m) - RH(zi) < -0.3.  If such a layer does not exist below 4 km or 
        the top of the sounding, we say an inversion is not present.
    heffter
    Richardson
"""


def get_GOES_cloud_top_height(lat, lon, time, percentile, degrees=2, remove_highcloud=True):
    """
    Get the GOES cloud top height value from to the space/time, filtering for high cloud
    """
    variable_list = ['reflectance_vis', 'cloud_phase', 'cloud_top_height', 'cloud_top_temperature']
    data = utils.get_GOES_data(variable_list, lat=lat, lon=lon, time=time, degree=degrees)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warm_low_cloud = (
                        (data['cloud_phase'] == 1) &
                        (data['cloud_top_height'] < 4.) &
                        (data['cloud_top_temperature'] > 273.15))
        cloud_top_heights = data['cloud_top_height'][warm_low_cloud].flatten()

        if percentile == "mean":
            res = np.nanmean(cloud_top_heights)
        elif type(percentile) in (float, int) and percentile < 100 and percentile > 0:
            res = np.nanpercentile(cloud_top_heights, percentile)
        else: 
            raise TypeError("percentile should be an int, float, or 'mean'")
    return res
    

def get_data_from_flight(flight_num, start=None, end=None, var_list=[]):

    flight_file = glob.glob(os.path.join(flight_dir, 'RF{:02d}*.nc'.format(flight_num)))[0]
#     data = xr.open_dataset(flight_file, decode_times=False)
#     data['time'] = nc.num2date(data.Time[:],units=data.Time.units)
    data = xr.open_dataset(flight_file, decode_times=True)
    dates = utils.as_datetime(data.Time.values)
    alt = data['GGALT'].values
    if start is None:
        start = dates[0]
    if end is None:
        end = dates[-1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        index = np.logical_and(np.logical_and(dates >= start, dates <= end), alt < 3600)
    
    [(k, v.long_name) for k,v in data.data_vars.items() if 'Pres' in v.long_name]
    ret = {}
    ret['TIME'] = dates[index]
    for i in var_list:
        if i == 'ATX':
            ret[i] = data[i].values[index]+273.15
        else:
            ret[i] = data[i].values[index]
    
    ret['DENS'] = mu.density_from_p_Tv(data['PSX'].values[index]*100, data['TVIR'].values[index]+273.15)  
    ret['QL'] = data['PLWCC'].values[index]/ret['DENS']
    ret['THETAL'] = mu.get_liquid_water_theta(ret['ATX'], ret['THETA'], ret['QL'])
    ret['QV'] = data['MR'].values[index]/(1+data['MR'].values[index]/1000)
    return ret    


def calc_decoupling_and_zi_from_flight_data(flight_data, usetheta=False):
    
    var_list = ['GGLAT', 'GGLON', 'GGALT', 'RHUM', 'ATX', 'MR', 'THETAE', 'THETA', 'PSX', 'DPXC', 'PLWCC']    
    
    
    sounding_dict = {}
    sounding_dict['TIME'] = flight_data.time.values
    for i in var_list:
        sounding_dict[i] = flight_data[i].values
    if 'ATX' in var_list:
        sounding_dict['ATX'] = sounding_dict['ATX'] + 273.15
    sounding_dict['DENS'] = mu.density_from_p_Tv(flight_data['PSX'].values*100, flight_data['TVIR'].values+273.15)  
    sounding_dict['QL'] = flight_data['PLWCC'].values/sounding_dict['DENS']
    sounding_dict['THETAL'] = mu.get_liquid_water_theta(
        sounding_dict['ATX'], sounding_dict['THETA'], sounding_dict['QL'])
    sounding_dict['QV'] = flight_data['MR'].values/(1+flight_data['MR'].values/1000)
    
    decoupling_dict = mu.calc_decoupling_from_sounding(sounding_dict, usetheta=usetheta)
    zi_dict = mu.calc_zi_from_sounding(sounding_dict)
    return {**decoupling_dict, **zi_dict}


    
def label_points(x, y, labs, ax):
    for label, x, y, in zip(labs, x, y):
        ax.annotate(
        label,
        xy=(x, y), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

# %% Main execution
if __name__ == "__main__":
    
    
    path = r'/home/disk/eos4/jkcm/Data/CSET/LookupTable_all_flights.xls'
    flight = utils.read_CSET_Lookup_Table(path, 
                                       rf_num='all', 
                                       sequences=['d', 'k'],
                                       variables=['Date', 'ST', 'ET'])
    start_times = utils.as_datetime([utils.CSET_date_from_table(d, t) for d, t in
                   zip(flight['Date']['values'], flight['ST']['values'])])
    end_times = utils.as_datetime([utils.CSET_date_from_table(d, t) for d, t in
                 zip(flight['Date']['values'], flight['ET']['values'])])
    sounding_times = list(zip(flight['rf'], start_times, end_times))
    
# %% read in data    
    # get flight info for each sounding
    var_list = ['GGLAT', 'GGLON', 'GGALT', 'RHUM', 'ATX', 'MR', 'THETAE', 'THETA', 'PSX', 'DPXC', 'PLWCC']
    soundings = []
    lt = LoopTimer(len(sounding_times))
    for i in sounding_times:
        lt.update()
        soundings.append(get_data_from_flight(i[0], i[1], i[2], var_list))

    add_dropsondes = True
    if add_dropsondes:
        sondes = []
        sonde_files = glob.glob(os.path.join(utils.dropsonde_dir, "*.nc"))
        for f in sonde_files:
            sondes.append(get_data_from_dropsonde(f))
        


# %% calc inv and dec 
    # get inversion height estimates for each sounding
    heights = []
    for i, snd in enumerate(soundings):
        heights.append(mu.calc_zi_from_sounding(snd))
    
    snd_heights = []
    for sonde in sondes:
        snd_heights.append(mu.calc_zi_from_sounding(sonde))
    
    # get decoupling ests for each sounding
    decouplings = []
    lt = LoopTimer(len(soundings))
    for i, snd in enumerate(soundings):
        lt.update(i)
        decouplings.append(mu.calc_decoupling_from_sounding(snd))
        
    snd_decouplings = []
    for sonde in sondes:
        snd_decouplings.append(mu.calc_decoupling_from_sounding(sonde, usetheta=True))
    
# %% get goes data
    percentiles = [50, 75, 90, 95]
    all_GOES_percentiles = {}
    lt = LoopTimer(len(heights + snd_heights)*len(percentiles))
    for percentile in percentiles:
        GOES_cth = []
        for i, hgt in enumerate((heights + snd_heights)):
            lt.update()
            goes_hgt = get_GOES_cloud_top_height(hgt['lat'], hgt['lon'], hgt['time'], percentile=percentile, degrees=1)
            GOES_cth.append(goes_hgt)
        all_GOES_percentiles[str(percentile)] = np.array(GOES_cth)
    
    lt = LoopTimer(len(heights + snd_heights))
    GOES_CF = []
    for i, hgt in enumerate((heights + snd_heights)):
        cf = get_GOES_cloud_fraction(hgt['lat'], hgt['lon'], hgt['time'], degrees=1)
        GOES_CF.append(cf)

    

    
# %% Plots start here
            

    
    # %%
#    fig, ax = plt.subplots()
#    for i,(snd,hgt) in enumerate(zip(soundings, heights)):
##    for i in range(5):
#        snd, hgt = soundings[i], heights[i]
##        marker = '.' if hgt['Heff']['inversion'] else 'o'
#        inv = hgt['Heff']['inversion']
#        if inv:
#            p = ax.plot(snd['QV'], snd['GGALT'])
#            c = p[0].get_color()
#            mfc = c if inv else 'w'
##            ax.plot(snd['THETA'][hgt['Heff']['i_bot']], hgt['Heff']['z_bot'], '.', ms=20, c = c, mfc=mfc, mew=2)
#    ax.set_ylim(0, 4000)
    

#    fig, ax = plt.subplots()
#    rhs = np.ones(len(soundings)) * 100
#    for i,(snd,hgt) in enumerate(zip(soundings, heights)):
##    for i in range(5):
##    for i in lows:
#        snd, hgt = soundings[i], heights[i]
##        marker = '.' if hgt['Heff']['inversion'] else 'o'
#        inv = hgt['RH50']['inversion']
#        if inv:
#            rhs[i] = snd['RHUM'][hgt['RH50']['i']]
#            print('inv')
#            p = ax.plot(snd['RHUM'], snd['GGALT'])
#            c = p[0].get_color()
##            mfc = c if inv else 'w'
##            ax.plot(i, hgt['RH50']['z'], '.')
#            ax.plot(snd['RHUM'][hgt['RH50']['i']], hgt['RH50']['z'], '.', ms=20, c=c)
##            ax.plot(snd['THETA'][hgt['Heff']['i_bot']], hgt['Heff']['z_bot'], '.', ms=20, c = c, mfc=mfc, mew=2)
#
#    ax.set_ylim(0, 3000)
    
#    lows = np.argwhere(rhs < 40).flatten()

    source = np.concatenate((np.full_like(heights, fill_value='gv'), np.full_like(snd_heights, fill_value='sonde')))
    heights = heights + snd_heights
    decouplings = decouplings + snd_decouplings
    all_soundings = soundings + sondes


    zi_RHCB = np.empty_like(heights, dtype=float)
    zi_Heff_bot = np.empty_like(zi_RHCB)
    zi_Heff_top = np.empty_like(zi_RHCB)
    zi_RH50 = np.empty_like(zi_RHCB)
    lon_p = np.empty_like(zi_RHCB)
    d_theta_e = np.empty_like(zi_RHCB)
    d_theta_l = np.empty_like(zi_RHCB)
    d_qt = np.empty_like(zi_RHCB)
    alpha_thetae = np.empty_like(zi_RHCB)
    alpha_thetal = np.empty_like(zi_RHCB)
    alpha_qt = np.empty_like(zi_RHCB)
    goes_cf = np.empty_like(zi_RHCB)
    lats = np.empty_like(zi_RHCB)
    lons = np.empty_like(zi_RHCB)


    Heff_inv_flag = np.empty_like(zi_RHCB)
    RHCB_inv_flag = np.empty_like(zi_RHCB)
    time = np.empty_like(zi_RHCB, dtype='object')
#    zi_RHCB = np.empty_like(len(heights))


    for i, (hgt, dec) in enumerate(zip(heights, decouplings)):
        zi_RHCB[i] = hgt['RHCB']['z']
        time[i] = hgt['time']
        RHCB_inv_flag[i] = hgt['RHCB']['inversion']
        zi_RH50[i] = hgt['RH50']['z']
        zi_Heff_bot[i] = hgt['Heff']['z_bot']
        zi_Heff_top[i] = hgt['Heff']['z_top']
        Heff_inv_flag[i] = hgt['Heff']['inversion']
        lon_p[i] = hgt['lon_p']
        d_theta_e[i] = dec['d_theta_e']
        d_theta_l[i] = dec['d_theta_l']
        d_qt[i] = dec['d_qt']
        alpha_thetae[i] = dec['alpha_thetae']
        alpha_thetal[i] = dec['alpha_thetal']
        alpha_qt[i] = dec['alpha_qt']
    
    x = np.argsort(lon_p)
    to_exclude = [153]
    x = [i for i in x if i not in to_exclude]
    zi_RHCB = zi_RHCB[x]
    zi_Heff_bot = zi_Heff_bot[x]
    zi_Heff_top = zi_Heff_top[x]
    zi_RH50 = zi_RH50[x]
    lon_p = lon_p[x]
    d_theta_e = d_theta_e[x]
    d_theta_l = d_theta_l[x]
    d_qt = d_qt[x]
    source = source[x]
    time = time[x]
    RHCB_inv_flag = RHCB_inv_flag[x]
    Heff_inv_flag = Heff_inv_flag[x]
    gv_i = source == 'gv'
    labs = np.argsort(x)
    alpha_thetae = alpha_thetae[x]
    alpha_thetal = alpha_thetal[x]
    alpha_qt = alpha_qt[x]
    
    GOES_sorted = {}
    for k, v in all_GOES_percentiles.items():
        GOES_sorted[k] = v[x]
    
    

    save_dict = {"lon_prime": lon_p,
                 "date": time,                 
                 "Heffter_inversion_base": zi_Heff_bot,
                 "Heffter_inversion_top": zi_Heff_top,
                 "Heffter_inversion_flag": Heff_inv_flag,
                 "RelHum_inversion_base": zi_RHCB,
                 "RHCV_inversion_flag": RHCB_inv_flag,
                 "d_theta": d_theta_l,
                 "d_q": d_qt,
                 "source": source}
    
    savefile = r"/home/disk/eos4/jkcm/Data/CSET/Python/inversion_and_decoupling.pickle"
#    with open(savefile, 'wb') as f:
#        pickle.dump(save_dict, f)
 
    # %% GOES_CTH vs zi
    fig, ax = plt.subplots()
    
    cols = list(reversed(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']))
    
    for k, v in GOES_sorted.items():
        c = cols.pop()
        mask = ~np.isnan(v)
        p = ax.plot(zi_Heff_bot, v*1000, '.', c=c, label='{}th %ile'.format(k))
        slp, icept, rval, _, _ = linregress(zi_Heff_bot[mask], v[mask]*1000)
        x = np.arange(0,3500, 100)
        ax.plot(x, icept + slp*x, c=p[0].get_color())
#    p1 = ax.plot(zi_Heff_bot, GOES_sorted['50']*1000, '.', c='b', label='50th %ile')
#    ax.plot(zi_Heff_bot, GOES_sorted['75']*1000, '.', c='r', label='75th %ile')
#    ax.plot(zi_Heff_bot, GOES_sorted['90']*1000, '.', c='g', label='90th %ile')
#    ax.plot(zi_Heff_bot, GOES_sorted['95']*1000, '.', c='y', label='95th %ile')
    ax.plot([0,3500],[0,3500], c='k')
    ax.set_ylabel('GOES CTH percentile')
    ax.set_xlabel('Heffter inversion base')
    ax.legend()
    
    
    
    # %%
    all_flight_dict = {}
    for flt in np.arange(1,17):
        flight_dict = {}
        (s, e) = utils.get_flight_start_end_times(flt, path)
        s, e = utils.as_datetime(s), utils.as_datetime(e)
        mask = np.logical_and(save_dict['date'] >= s, save_dict['date'] <= e)
        print(sum(mask))
        for k,v in save_dict.items():
            flight_dict[k] = v[mask]
        all_flight_dict['rf{:02}'.format(flt)] = flight_dict
        
    savefile = r"/home/disk/eos4/jkcm/Data/CSET/Python/inversion_and_decoupling_by_flight.pickle"
    with open(savefile, 'wb') as f:
        pickle.dump(all_flight_dict, f)
    
    # %%
    fig, ax = plt.subplots()
    ax.set_title("RH50 vs Heffter Top")
    for i,(snd,hgt) in enumerate(zip(all_soundings, heights)):
        ax.plot(hgt['RH50']['z'], hgt['Heff']['z_top'], '.')
    ax.plot([0,3000], [0, 3000], 'k')
    ax.set_xlabel('z_i using RH 50% (m)')
    ax.set_ylabel('z_i using Heffter (top) (m)')
    ax.set_ylim(0,3000)
    ax.set_xlim(0,3000)
    
    # %%
#    fig, ax = plt.subplots()
#    ax.set_title("RH50 vs Chris' RH")
#    for i,(snd,hgt) in enumerate(zip(all_soundings, heights)):
#        ax.plot(hgt['RH50']['z'], hgt['RHCB']['z'], '.')
#    ax.plot([0,3000], [0, 3000], 'k')
#    ax.set_xlabel('z_i using RH 50% (m)')
#    ax.set_ylabel('z_i using Chris\' fancy RH (m)')
#    ax.set_ylim(0,3000)
#    ax.set_xlim(0,3000)

    # %%
    fig, ax = plt.subplots()
    ax.set_title("RH Chris vs Heffter bottom")
    for i,(snd,hgt) in enumerate(zip(all_soundings, heights)):
        ax.plot(hgt['RHCB']['z'], hgt['Heff']['z_bot'], '.')
    ax.plot([0,3000], [0, 3000], 'k')
    ax.set_xlabel('z_i using Chris\' fancy RH (m)')
    ax.set_ylabel('z_i using Heffter (bottom) (m)')
    ax.set_ylim(0,3000)
    ax.set_xlim(0,3000)
    
    
    # %%
    fig, ax = plt.subplots()
    ax.set_title("all measures along lon")
    ax.plot(lon_p, zi_RHCB, '-', marker='o', ms=5, label='Chris \' fancy RH, ({})'.format(sum(~np.isnan(zi_RHCB))))
    ax.plot(lon_p, zi_RH50, '-', marker='o', ms=5, label='RH below 50%, ({})'.format(sum(~np.isnan(zi_RH50))))
    ax.plot(lon_p, zi_Heff_bot, '-', marker='o', ms=5, label='Heffter (bottom), ({})'.format(sum(~np.isnan(zi_Heff_bot))))
    ax.plot(lon_p, zi_Heff_top, '-', marker='o', ms=5, label='Heffer (top), ({})'.format(sum(~np.isnan(zi_Heff_top))))
    ax.set_xlabel('lon-prime coordinate (deg E)')
    ax.set_ylabel('inversion height estimate (m)')
    ax.legend()
    
        # %%
    fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(8,4))
    ax.set_title("PBL depth vs decoupling")
    ax.plot(alpha_thetal[gv_i], zi_Heff_bot[gv_i], '.', c='b', label='GV soundings (q_t, theta_l)')
    ax.plot(alpha_thetal[~gv_i], zi_Heff_bot[~gv_i], '.', c='r', label='dropsondes (q_v , theta, only)')
    ax.legend()
    ax.set_xlabel('α$_ϴ$')
    ax.set_ylabel('Heffter inversion base (m)')
    ax.grid('on')

    ax2.set_title('decoupling vs longitude')
    ax2.plot(lon_p[gv_i], alpha_thetal[gv_i], '.', c='b', label='GV soundings')
    ax2.plot(lon_p[~gv_i], alpha_thetal[~gv_i], '.', c='r', label='dropsondes (q_v only)')
    ax2.set_xlabel('lon-prime (deg)')
    ax2.set_ylabel('α$_ϴ$')
    ax2.legend()
    ax2.grid('on')
    #ϴα
    fig.tight_layout()
    fig.savefig('/home/disk/p/jkcm/plots/cset_lagrangian/dec_Betts_vs_zi.png')
    
    # %%
    fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(8,4))
    ax.set_title("decoupling vs PBL depth")
    ax.plot(d_qt[gv_i], zi_Heff_bot[gv_i], '.', c='b', label='GV soundings (q_t, theta_l)')
    ax.plot(d_qt[~gv_i], zi_Heff_bot[~gv_i], '.', c='r', label='dropsondes (q_v , theta, only)')
    ax.legend()
    ax.set_xlabel('d_q (g/kg)')
    ax.set_ylabel('Heffter inversion base (m)')
    ax.grid('on')

    ax2.set_title('decoupling vs longitude')
    ax2.plot(lon_p[gv_i], d_qt[gv_i], '.', c='b', label='GV soundings')
    ax2.plot(lon_p[~gv_i], d_qt[~gv_i], '.', c='r', label='dropsondes (q_v only)')
    ax2.set_xlabel('lon-prime (deg)')
    ax2.set_ylabel('d_q (g/kg)')
    ax2.axhline(0.5, ls='--', label='Jones et al decoupling threshold')
    ax2.legend()
    ax2.grid('on')
    
    fig.tight_layout()
    fig.savefig('/home/disk/p/jkcm/plots/cset_lagrangian/dec_vs_zi.png')

    # %%
    fig, ax = plt.subplots()
    ax.plot(d_qt[gv_i], d_theta_l[gv_i], '.', c='b', label='GV soundings (q_t, theta_l)')
    ax.plot(d_qt[~gv_i], d_theta_l[~gv_i], '.', c='r', label='dropsondes (q_v , theta, only)')
    lons = [int(i) for i in lon_p]
    ax.set_title('theta_l decoupling vs qt decoupling')
    ax.set_xlabel('d_q')
    ax.set_ylabel('d_theta')
    ax.axvline(0.5, ls='--', label='Jones et al decoupling threshold')
    ax.axhline(0.5, ls='--')
    ax.legend()



    # %%
#    fig, ax = plt.subplots()
#    ax.set_title("decoupling along lon")
#    ax.plot(lon_p, d_theta_e, '-', marker='o', ms=5, label='Theta_e')
#    ax.plot(lon_p, d_theta_l, '-', marker='o', ms=5, label='Theta_l')
#
##    ax.plot(lon_p, zi_RH50, '-', marker='o', ms=5, label='RH below 50%, ({})'.format(sum(~np.isnan(zi_RH50))))
#    
##    ax.plot(lon_p, zi_Heff_bot, '-', marker='o', ms=5, label='Heffter (bottom), ({})'.format(sum(~np.isnan(zi_Heff_bot))))
##    ax.plot(lon_p, zi_Heff_top, '-', marker='o', ms=5, label='Heffer (top), ({})'.format(sum(~np.isnan(zi_Heff_top))))
#    ax.set_xlabel('lon-prime coordinate (deg E)')
#    ax.set_ylabel('decoupling estimate (C)')
#    ax.legend()
    
    # %%
    fig, (ax, ax2, ax3) = plt.subplots(ncols=3)
    ax.plot(d_qt[gv_i], alpha_qt[gv_i], '.', c='b', label='GV soundings (q_t, theta_l)')
    ax.plot(d_qt[~gv_i], alpha_qt[~gv_i], '.', c='r', label='dropsondes (q_v , theta, only)')
    ax.set_title('alpha_qt vs qt decoupling')
    ax.set_xlabel('d_q')
    ax.set_ylabel('alpha_qt')
    ax.axvline(0.5, ls='--', label='Jones et al decoupling threshold')
    ax.grid('on')
    ax.legend()
    
    ax2.plot(d_theta_l[gv_i], alpha_thetal[gv_i], '.', c='b', label='GV soundings (q_t, theta_l)')
    ax2.plot(d_theta_l[~gv_i], alpha_thetal[~gv_i], '.', c='r', label='dropsondes (q_v , theta, only)')
    ax2.set_title('alpha_thetal vs theta_l decoupling')
    ax2.set_xlabel('d_thetal')
    ax2.set_ylabel('alpha_theta_l')
    ax2.axvline(0.5, ls='--', label='Jones et al decoupling threshold')
    ax2.grid('on')
    ax2.legend()
    
    ax3.plot(alpha_qt[gv_i], alpha_thetal[gv_i], '.', c='b', label='GV soundings (q_t, theta_l)')
    ax3.plot(alpha_qt[~gv_i], alpha_thetal[~gv_i], '.', c='r', label='dropsondes (q_v , theta, only)')
    ax3.set_title('alpha_thetal vs alpha_qt')
    ax3.set_xlabel('alpha_qt')
    ax3.set_ylabel('alpha_theta_l')
    ax3.plot([0,1],[0,1], c='k')
    ax3.grid('on')
    ax3.set_xlim([0,1])
    ax3.set_ylim([0,1])
    ax3.legend()
#    label_points(d_qt, d_theta_l, x, ax)


    # %%

    fig, ax = plt.subplots()
    ax.set_title('theta decoupling vs depth')
    ax.plot(zi_Heff_bot[gv_i], alpha_thetal[gv_i], '.', c='b', label='GV soundings')
    ax.plot(zi_Heff_bot[~gv_i], alpha_thetal[~gv_i], '.', c='r', label='dropsondes (q_v only)')
    ax.set_xlabel('Heffter inversion height (m)')
    ax.set_ylabel('alpha_thetal')
    ax.grid('on')
    ax.legend()



    # %%
    fig, ax = plt.subplots()
    ax.set_title('decoupling vs longitude')
    ax.plot(lon_p[gv_i], d_qt[gv_i], '.', c='b', label='GV soundings')
    ax.plot(lon_p[~gv_i], d_qt[~gv_i], '.', c='r', label='dropsondes (q_v only)')
    ax.set_xlabel('lon-prime (deg)')
    ax.set_ylabel('d_q (g/kg)')
    ax.axhline(0.5, ls='--', label='Jones et al decoupling threshold')
    ax.legend()
#    label_points(lon_p, d_qt, x, ax)

    
    # %%
    i = 154
    # trouble cases: 100/154
    # 77: deep sounding, true inversion, but some cu in BL
    # 12: is heff getting the inversion wrong?
    
    snd = all_soundings[i]
    dec = decouplings[i]
    hgt = heights[i]
    
    fig, [ax, ax4, ax2, ax5, ax3] = plt.subplots(ncols=5)
    ax.plot(snd['RHUM'], snd['GGALT'])
    ax.axhline(hgt['Heff']['z_bot'], label='Heff bot')
    ax.axhline(hgt['Heff']['z_top'], ls='--', label='Heff top')

    ax.axhline(hgt['RHCB']['z'], c='r', label='RHCB')
    ax.legend()
    ax.set_xlabel('rel hum')
    ax2.plot(snd['THETAL'], snd['GGALT'], label='liq_theta')
    ax2.plot(snd['THETA'], snd['GGALT'], label='theta')
    ax2.legend()
    ax2.set_xlabel('temp')
    ax3.plot(snd['QL']+snd['QV'], snd['GGALT'], label='qt')
    ax3.plot(snd['QL']*10, snd['GGALT'], label='ql x 10')
    ax3.plot(snd['QV'], snd['GGALT'], label='qv')
    ax3.legend()
    ax3.set_xlabel('q')
    dt = snd['GGALT'][1:] - snd['GGALT'][:-1]
    drhdt = (snd['RHUM'][1:] - snd['RHUM'][:-1])/dt
    ax4.plot(drhdt, snd['GGALT'][1:])
    dthetadt = (snd['THETA'][1:] - snd['THETA'][:-1])/dt
    ax5.plot(dthetadt, snd['GGALT'][1:])
    ax5.axvline(0.005, c='r')
    ax
#    ax3.plot(snd[])