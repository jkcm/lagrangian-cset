#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 14:17:57 2017

@author: jkcm
"""

from ecmwfapi import ECMWFDataServer
import datetime as dt
from LoopTimer import LoopTimer
import cdsapi

def get_flux_forecast_data(date):
    server = ECMWFDataServer()
    server.retrieve({
        "class": "ea",
        "dataset": "era5",
        "date": "2015-07-01/to/2015-08-31",
        "expver": "1",
        "levtype": "sfc",
        "param": "33.235/34.235/146.128/147.128",
        "step": "0/1/2/3/4/5/6/7/8/9/10/11",
        "stream": "oper",
        "grid": "0.3/0.3",
        "area": "45/-160/15/-115",
        "time": "06:00:00/18:00:00",
        "type": "fc",
        "target": "/home/disk/eos4/jkcm/Data/CSET/ERA5/sfcflux/ERA5.sfcflux.NEP.{}.nc",
})
    
def get_flux_4dvar_data(date):
    datestr = dt.datetime.strftime(date, '%Y-%m-%d')
#     datestr = 'all'
    server = ECMWFDataServer()
    server.retrieve({
        "class": "ea",
        "dataset": "era5",
        "date": datestr,
#         "date": "2015-07-01/to/2015-08-31",
        "expver": "1",
        "levtype": "sfc",
        "param": "33.235/34.235/146.128/147.128",
        "step": "0/1/2/3/4/5/6/7/8/9/10/11",
        "stream": "oper",
        "grid": "0.3/0.3",
        "area": "45/-160/15/-115",
        "time": "09:00:00/21:00:00",
        "type": "4v",
        "format": "netcdf",
        "target": "/home/disk/eos4/jkcm/Data/CSET/ERA5/ERA5.4Dvarflux.NEP.{}.nc".format(datestr),
        })



def get_z_ERA5_data():
    server = ECMWFDataServer()
    server.retrieve({
        "class": "ea",  # Do not change
        "dataset": "era5",  # Do not change
        "expver": "1",  # Do not change
        "stream": "oper",
    # can be "oper", "wave", etcetera; see ERA5 catalogue (http://apps.ecmwf.int/data-catalogues/era5 ) and ERA5 documentation (https://software.ecmwf.int/wiki/display/CKB/ERA5+data+documentation )
        "type": "an",  # can be an (Analysis) or fc (forecast) or 4v (4D variational analysis)
        "levtype": "pl",  # can be "sfc", "pl", "ml", etcetera; see ERA5 documentation
        "param": "129.128",
    # Parameters you want to retrieve. For available parameters see the ERA5 documentation. Specify here using shortName or paramID, and separated by '/'.
        "levelist": "1/2/3/5/7/10/20/30/50/70/100/125/150/175/200/225/250/300/350/400/450/500/550/600/650/700/750/775/800/825/850/875/900/925/950/975/1000",
        "date": "2015-07-01/to/2015-08-31",
        "time": "00:00:00/01:00:00/02:00:00/03:00:00/04:00:00/05:00:00/06:00:00/07:00:00/08:00:00/09:00:00/10:00:00/11:00:00/12:00:00/13:00:00/14:00:00/15:00:00/16:00:00/17:00:00/18:00:00/19:00:00/20:00:00/21:00:00/22:00:00/23:00:00",
    # If above you set "type":"an", "time" is the time of analysis. If above you set "type":"fc", "time" is the initialisation time of the forecast.
        "step": "0",
    # The forecast step. If above you set "type":"an", set "step":"0". If above you set "type":"fc", set "step" > 0.
        "grid": "0.3/0.3",
    # Optional. The horizontal resolution in decimal degrees. If not set, the archived grid as specified in the data documentation is used.
        "area": "45/-160/15/-115",
    # Optional. Subset (clip) to an area. Specify as N/W/S/E in Geographic lat/long degrees. Southern latitudes and western longitudes must be
        # given as negative numbers. Requires "grid" to be set to a regular grid, e.g. "0.3/0.3".
        "format": "netcdf",
    # Optional. Output in NetCDF format. Requires that you also specify 'grid'. If not set, data is delivered in GRIB format, as archived.
        "target": "/home/disk/eos4/jkcm/Data/CSET/ERA5/z/ERA5.z.NEP.{}.nc",
    # Change this to the desired output path and file name, e.g. "data1.nc" or "./data/data1.grib". The default path is the current working directory.
    })


def get_isabel_ERA5_data():
    server = ECMWFDataServer()
    server.retrieve({
        "class": "ea",  # Do not change
        "dataset": "era5",  # Do not change
        "expver": "1",  # Do not change
        "stream": "oper",
    # can be "oper", "wave", etcetera; see ERA5 catalogue (http://apps.ecmwf.int/data-catalogues/era5 ) and ERA5 documentation (https://software.ecmwf.int/wiki/display/CKB/ERA5+data+documentation )
        "type": "an",  # can be an (Analysis) or fc (forecast) or 4v (4D variational analysis)
        "levtype": "pl",  # can be "sfc", "pl", "ml", etcetera; see ERA5 documentation
        "param": "75.128/246.128/248.128",
    # Parameters you want to retrieve. For available parameters see the ERA5 documentation. Specify here using shortName or paramID, and separated by '/'.
        "levelist": "1/2/3/5/7/10/20/30/50/70/100/125/150/175/200/225/250/300/350/400/450/500/550/600/650/700/750/775/800/825/850/875/900/925/950/975/1000",
        "date": "2015-07-01/to/2015-08-31",
        "time": "00:00:00/01:00:00/02:00:00/03:00:00/04:00:00/05:00:00/06:00:00/07:00:00/08:00:00/09:00:00/10:00:00/11:00:00/12:00:00/13:00:00/14:00:00/15:00:00/16:00:00/17:00:00/18:00:00/19:00:00/20:00:00/21:00:00/22:00:00/23:00:00",
    # If above you set "type":"an", "time" is the time of analysis. If above you set "type":"fc", "time" is the initialisation time of the forecast.
        "step": "0",
    # The forecast step. If above you set "type":"an", set "step":"0". If above you set "type":"fc", set "step" > 0.
        "grid": "0.3/0.3",
    # Optional. The horizontal resolution in decimal degrees. If not set, the archived grid as specified in the data documentation is used.
        "area": "45/-160/15/-115",
    # Optional. Subset (clip) to an area. Specify as N/W/S/E in Geographic lat/long degrees. Southern latitudes and western longitudes must be
        # given as negative numbers. Requires "grid" to be set to a regular grid, e.g. "0.3/0.3".
        "format": "netcdf",
    # Optional. Output in NetCDF format. Requires that you also specify 'grid'. If not set, data is delivered in GRIB format, as archived.
        "target": "/home/disk/eos4/jkcm/Data/CSET/ERA5/isabel/ERA5.isabel.NEP.{}.nc",
    # Change this to the desired output path and file name, e.g. "data1.nc" or "./data/data1.grib". The default path is the current working directory.
    })

    
    
def get_sfc_ERA5_Data(date):
    datestr = dt.datetime.strftime(date, '%Y-%m-%d')

    server = ECMWFDataServer()
    server.retrieve({
        "class": "ea",  # Do not change
        "dataset": "era5",  # Do not change
        "expver": "1",  # Do not change
        "stream": "oper",
        # can be "oper", "wave", etcetera; see ERA5 catalogue (http://apps.ecmwf.int/data-catalogues/era5 ) and ERA5 documentation (https://software.ecmwf.int/wiki/display/CKB/ERA5+data+documentation )
        "type": "an",  # can be an (Analysis) or fc (forecast) or 4v (4D variational analysis)
        "levtype": "sfc",  # can be "sfc", "pl", "ml", etcetera; see ERA5 documentation
        "param": "34.128/134.128/164.128/172.128/186.128/187.128/188.128",
        # Parameters you want to retrieve. For available parameters see the ERA5 documentation. Specify here using shortName or paramID, and separated by '/'.
        "date": datestr,  # Set a single date as "YYYY-MM-DD" or a range as "YYYY-MM-DD/to/YYYY-MM-DD".
        "time": "00:00:00/01:00:00/02:00:00/03:00:00/04:00:00/05:00:00/06:00:00/07:00:00/08:00:00/09:00:00/10:00:00/11:00:00/12:00:00/13:00:00/14:00:00/15:00:00/16:00:00/17:00:00/18:00:00/19:00:00/20:00:00/21:00:00/22:00:00/23:00:00",
        # If above you set "type":"an", "time" is the time of analysis. If above you set "type":"fc", "time" is the initialisation time of the forecast.
        "step": "0",
        # The forecast step. If above you set "type":"an", set "step":"0". If above you set "type":"fc", set "step" > 0.
        "grid": "0.3/0.3",
        # Optional. The horizontal resolution in decimal degrees. If not set, the archived grid as specified in the data documentation is used.
        "area": "45/-160/15/-115",
        # Optional. Subset (clip) to an area. Specify as N/W/S/E in Geographic lat/long degrees. Southern latitudes and western longitudes must be
        # given as negative numbers. Requires "grid" to be set to a regular grid, e.g. "0.3/0.3".
        "format": "netcdf",
        # Optional. Output in NetCDF format. Requires that you also specify 'grid'. If not set, data is delivered in GRIB format, as archived.
        "target": "/home/disk/eos4/jkcm/Data/CSET/ERA5/ERA5.sfc.NEP.{}.nc".format(datestr),
        # Change this to the desired output path and file name, e.g. "data1.nc" or "./data/data1.grib". The default path is the current working directory.
    })


def get_ensemble_sfc_ERA5_Data(date):
    datestr = dt.datetime.strftime(date, '%Y-%m-%d')

    server = ECMWFDataServer()
    server.retrieve({
        "number": "0/1/2/3/4/5/6/7/8/9",
        "class": "ea",  # Do not change
        "dataset": "era5",  # Do not change
        "expver": "1",  # Do not change
        "stream": "enda",
        # can be "oper", "wave", etcetera; see ERA5 catalogue (http://apps.ecmwf.int/data-catalogues/era5 ) and ERA5 documentation (https://software.ecmwf.int/wiki/display/CKB/ERA5+data+documentation )
        "type": "an",  # can be an (Analysis) or fc (forecast) or 4v (4D variational analysis)
        "levtype": "sfc",  # can be "sfc", "pl", "ml", etcetera; see ERA5 documentation
        "param": "34.128/134.128/164.128/172.128/186.128/187.128/188.128",
        # Parameters you want to retrieve. For available parameters see the ERA5 documentation. Specify here using shortName or paramID, and separated by '/'.
        "date": datestr,  # Set a single date as "YYYY-MM-DD" or a range as "YYYY-MM-DD/to/YYYY-MM-DD".
        "time": "00:00:00/01:00:00/02:00:00/03:00:00/04:00:00/05:00:00/06:00:00/07:00:00/08:00:00/09:00:00/10:00:00/11:00:00/12:00:00/13:00:00/14:00:00/15:00:00/16:00:00/17:00:00/18:00:00/19:00:00/20:00:00/21:00:00/22:00:00/23:00:00",
        # If above you set "type":"an", "time" is the time of analysis. If above you set "type":"fc", "time" is the initialisation time of the forecast.
        "step": "0",
        # The forecast step. If above you set "type":"an", set "step":"0". If above you set "type":"fc", set "step" > 0.
        "grid": "0.3/0.3",
        # Optional. The horizontal resolution in decimal degrees. If not set, the archived grid as specified in the data documentation is used.
        "area": "45/-160/15/-115",
        # Optional. Subset (clip) to an area. Specify as N/W/S/E in Geographic lat/long degrees. Southern latitudes and western longitudes must be
        # given as negative numbers. Requires "grid" to be set to a regular grid, e.g. "0.3/0.3".
        "format": "netcdf",
        # Optional. Output in NetCDF format. Requires that you also specify 'grid'. If not set, data is delivered in GRIB format, as archived.
        "target": "/home/disk/eos4/jkcm/Data/CSET/ERA5/ensemble/ERA5.sfc.NEP.{}.nc".format(datestr),
        # Change this to the desired output path and file name, e.g. "data1.nc" or "./data/data1.grib". The default path is the current working directory.
    })
    
    
def get_sfc_flux_data(date):
    datestr = dt.datetime.strftime(date, '%Y-%m-%d')

    server = ECMWFDataServer()
    server.retrieve({
        "class": "ea",  # Do not change
        "dataset": "era5",  # Do not change
        "expver": "1",  # Do not change
        "stream": "oper",
        # can be "oper", "wave", etcetera; see ERA5 catalogue (http://apps.ecmwf.int/data-catalogues/era5 ) and ERA5 documentation (https://software.ecmwf.int/wiki/display/CKB/ERA5+data+documentation )
        "type": "an",  # can be an (Analysis) or fc (forecast) or 4v (4D variational analysis)
        "levtype": "sfc",  # can be "sfc", "pl", "ml", etcetera; see ERA5 documentation
        "param": "231.128/232.128",
        # Parameters you want to retrieve. For available parameters see the ERA5 documentation. Specify here using shortName or paramID, and separated by '/'.
        "date": datestr,  # Set a single date as "YYYY-MM-DD" or a range as "YYYY-MM-DD/to/YYYY-MM-DD".
        "time": "00:00:00/01:00:00/02:00:00/03:00:00/04:00:00/05:00:00/06:00:00/07:00:00/08:00:00/09:00:00/10:00:00/11:00:00/12:00:00/13:00:00/14:00:00/15:00:00/16:00:00/17:00:00/18:00:00/19:00:00/20:00:00/21:00:00/22:00:00/23:00:00",
        # If above you set "type":"an", "time" is the time of analysis. If above you set "type":"fc", "time" is the initialisation time of the forecast.
        "step": "0",
        # The forecast step. If above you set "type":"an", set "step":"0". If above you set "type":"fc", set "step" > 0.
        "grid": "0.3/0.3",
        # Optional. The horizontal resolution in decimal degrees. If not set, the archived grid as specified in the data documentation is used.
        "area": "45/-160/15/-115",
        # Optional. Subset (clip) to an area. Specify as N/W/S/E in Geographic lat/long degrees. Southern latitudes and western longitudes must be
        # given as negative numbers. Requires "grid" to be set to a regular grid, e.g. "0.3/0.3".
        "format": "netcdf",
        # Optional. Output in NetCDF format. Requires that you also specify 'grid'. If not set, data is delivered in GRIB format, as archived.
        "target": "/home/disk/eos4/jkcm/Data/CSET/ERA5/ERA5.flux.NEP.{}.nc".format(datestr),
        # Change this to the desired output path and file name, e.g. "data1.nc" or "./data/data1.grib". The default path is the current working directory.
    })


def get_pressure_level_ERA5_Data(date, levels):
    datestr = dt.datetime.strftime(date, '%Y-%m-%d')

    server = ECMWFDataServer()
    server.retrieve({
        "class": "ea",  # Do not change
        "dataset": "era5",  # Do not change
        "expver": "1",  # Do not change
        "stream": "oper",
    # can be "oper", "wave", etcetera; see ERA5 catalogue (http://apps.ecmwf.int/data-catalogues/era5 ) and ERA5 documentation (https://software.ecmwf.int/wiki/display/CKB/ERA5+data+documentation )
        "type": "an",  # can be an (Analysis) or fc (forecast) or 4v (4D variational analysis)
        "levtype": "pl",  # can be "sfc", "pl", "ml", etcetera; see ERA5 documentation
        "param": "u/v/w/r/z/t/o3",
    # Parameters you want to retrieve. For available parameters see the ERA5 documentation. Specify here using shortName or paramID, and separated by '/'.
        "levelist": levels,
        "date": datestr,  # Set a single date as "YYYY-MM-DD" or a range as "YYYY-MM-DD/to/YYYY-MM-DD".
        "time": "00:00:00/01:00:00/02:00:00/03:00:00/04:00:00/05:00:00/06:00:00/07:00:00/08:00:00/09:00:00/10:00:00/11:00:00/12:00:00/13:00:00/14:00:00/15:00:00/16:00:00/17:00:00/18:00:00/19:00:00/20:00:00/21:00:00/22:00:00/23:00:00",
    # If above you set "type":"an", "time" is the time of analysis. If above you set "type":"fc", "time" is the initialisation time of the forecast.
        "step": "0",
    # The forecast step. If above you set "type":"an", set "step":"0". If above you set "type":"fc", set "step" > 0.
        "grid": "0.3/0.3",
    # Optional. The horizontal resolution in decimal degrees. If not set, the archived grid as specified in the data documentation is used.
        "area": "45/-160/15/-115",
    # Optional. Subset (clip) to an area. Specify as N/W/S/E in Geographic lat/long degrees. Southern latitudes and western longitudes must be
        # given as negative numbers. Requires "grid" to be set to a regular grid, e.g. "0.3/0.3".
        "format": "netcdf",
    # Optional. Output in NetCDF format. Requires that you also specify 'grid'. If not set, data is delivered in GRIB format, as archived.
        "target": "/home/disk/eos4/jkcm/Data/CSET/ERA5/ERA5.pres.NEP.{}.nc".format(datestr),
    # Change this to the desired output path and file name, e.g. "data1.nc" or "./data/data1.grib". The default path is the current working directory.
    })



def get_cds_ensemble_pressure_level_ERA5_data(namestr, datestr, levels, param):

    c = cdsapi.Client()

    c.retrieve('reanalysis-era5-complete', {
        'class': 'ea',
        'date': datestr,
        'expver': '1',
        'levelist': levels,
        'levtype': 'pl',
        'number': '0/1/2/3/4/5/6/7/8/9',
        'param': param,
        'stream': 'enda',
        "time": "00:00:00/03:00:00/06:00:00/09:00:00/12:00:00/15:00:00/18:00:00/21:00:00",
        'type': 'an',
        "format": "netcdf", #added
        "grid": "0.3/0.3", #added
        "area": "45/-160/15/-115", #added
        #"step": "0",        #maybe?
    }, "/home/disk/eos4/jkcm/Data/CSET/ERA5/ensemble/ERA5.enda.pres.NEP.temp.{}.nc".format(namestr))


def get_ensemble_pressure_level_ERA5_Data(namestr, datestr, levels, param):
#     datestr = dt.datetime.strftime(date, '%Y-%m-%d')

    server = ECMWFDataServer()
    server.retrieve({
        "number": "0/1/2/3/4/5/6/7/8/9",
        "stream": "enda",
        "class": "ea",  # Do not change
        "dataset": "era5",  # Do not change
        "expver": "1",  # Do not change
        "type": "an",  # can be an (Analysis) or fc (forecast) or 4v (4D variational analysis)
        "levtype": "pl",  # can be "sfc", "pl", "ml", etcetera; see ERA5 documentation
        "param": param, # Parameters you want to retrieve. For available parameters see the ERA5 documentation. Specify here using shortName or paramID, and separated by '/'.
        "levelist": levels,
        "date": datestr,  # Set a single date as "YYYY-MM-DD" or a range as "YYYY-MM-DD/to/YYYY-MM-DD".
        "time": "00:00:00/03:00:00/06:00:00/09:00:00/12:00:00/15:00:00/18:00:00/21:00:00",
        "step": "0",        
        "grid": "0.3/0.3",
        "area": "45/-160/15/-115",
        "format": "netcdf",
        "target": "/home/disk/eos4/jkcm/Data/CSET/ERA5/ensemble/ERA5.enda.pres.NEP.temp.{}.nc".format(namestr),
    })

    
def get_ensemble_sfc_ERA5_Data(namestr, datestr, param):
#     datestr = dt.datetime.strftime(date, '%Y-%m-%d')

    server = ECMWFDataServer()
    server.retrieve({
        "number": "0/1/2/3/4/5/6/7/8/9",
        "stream": "enda",
        "class": "ea",  # Do not change
        "dataset": "era5",  # Do not change
        "expver": "1",  # Do not change
        "type": "an",  # can be an (Analysis) or fc (forecast) or 4v (4D variational analysis)
        "levtype": "sfc",  # can be "sfc", "pl", "ml", etcetera; see ERA5 documentation
        "param": param,
    # Parameters you want to retrieve. For available parameters see the ERA5 documentation. Specify here using shortName or paramID, and separated by '/'.
        "date": datestr,  # Set a single date as "YYYY-MM-DD" or a range as "YYYY-MM-DD/to/YYYY-MM-DD".
        "time": "00:00:00/03:00:00/06:00:00/09:00:00/12:00:00/15:00:00/18:00:00/21:00:00",
    # If above you set "type":"an", "time" is the time of analysis. If above you set "type":"fc", "time" is the initialisation time of the forecast.
        "step": "0",
    # The forecast step. If above you set "type":"an", set "step":"0". If above you set "type":"fc", set "step" > 0.
        "grid": "0.3/0.3",
    # Optional. The horizontal resolution in decimal degrees. If not set, the archived grid as specified in the data documentation is used.
        "area": "45/-160/15/-115",
    # Optional. Subset (clip) to an area. Specify as N/W/S/E in Geographic lat/long degrees. Southern latitudes and western longitudes must be
        # given as negative numbers. Requires "grid" to be set to a regular grid, e.g. "0.3/0.3".
        "format": "netcdf",
    # Optional. Output in NetCDF format. Requires that you also specify 'grid'. If not set, data is delivered in GRIB format, as archived.
        "target": "/home/disk/eos4/jkcm/Data/CSET/ERA5/ensemble/ERA5.enda.sfc.NEP.{}.nc".format(namestr),
    # Change this to the desired output path and file name, e.g. "data1.nc" or "./data/data1.grib". The default path is the current working directory.
    })
    
    
if __name__ == "__main__":

    dates = [dt.datetime(2015, 7, 1) + dt.timedelta(days=i) for i in range(62)]
#     dates = [dt.datetime(2015, 7, 17)]# + dt.timedelta(days=i) for i in range(35)]

#     rf06_dates = [dt.datetime(2015, 7, 17) + dt.timedelta(days=i) for i in range(4)]
#     rf10_dates = [dt.datetime(2015, 7, 27) + dt.timedelta(days=i) for i in range(4)]

#     dates = rf06_dates + rf10_dates

    bl_levels = "700/750/775/800/825/850/875/900/925/950/975/1000"
    all_levels = "1/2/3/5/7/10/20/30/50/70/100/125/150/175/200/225/250/300/350/400/450/500/550/600/650/700/750/775/800/825/850/875/900/925/950/975/1000"
    all_param = "u/v/w/r/z/t/o3"

    
    dates = {'2015-07': "2015-07-01/to/2015-07-31",
             '2015-08': "2015-08-01/to/2015-08-31"}
    
    
    lt = LoopTimer(len(dates))
    for k,v in dates.items():
        lt.update()
#         get_ensemble_pressure_level_ERA5_Data(namestr=k, datestr=v, levels=bl_levels, param="130.128")
        get_cds_ensemble_pressure_level_ERA5_data(namestr=k, datestr=v, levels=bl_levels, param="130.128")
#         get_flux_4dvar_data(i)
#         get_ensemble_pressure_level_ERA5_Data(namestr=k, datestr=v, levels=bl_levels, param="135.128/157.128")

# get_ensemble_sfc_ERA5_Data(namestr=k, datestr=v, param="134.128")
    #     # get_pressure_level_ERA5_Data(i, all_levels)
    #     # get_ensemble_sfc_ERA5_Data(i) 
    #     get_sfc_flux_data(i) 

    #get_isabel_ERA5_data()