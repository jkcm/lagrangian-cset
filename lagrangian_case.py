# -*- coding: utf-8 -*-
"""lagrangian_case.py: Defines the classes for CSET Lagrangian data case types."""

__author__ = "Johannes Mohrmann"
__version__ = "0.1"

from CSET_data_classes import CSET_Data, CSET_Flight_Piece
import utils
import met_utils as mu

import re
import numpy as np
import os
import xarray as xr
import matplotlib.pyplot as plt

all_cases = {
    1: {'ALC_name': 'ALC_RF02B-RF03CD',
        'TLC_name': 'TLC_RF02-RF03_1.0-1.5-2.0',            #opt 1.0, fine
        'trajectories': [0, 1]},
    2: {'ALC_name': 'ALC_RF02C-RF03AB',
        'TLC_name': 'TLC_RF02-RF03_0.5-1.0',                #opt 1.0, fine
        'trajectories': [0, 1]},
    3: {'ALC_name': 'ALC_RF04A-RF05CDE',
        'TLC_name': 'TLC_RF04-RF05_2.0-2.3-2.5-3.0',            #opt 2.0. check
        'trajectories': [0, 1]},
    4: {'ALC_name': 'ALC_RF04BC-RF05AB',
        'TLC_name': 'TLC_RF04-RF05_1.0-2.0',                #opt 2.0, ok
        'trajectories': [0, 1]},
    5: {'ALC_name': 'ALC_RF06A-RF07BCDE',
        'TLC_name': 'TLC_RF06-RF07_3.5-4.0-4.3-4.6-5.0',        #opt 3.0, check 3.5
        'trajectories': [0, 1]},
    6: {'ALC_name': 'ALC_RF06BC-RF07A',
        'TLC_name': 'TLC_RF06-RF07_1.6-2.0-2.3-2.6-3.0',    #opt 1.6, check
        'trajectories': [0, 1]},
    7: {'ALC_name': 'ALC_RF08A-RF09DEF',
        'TLC_name': 'TLC_RF08-RF09_4.0-4.5-5.0',
        'trajectories': [0, 1]},
    8: {'ALC_name': 'ALC_RF08B-RF09BC',
        'TLC_name': 'TLC_RF08-RF09_3.0-3.5', 
        'trajectories': [0, 1]},
    9: {'ALC_name': 'ALC_RF08CD-RF09A',
        'TLC_name': 'TLC_RF08-RF09_1.5-2.0', 
        'trajectories': [0, 1]},
    10: {'ALC_name': 'ALC_RF10A-RF11DE',
        'TLC_name': 'TLC_RF10-RF11_5.5-6.0',                #opt 5.0, removed 
        'trajectories': [0, 1]},
    11: {'ALC_name': 'ALC_RF10BC-RF11BC',
        'TLC_name': 'TLC_RF10-RF11_3.0-3.5-4.0-5.0',        #opt 5.0, fine
        'trajectories': [0, 1]},
    12: {'ALC_name': 'ALC_RF10D-RF11A',
        'TLC_name': 'TLC_RF10-RF11_1.0-1.5',                #opt 1.0, ok
        'trajectories': [0, 1]},
    13: {'ALC_name': 'ALC_RF12A-RF13E',
        'TLC_name': 'TLC_RF12-RF13_4.5',                    #opt 5.0, removed
        'trajectories': [0, 1]},
    14: {'ALC_name': 'ALC_RF12B-RF13CD',
        'TLC_name': 'TLC_RF12-RF13_3.0-3.5',                #added 3.0, ok
        'trajectories': [0, 1]},
    15: {'ALC_name': 'ALC_RF12C-RF13B',
        'TLC_name': 'TLC_RF12-RF13_2.5-3.0',                
        'trajectories': [0, 1]},
    16: {'ALC_name': 'ALC_RF14A-RF15CDE',
        'TLC_name': 'TLC_RF14-RF15_3.5-4.0',            
        'trajectories': [0, 1]},
    17: {'ALC_name': 'ALC_RF14B-RF15B',
        'TLC_name': 'TLC_RF14-RF15_3.0',
        'trajectories': [0, 1]},    
    18: {'ALC_name': 'ALC_RF14CD-RF15A',
        'TLC_name': 'TLC_RF14-RF15_1.0-2.0', 
        'trajectories': [0, 1]}
}



class LagrangianCase(CSET_Data):
    """Parent class for all CSET Lagrangian classes. Inheritance from CSET_Data gives load/save methods
    necessitating a self.data_location (class var), self.name (instance var), self.name_re(class var) """
    # notes to me: shouldn't ever be initialized, so no init. If there's nothing in this 
    # class when you're done writing these classes, delete it.
    
    def __repr__(self):
        return "\n".join(["Type: {}".format(self.casetype), 
                         "Name: {}".format(self.name),
                          "Flights: {}, {}".format(self.outbound_flight, self.return_flight)])

    
#%%
#################################################################################################
### Aircraft CASE
#################################################################################################

class AircraftCase(LagrangianCase): 
    """Lagrangian Case containing only data from aircraft, with a before/after 
    from the out/return flights"""
    casetype = "Aircraft Lagrangian"
    data_location = r'/home/disk/eos4/jkcm/Data/CSET/Python/aircraft_lagrangians'
    name_re = re.compile('ALC_RF(\d\d)([A-F]{1,4})-RF(\d\d)([A-F]{1,4})')  # e.g. ALC_RF02B/RF03CDF
    
    def __init__(self, name, extend_to_next_profile=False):
        match = re.match(self.name_re, name)
        if not match:
            raise ValueError('could not parse input name')
        self.outbound_flight = 'RF'+match.group(1)
        self.return_flight = 'RF'+match.group(3)
        self.outbound_sequences = list(match.group(2))
        self.return_sequences = list(match.group(4))
        self.name = name
        self.add_legs(extend_to_next_profile)
        self.add_flight_data()
        if extend_to_next_profile:
            self.outbound_Flight_Piece.extended = True
            self.return_Flight_Piece.extended = True


    def add_legs(self, extend_to_next_profile=False):
        def get_leg_times(flightname, sequences):
            rfnum = int(flightname[-2:])
            x = utils.read_CSET_Lookup_Table(rf_num=rfnum, legs='all',
                                             sequences=sequences, variables=['Date', 'ST', 'ET'])
            if extend_to_next_profile:
                try:
                    print('extending')
                    subseq_seq = chr(ord(max(sequences))+1)

                    x_e = utils.read_CSET_Lookup_Table(rf_num=rfnum, legs='d',
                                                 sequences=subseq_seq, variables=['Date', 'ST', 'ET'])

                    for k,v in x.items():
                        if isinstance(v, dict): # date, et, or st
                            x[k]['values'] = np.concatenate((v['values'], x_e[k]['values']))
                        else:
                            x[k] = np.concatenate((v, x_e[k]))
                except ValueError as e:
                    print('cannot extend, continuing')
#                     self.outbound_Flight_Piece.extended = False
#                     self.return_Flight_Piece.extended = False
            
            start_dates = [utils.CSET_date_from_table(x['Date']['values'][i], x['ST']['values'][i])
                           for i in range(len(x['Date']['values']))]
            end_dates = [utils.CSET_date_from_table(x['Date']['values'][i], x['ET']['values'][i])
                         for i in range(len(x['Date']['values']))]
            case_legs = {key: x[key] for key in ['leg', 'rf', 'seq']}
            case_legs['Start'] = np.array(start_dates)
            case_legs['End'] = np.array(end_dates)
            return case_legs

        self.outbound_legs = get_leg_times(self.outbound_flight, self.outbound_sequences)
        self.return_legs = get_leg_times(self.return_flight, self.return_sequences)
        self.outbound_start_time = min(self.outbound_legs['Start'])
        self.outbound_end_time = max(self.outbound_legs['End'])
        self.return_start_time = min(self.return_legs['Start'])
        self.return_end_time = max(self.return_legs['End'])
        
        
    def add_flight_data(self):
        self.outbound_Flight_Piece = CSET_Flight_Piece(flight_name=self.outbound_flight,
                                                   start_time = self.outbound_start_time,
                                                   end_time = self.outbound_end_time)
        self.outbound_Flight_Piece.flight_data, seqs = utils.add_leg_sequence_labels(self.outbound_Flight_Piece.flight_data, 
                                          start_times=self.outbound_legs['Start'],
                                          end_times=self.outbound_legs['End'],
                                          legs=self.outbound_legs['leg'],
                                          sequences=self.outbound_legs['seq'])
        self.outbound_Flight_Piece.sequences = sorted(list(set(seqs)))
        self.outbound_flight_data = self.outbound_Flight_Piece.flight_data
        
        self.return_Flight_Piece = CSET_Flight_Piece(flight_name=self.return_flight,
                                                   start_time=self.return_start_time,
                                                   end_time=self.return_end_time)
        
        self.return_Flight_Piece.flight_data, seqs = utils.add_leg_sequence_labels(self.return_Flight_Piece.flight_data, 
                                             start_times=self.return_legs['Start'],
                                             end_times=self.return_legs['End'],
                                             legs=self.return_legs['leg'],
                                             sequences=self.return_legs['seq'])
        self.return_Flight_Piece.sequences = sorted(list(set(seqs)))        
        self.return_flight_data = self.return_Flight_Piece.flight_data
        
    def add_ERA_data(self):
        self.outbound_Flight_Piece.add_ERA_data()
        self.return_Flight_Piece.add_ERA_data()
        
    def add_radarlidar_data(self):
        self.outbound_Flight_Piece.add_radarlidar_data()
        self.return_Flight_Piece.add_radarlidar_data()
        
    def add_precip_data(self):
        self.outbound_Flight_Piece.add_precip_data()
        self.outbound_Flight_Piece.precip_data, _ = utils.add_leg_sequence_labels(self.outbound_Flight_Piece.precip_data, 
                                      start_times=self.outbound_legs['Start'],
                                      end_times=self.outbound_legs['End'],
                                      legs=self.outbound_legs['leg'],
                                      sequences=self.outbound_legs['seq'])
        self.return_Flight_Piece.add_precip_data()
        self.return_Flight_Piece.precip_data, _ = utils.add_leg_sequence_labels(self.return_Flight_Piece.precip_data, 
                                      start_times=self.return_legs['Start'],
                                      end_times=self.return_legs['End'],
                                      legs=self.return_legs['leg'],
                                      sequences=self.return_legs['seq'])

        
#%%
#################################################################################################
### Trajectory CASE
#################################################################################################
        
class TrajectoryCase(LagrangianCase):
    casetype = "Trajectory Lagrangian"
    data_location = r'/home/disk/eos4/jkcm/Data/CSET/Python/trajectory_lagrangians'
    name_re = re.compile('TLC_RF(\d\d)-RF(\d\d)_(\d\.\d)(-\d\.\d)*')  # e.g. TLC_RF02-RF03_1.3-1.6-2.0

    def __init__(self, name, goes_degree='2deg'):
        match = re.match(self.name_re, name)
        if not match:
            raise ValueError('could not parse input name')
        self.name = name
        self.outbound_flight = 'RF'+match.group(1)
        self.return_flight = 'RF'+match.group(2)
        self.trajectories = name[14:].split('-')
        
        self.trajectory_files = {}
        self.goes_files = {}
        
        for traj in self.trajectories:
            self.trajectory_files[traj] = os.path.join(utils.trajectory_netcdf_dir, 
                                  "{}_72h_forward_{}.nc".format(self.outbound_flight.lower(), traj))
            self.goes_files[traj] = os.path.join(utils.GOES_trajectories, 
                                  "{}_{}-{}-{}.nc".format(self.outbound_flight.lower(),
                                                          self.return_flight.lower(),
                                                          traj, goes_degree))
        for f in list(self.trajectory_files.values()) + list(self.goes_files.values()):
            if not os.path.exists(f):
                raise IOError("file does not exist: " + f)
                
    def get_goes_data(self, traj):
        return xr.open_dataset(self.goes_files[traj])
                
    def add_traj_data(self):
        self.traj_data = dict()
        for trajname, tf in self.trajectory_files.items():
            data = xr.open_dataset(tf)
            #add vertical velocity
            w_pres = data.ERA_w.values
            rho = mu.density_from_p_Tv(np.broadcast_to(data.level.values, w_pres.shape)*100, Tv=data.ERA_t.values)  # TODO get virtual temp right
            w_vert = -w_pres/(rho*9.81)
            data['ERA_w_vert'] = (('time', 'level'), np.array(w_vert))
            data['ERA_w_vert'] = data['ERA_w_vert'].assign_attrs({"long_name": "vertical_velocity", "units": "m/s"})
            data['ERA_rho'] = (('time', 'level'), np.array(rho))
            data['ERA_rho'] = data['ERA_rho'].assign_attrs({"long_name": "air density", "units": "kg/m3"})
            
            wspd = np.sqrt(data.ERA_u.values**2 + data.ERA_v.values**2)
            data['ERA_wspd'] = (data.ERA_u.dims, wspd, {"long_name": "wind speed", "units": data.ERA_u.units})
            
            
            if 'ERA_ens_w' in data.data_vars.keys():
#                 print('adding ERA ens w_vert')
                w_pres_ens = data.ERA_ens_w.values
                ens_t = data.ERA_ens_t.values
                rho_ens = mu.density_from_p_Tv(np.broadcast_to(data.ens_level.values, w_pres_ens.shape)*100, 
                                               Tv=ens_t)  # TODO get virtual temp right
                ens_w_vert = -w_pres_ens/(rho_ens*9.81)
                data['ERA_ens_w_vert'] = (('time', 'number', 'ens_level'), np.array(ens_w_vert))
                data['ERA_ens_w_vert'] = data['ERA_ens_w_vert'].assign_attrs({"long_name": "ensemble_vertical_velocity", "units": "m/s"})
                data['ERA_ens_rho'] = (('time', 'number', 'ens_level'), np.array(rho_ens))
                data['ERA_ens_rho'] = data['ERA_ens_rho'].assign_attrs({"long_name": "ensemble_air_density", "units": "kg/m3"})
            
            
            self.traj_data[trajname] = data  

        return self.traj_data
    
    
    def get_variable(self, datatype, varname, level=None, level2=None):
        results = {}
        if datatype not in ['goes', 'traj', 'traj_lev', 'traj_int']:
            raise ValueError('cound not parse data type')
        if datatype not in 'goes' and self.traj_data is not None:
            for traj, ds in self.traj_data.items():
                if datatype == 'traj':
                    results[traj] = ds[varname]
                elif datatype=='traj_lev':
                    if level is None:
                        raise ValueError("specify a level")
                    results[traj] = ds[varname].sel(level=level)
                elif datatype=='traj_int':
                    if level is None or level2 is None:
                        raise ValueError("specify a level")
                    results[traj] = ds[varname].sel(level=slice(min(level, level2), max(level, level2))).mean(axis=1)
        else:
            files = self.goes_files if datatype=='goes' else self.trajectory_files
            for traj, gfile in files.items():
                with xr.open_dataset(gfile) as ds:
                    if datatype in ['goes', 'traj']:
                        results[traj] = ds[varname]
                    elif datatype=='traj_lev':
                        if level is None:
                            raise ValueError("specify a level")
                        results[traj] = ds[varname].sel(level=level)
                    elif datatype=='traj_int':
                        if level is None or level2 is None:
                            raise ValueError("specify a level")
                        results[traj] = ds[varname].sel(level=slice(min(level, level2), max(level, level2))).mean(axis=1)
        return results
    
    def get_from_inv(self, varname, ens=False):
        if ens:
            lev = 'ens_level'
            temp = 'ERA_ens_t'
        else:
            lev = 'level'    
            temp = 'ERA_t'
        if not hasattr(self, 'traj_data'):
            print("warning: ERA traj data not added; adding now")
            self.add_traj_data()
        ret = dict()
        for trajname, tf in zip(self.trajectories, self.trajectory_files):
            ERA_data = self.traj_data[trajname]
#             t = ERA_data[temp].values
#             p = np.broadcast_to(ERA_data[lev].values, t.shape)
#             z_bad = ERA_data.ERA_z.values/9.81
#             z = np.broadcast_to(z_bad, t.shape)
            t = ERA_data.ERA_t.values
            p = np.broadcast_to(ERA_data.level.values, t.shape)
        
            z = ERA_data.ERA_z.values/9.81
            try:
#                 heff = mu.inversion_(z, theta, handle_nans=True)
                inv_dict = mu.get_inversion_layer_2d(z, t, p, axis=0, handle_nans=True)
            except ValueError as e:
                print(self.name)
                print(trajname)
                print(ERA_data.lon)
                raise e
            if not ens:
                vals = np.empty_like(ERA_data.time.values).astype(float)
            else:
                vals = np.empty_like(ERA_data.ERA_ens_sp.values).astype(float)
            if varname == 'z_i':
                vals = inv_dict['z_bot']
                newarray = ERA_data.ERA_z.mean(dim='level').copy(deep=True)
                newarray.values = vals
                newarray.name = 'z_i'
            else:
#                 print(inv_dict['z_bot'])
#                 print(inv_dict['z_top'])
                for i, (b, t) in enumerate(zip(inv_dict['i_bot'].astype(int), inv_dict['i_top'].astype(int))):
                    if np.abs(b)>200:
                        print(inv_dict['i_bot'][i])
                    if np.isnan(b) or np.isnan(t):
                        vals[i] = float('nan')
                        print("NAAAAN")
                    else:
                        if not ens:
                            vals[i] = np.nanmean(ERA_data[varname][i,slice(min(b,t), max(b,t))])
                        else: 
                            try:
                                p_bot = ERA_data.level.values[b]
                                p_top = ERA_data.level.values[t]
                            except IndexError as e:
                                vals[i,:] = np.nan
                                print("muck up on case {}, traj {}, i={}".format(self, trajname, i))
                                continue
                            i_bot = int(np.nonzero(ERA_data.ens_level.values == p_bot)[0][0])
                            try:
                                i_top = int(np.nonzero(ERA_data.ens_level.values == p_top)[0][0])
                            except IndexError as e:
                                i_top = int(np.nonzero(ERA_data.ens_level.values == max(ERA_data.ens_level.values))[0][0])
#                             print("idx of level b: {}".format(b))
#                             print("level b: {}".format(p_bot))
#                             print("idx of ens level b: {}".format(i_bot))
#                             print("ens level b: {}".format(ERA_data.ens_level.values[i_bot]))
#                             print("idx of level t: {}".format(t))
#                             print("level t: {}".format(p_top))
#                             print("idx of ens level t: {}".format(i_top))
#                             print("ens level t: {}".format(ERA_data.ens_level.values[i_top]))
#                             print(ERA_data.ens_level[slice(min(i_bot,i_top), max(i_bot,i_top))])
                            vals[i,:] = ERA_data[varname][i,:,slice(min(i_bot,i_top), max(i_bot,i_top))].mean(dim='ens_level')
                newarray = ERA_data[varname].mean(dim=lev).copy(deep=True)
#                 print(newarray.shape)
#                 print(vals.shape)
                newarray.values = vals
            ret[trajname] = newarray
        return ret
    
    
    def dep_get_subsidence(self):
        def get_from_inv(ERA_data, varname):
            z = ERA_data.ERA_z.values/9.81
            theta = mu.theta_from_p_T(p=np.broadcast_to(ERA_data.level.values, z.shape), T=ERA_data.ERA_t.values)
            heff = mu.heffter_pblht_2d(z, theta)
            ret = np.empty_like(ERA_data.time.values).astype(float)
            for i, (b, t) in enumerate(zip(heff['i_bot'].astype(int), heff['i_top'].astype(int))):
                ret[i] = np.nanmean(ERA_data[varname][i,slice(min(b,t), max(b,t))])
            return ret

        if not hasattr(self, 'traj_data'):
            self.add_traj_data()
        ret = dict()
        for trajname, tf in zip(self.trajectories, self.trajectory_files):
            data = self.traj_data[trajname]
            ERA_w = get_from_inv(data, 'ERA_w')
            ret[trajname] = ERA_w
            
        return ret
            
            
    
    
class CombinedCase(LagrangianCase):
    casetype = "Combined Aircraft/Trajectory Lagrangian"

    def __init__(self, ALC, TLC, number):
        self.ALC = ALC
        self.TLC = TLC 
        self.name = str(number) + "_" + self.ALC.name.split('_')[1]
        self.outbound_flight = ALC.outbound_flight
        self.return_flight = ALC.return_flight
        
    def plot(self, save=False, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8.5,5.5))

        llr = {'lat': (20, 45), 'lon': (-160, -120)}

        m = utils.bmap(ax=ax, llr=llr)

        ax.text(0, 0.9, self.name, 
            horizontalalignment='left', transform=ax.transAxes, fontdict = {'size': 20})

        m.plot(self.ALC.outbound_flight_data['GGLON'], self.ALC.outbound_flight_data['GGLAT'],
              latlon=True, lw=10, label='outbound')
        m.plot(self.ALC.return_flight_data['GGLON'], self.ALC.return_flight_data['GGLAT'],
              latlon=True, lw=10, label='return')

        waypoints = utils.get_waypoint_data(self.TLC.outbound_flight, waypoint_type='b')
        for f in self.TLC.trajectory_files:
            with xr.open_dataset(f) as data:
                end = waypoints.loc[float(f[-6:-3])].ret_time.to_pydatetime()
                idx = utils.as_datetime(data.time.values) < utils.as_datetime(end)
                m.plot(data.lon[idx], data.lat[idx], latlon=True, lw=5, ls='--', label='traj '+f[-6:-3])


        ax.legend(loc='lower right', ncol=2)
        if save:
            fig.savefig(os.path.join(utils.plot_dir, 'case_plots', 'map_{}.png'.format(self.name)),
                        dpi=300, bbox_inches='tight')
