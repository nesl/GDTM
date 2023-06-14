import sys
import numpy as np
import argparse
from pathlib import Path
from pandas import Timestamp
import pandas as pd
from tqdm import tqdm
import csv
import os
import pickle
import json

import scipy.interpolate as spi
from datetime import datetime
import datetime as dt

import rosbag
from std_msgs.msg import Int32, String
from msg_pkg.msg import RadarData
import mmwave_utils as mf
from utils import datetime_to_ms
import h5py

csv_filename = 'None'
output_folder = 'mmwave'

cfg = {}
par = {}

def get_cfg(cfg_str):
  global cfg
  global par
  
  reset = "xWR14xx"
  cfg = json.loads(cfg_str)
  cfg, par = mf.get_conf(cfg)


def convert(node, mmwave_bag, aligned_csv, output_folder):

  # check if mmwave bag exists
  if not os.path.isfile(mmwave_bag): 
    print(f'Error: Bag file {mmwave_bag} not found')
    exit(1)

  # check if aligned csv exists
  if not os.path.isfile(aligned_csv): 
    print(f'Error: CSV file {aligned_csv} not found')
    exit(1)

  # open bag
  print(f'Converting {mmwave_bag} to HDF5')
  bag = rosbag.Bag(mmwave_bag)
  msgs = bag.read_messages()

  df = pd.read_csv(aligned_csv)

  # create directory for output
  Path(os.path.join(output_folder, f'node_{node}')).mkdir(parents=True, exist_ok=True)
  output_file = os.path.join(output_folder, f'node_{node}', 'mmwave.hdf5')
  f = h5py.File(output_file, 'a')

  node = f'node_{node}'
  idx = 0
  for topic, msg, t in tqdm(msgs, total=bag.get_message_count()):

    if not "RadarData" in topic:
      continue

    cfg_str = msg.radar_cfg
    get_cfg(cfg_str)

    tmp_d = {}

    if msg.range_profile_valid == True:
      y = msg.range_profile
      tmp_d["range_profile"] = y
    else:
      tmp_d["range_profile"] = None
      
    if msg.noise_profile_valid == True:
      tmp_d["noise_profile"] = msg.noise_profile
    else:
      tmp_d["noise_profile"] = None

    if msg.azimuth_static_valid == True:
      a = msg.azimuth_static

      if len(a) != mf.num_range_bin(cfg) * mf.num_tx_azim_antenna(cfg) * mf.num_rx_antenna(cfg) * 2:
        continue

      a = np.array([a[i] + 1j * a[i+1] for i in range(0, len(a), 2)])
      a = np.reshape(a, (mf.num_range_bin(cfg), mf.num_tx_azim_antenna(cfg) * mf.num_rx_antenna(cfg)))
      a = np.fft.fft(a, mf.num_angular_bin(cfg))
      a = np.abs(a)

      # put left to center, put center to right     
      a = np.fft.fftshift(a, axes=(1,))

      # cut off first angle bin
      a = a[:,1:]

      t = np.array(range(-mf.num_angular_bin(cfg)//2 + 1, mf.num_angular_bin(cfg)//2)) * (2 / mf.num_angular_bin(cfg))
      # t * ((1 + np.sqrt(5)) / 2)
      t = np.arcsin(t)
      r = np.array(range(mf.num_range_bin(cfg))) * mf.range_resolution(cfg)
  
      range_depth = mf.num_range_bin(cfg) * mf.range_resolution(cfg)
      range_width, grid_res = range_depth / 2, 400
      
      xi = np.linspace(-range_width, range_width, grid_res)
      yi = np.linspace(0, range_depth, grid_res)
      xi, yi = np.meshgrid(xi, yi)
  
      x = np.array([r]).T * np.sin(t)
      y = np.array([r]).T * np.cos(t)
      y = y - par['range_bias']
      
      zi = spi.griddata((x.ravel(), y.ravel()), a.ravel(), (xi, yi), method='linear')
      zi = zi[:-1,:-1]

      tmp_d["azimuth_static"] = zi[::-1,::-1]
    else:
      tmp_d["azimuth_static"] = None

    if msg.range_doppler_valid == True:
      if len(msg.range_doppler) != mf.num_range_bin(cfg) * mf.num_doppler_bin(cfg):
        continue

      a = np.array(msg.range_doppler)
      b = np.reshape(a, (mf.num_range_bin(cfg), mf.num_doppler_bin(cfg)))
      # put left to center, put center to right
      c = np.fft.fftshift(b, axes=(1,))
      tmp_d["range_doppler"] = c
    else:
      tmp_d["range_doppler"] = None
    
    points = msg.points

    p_all = {}
    ii = 0
    for p in points:
      p_tmp = {}
      p_tmp["x"] = float(p.x)
      p_tmp["y"] = float(p.y)
      p_tmp["z"] = float(p.z)
      p_tmp["v"] = float(p.intensity)
      p_all["{},{}".format(int(p.range), int(p.doppler))] = p_tmp
      
      ii += 1
    
    tmp_d["detected_points"] = p_all

    h_t = {}
    h_t["time"] = 2021
    h_t["number"] = 0
    tmp_d["header"] = h_t

    if tmp_d['range_doppler'] is None:
      continue

    timestamp = datetime.strptime(msg.timestamp, "%Y-%m-%d %H:%M:%S.%f")
    dpoints = json.dumps(tmp_d['detected_points'])
    
    while idx < df['timestamp'].shape[0]: 
      curt = df.at[idx, f'{node}_mmwave']
      if '.' not in curt: curt += '.0'
      curt = datetime.strptime(curt, '%Y-%m-%d %H:%M:%S.%f')
      if curt < timestamp: 
        idx += 1
        continue
      elif curt > timestamp: 
        break

      framet = df.at[idx, 'timestamp']
      if '.' not in framet: framet += '.0'
      ms = datetime_to_ms(datetime.strptime(framet, '%Y-%m-%d %H:%M:%S.%f'))

      if ms not in f.keys():
        f.create_group(ms)

      if node not in f[ms].keys():
        f[ms].create_group(node)
      
      if tmp_d['azimuth_static'] is not None: 
        f[ms][node].create_dataset('azimuth_static', data=tmp_d['azimuth_static'].astype(np.float32), compression='gzip')
      if tmp_d['range_doppler'] is not None: 
        f[ms][node].create_dataset('range_doppler', data=tmp_d['range_doppler'].astype(np.float32), compression='gzip')
      if tmp_d['range_profile'] is not None: 
        f[ms][node].create_dataset('range_profile', data=np.array(tmp_d['range_profile']).astype(np.float32), compression='gzip')
      if tmp_d['noise_profile'] is not None: 
        f[ms][node].create_dataset('noise_profile', data=np.array(tmp_d['noise_profile']).astype(np.float32), compression='gzip')

      f[ms][node].create_dataset('detected_points', data=dpoints)

      idx += 1

  f.close()
  bag.close()
  print(f'Finished converting {mmwave_bag} to HDF5')

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('node', type=int, help='Node id')
  parser.add_argument('mmwave_bag', type=str, help='Path to mmWave bag file')
  parser.add_argument('aligned_csv', type=str, help='Path to aligned CSV file')
  parser.add_argument('output_folder', type=str, help='Path to output folder')
  args = parser.parse_args()

  convert(args.node, args.mmwave_bag, args.aligned_csv, args.output_folder)
