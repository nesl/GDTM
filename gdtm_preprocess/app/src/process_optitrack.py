import argparse
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import sys
import os
import json

from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm

def process(optitrack_csv, metadata_json, output_csv):

  # check if optitrack csv exists
  if not os.path.isfile(optitrack_csv):
    print(f'Error: OptiTrack CSV {optitrack_csv} not found')
    exit(1)

  # check if metadata json exists
  if not os.path.isfile(metadata_json): 
    print(f'Error: Metadata JSON {metadata_json} not found')
    exit(1)

  with open(metadata_json) as f: 
    metadata = json.load(f)

  # create directory for output
  Path(os.path.dirname(output_csv)).mkdir(parents=True, exist_ok=True)

  # load optitrack csv
  print(f'Processing {optitrack_csv}')
  df = pd.read_csv(optitrack_csv, skiprows=[0, 1, 2, 4, 5, 6], usecols=lambda c: not c.startswith('Unnamed:') and 'Marker' not in c)

  def rename(header): 
    if header == 'Name': return 'Time'
    if '.' not in header: return f'{header}.raw_qx'
    rigidbody = header.split('.')[0]
    if '.1' in header: return f'{rigidbody}.raw_qy'
    if '.2' in header: return f'{rigidbody}.raw_qz'
    if '.3' in header: return f'{rigidbody}.raw_qw'
    if '.4' in header: return f'{rigidbody}.raw_x'
    if '.5' in header: return f'{rigidbody}.raw_y'
    if '.6' in header: return f'{rigidbody}.raw_z'
  df = df.rename(columns = rename)
  pos_cols = list(filter(lambda c: c == 'Time' or ('raw' in c and 'q' not in c), df.columns))
  df = df.interpolate(columns = pos_cols).dropna(subset = pos_cols)

  # get start time
  start_time = ''
  with open(optitrack_csv, 'r') as f:
    line = f.readline().strip().split(',')
    start_time = datetime.strptime(line[11], '%Y-%m-%d %I.%M.%S.%f %p')

  # change time from delta start time to actual timestamp and sort
  print('Converting timestamps...')
  for idx, row in tqdm(df.iterrows(), total = df.shape[0]):
    df.loc[idx, 'Time'] = start_time + timedelta(seconds = float(df.loc[idx, 'Time'])) + timedelta(hours = 3)
  df = df.sort_values(by = 'Time', ascending = True)

  for car in ['red_car', 'green_car']: 
    if not metadata[car]['exists']: continue

    q = (df[f'{car}.raw_qx'], df[f'{car}.raw_qy'], df[f'{car}.raw_qz'], df[f'{car}.raw_qw'])
    qx = np.array(q[0])
    t = np.array(list(map(lambda t: t.timestamp(), df['Time'])))
    bad_idx = np.isnan(qx)
    good_idx = np.logical_not(bad_idx)
    q = np.array(list(zip(*q)))
    good_data = R.from_quat(q[good_idx])
    good_t = t[good_idx]
    rot = (Slerp(good_t, good_data)(t) * R.from_matrix(metadata[car]['calibration'])).as_rotvec()
    rot = np.swapaxes(rot, 0, 1)

    df[f'{car}.x'] = df[f'{car}.raw_x']
    df[f'{car}.y'] = df[f'{car}.raw_y']
    df[f'{car}.z'] = df[f'{car}.raw_z']
    df[f'{car}.rx'] = rot[0]
    df[f'{car}.ry'] = rot[1]
    df[f'{car}.rz'] = rot[2]

  # save to csv
  df.to_csv(output_csv, index=False)
  print(f'Saved to {output_csv}')



if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('optitrack_csv', type=str, help='Path to OptiTrack CSV')
  parser.add_argument('metadata_json', type=str, help='Path to metadata JSON')
  parser.add_argument('output_csv', type=str, help='Path to output CSV')
  args = parser.parse_args()
  process(args.optitrack_csv, args.metadata_json, args.output_csv)
