import pandas as pd
import argparse
import csv
import os

from pathlib import Path
from datetime import datetime
from tqdm import tqdm

def nearest(items, pivot):
  return pd.to_datetime(min(items, key=lambda x: abs(x - pivot)))

def align(optitrack_csv, input_csvs, output_csv, start_frame, frames):

  if not os.path.isfile(optitrack_csv): 
    print(f'OptiTrack CSV {optitrack_csv} not found')
    exit()

  for input_csv in input_csvs: 
    if not os.path.isfile(input_csv): 
      print(f'CSV {input_csv} not found')
      exit()

  Path(os.path.dirname(output_csv)).mkdir(parents=True, exist_ok=True)

  optitrack_df = pd.read_csv(optitrack_csv, sep=',')
  dfs = list(map(lambda csv: pd.read_csv(csv, sep=',', header=None), input_csvs))

  print(f'Converting time to datetime object in {optitrack_csv}...')
  start, end = None, None
  for idx, row in tqdm(optitrack_df.iterrows(), total=optitrack_df.shape[0]): 
    t = optitrack_df.at[idx, 'Time']
    if '.' not in t: t += '.0'
    t = optitrack_df.loc[idx, 'Time'] = datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f')
    if start == None: start = t
    else: start = min(start, t)
    if end == None: end = t
    else: end = max(end, t)

  for i in range(len(input_csvs)): 
    print(f'Converting time to datetime object in {input_csvs[i]}...')
    df = dfs[i]
    df.columns = ['time']
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]): 
      t = df.loc[idx, 'time']
      if '.' not in t: t += '.0'
      df.loc[idx, 'time'] = datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f')

  Path(os.path.dirname(output_csv)).mkdir(parents=True, exist_ok=True)

  csv_file = open(output_csv, 'w', newline='')
  writer = csv.writer(csv_file)

  headers = list(map(lambda header: f'optitrack.{header}', optitrack_df.columns))
  headers[0] = 'timestamp'
  for input_csv in input_csvs: 
    headers.append(input_csv.replace('.csv', '').replace('./data/processed/', '').replace('/', '_'))
  writer.writerow(headers)

  print('Aligning data...')
  cur_frame = 0
  time_range = pd.date_range(start=start, end=end, freq='66ms').to_pydatetime()[start_frame:]
  last_entry = ['n/a'] * len(headers)
  optitrack_idx = 0
  idx = [0] * (len(headers) - 1)
  run = 0
  for timestamp in tqdm(time_range, total=min(frames, len(time_range))):
    if cur_frame >= frames: break
    cur_frame += 1

    entry = [timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')]

    while optitrack_idx + 1 < optitrack_df['Time'].shape[0]: 
      if abs(timestamp - optitrack_df.at[optitrack_idx + 1, 'Time']) < abs(timestamp - optitrack_df.at[optitrack_idx, 'Time']): 
        optitrack_idx += 1
      else: break
    delta_time = abs((optitrack_df.at[optitrack_idx, 'Time'] - timestamp).total_seconds() * 1000)
    if delta_time > 100: 
      for i in range(1, optitrack_df.shape[1]): 
        entry.append(last_entry[len(entry)])
    else: 
      for i in range(1, optitrack_df.shape[1]): 
        entry.append(optitrack_df.at[optitrack_idx, headers[i][headers[i].index('.') + 1:]])

    for i in range(len(dfs)): 
      df = dfs[i]
      while idx[i] + 1 < df['time'].shape[0]: 
        if abs(timestamp - df.at[idx[i] + 1, 'time']) < abs(timestamp - df.at[idx[i], 'time']): 
          idx[i] += 1
        else: break
      delta_time = abs((df.at[idx[i], 'time'] - timestamp).total_seconds() * 1000)
      if delta_time > 100: 
        entry.append(last_entry[len(entry)])
      else: 
        entry.append(df.at[idx[i], 'time'])

    if 'n/a' in entry: continue
    writer.writerow(entry)
    if entry == last_entry: 
      run += 1
      if run == 5: 
        break
    else: run = 1
    last_entry = entry

  csv_file.close()

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--optitrack_csv', type=str, help='Path to processed OptiTrack CSV')
  parser.add_argument('--input_csvs', type=str, nargs='*', help='Paths to processed CSVs')
  parser.add_argument('--output_csv', type=str, help='Path to output CSV')
  parser.add_argument('--start_frame', type=int, help='Index of start frame', default=0)
  parser.add_argument('--frames', type=int, help='Number of frames to align', default=1000000000)
  args = parser.parse_args()
  align(args.optitrack_csv, args.input_csvs, args.output_csv, args.start_frame, args.frames)
