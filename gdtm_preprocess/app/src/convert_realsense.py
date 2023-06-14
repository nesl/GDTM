import argparse
import csv
import os
import cv2
import rosbag
import numpy as np
import pandas as pd
import h5py

from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from utils import datetime_to_ms

def convert(node, realsense_bag, aligned_csv, output_folder, div):

  # check if realsense bag exists
  if not os.path.isfile(realsense_bag):
    print(f'Error: Bag file {realsense_bag} not found')
    exit(1)

  # check if aligned csv exists
  if not os.path.isfile(aligned_csv):
    print(f'Error: CSV file {aligned_csv} not found')
    exit(1)

  # open bag
  print(f'Converting {realsense_bag} to hdf5')
  bag = rosbag.Bag(realsense_bag)

  mp4_base = realsense_bag.replace('.bag', '.mp4')
  rgb_mp4 = mp4_base.replace('realsense', 'realsense_rgb')
  depth_mp4 = mp4_base.replace('realsense', 'realsense_depth')

  if not os.path.isfile(rgb_mp4):
    print('RGB mp4 not found')
    exit()

  if not os.path.isfile(depth_mp4):
    print('Depth mp4 not found')
    exit()

  # create directory for output
  Path(os.path.join(output_folder, f'node_{node}')).mkdir(parents=True, exist_ok=True)

  # open output file and videos
  rgb_vid = cv2.VideoCapture(rgb_mp4)
  
  depth_vid = cv2.VideoCapture(depth_mp4)

  df = pd.read_csv(aligned_csv)

  output_file = os.path.join(output_folder, f'node_{node}', 'realsense.hdf5')
  f = h5py.File(output_file, 'a')

  base_name = os.path.basename(realsense_bag).replace('.bag', '')
  base_name = os.path.join(output_folder, 'realsense', base_name)
  node = f'node_{node}'
  msgs = bag.read_messages()
  rgb_idx = 0
  depth_idx = 0
  for topic, msg, t in tqdm(msgs, total=bag.get_message_count()):
    timestamp = np.array(msg.data).astype(np.datetime64).item()

    if 'rgb_timestamp' in topic:
      ret, frame = rgb_vid.read()
      H, W, _ = frame.shape
      dsize = int(W // div), int(H // div)
      frame = cv2.resize(frame, dsize = dsize)
      rgb_code = cv2.imencode('.jpg', frame)[1]

      while rgb_idx < df['timestamp'].shape[0]:
        curt = df.at[rgb_idx, f'{node}_realsense_rgb']
        if '.' not in curt: curt += '.0'
        curt = datetime.strptime(curt, '%Y-%m-%d %H:%M:%S.%f')
        if curt < timestamp: 
          rgb_idx += 1
          continue
        elif curt > timestamp: 
          break

        framet = df.at[rgb_idx, 'timestamp']
        if '.' not in framet: framet += '.0'
        ms = datetime_to_ms(datetime.strptime(framet, '%Y-%m-%d %H:%M:%S.%f'))
        if ms not in f.keys(): 
          f.create_group(ms)
        if node not in f[ms].keys():
          f[ms].create_group(node)

        f[ms][node].create_dataset('realsense_camera_img', data=rgb_code)

        rgb_idx += 1

    if 'depth_timestamp' in topic:
      ret, frame = depth_vid.read()
      H, W, _ = frame.shape
      dsize = int(W // div), int(H // div)
      frame = cv2.resize(frame, dsize = dsize)
      depth_code = cv2.imencode('.jpg', frame)[1]

      while depth_idx < df['timestamp'].shape[0]:
        curt = df.at[depth_idx, f'{node}_realsense_depth']
        if '.' not in curt: curt += '.0'
        curt = datetime.strptime(curt, '%Y-%m-%d %H:%M:%S.%f')
        if curt < timestamp: 
          depth_idx += 1
          continue
        elif curt > timestamp: 
          break

        framet = df.at[depth_idx, 'timestamp']
        if '.' not in framet: framet += '.0'
        ms = datetime_to_ms(datetime.strptime(framet, '%Y-%m-%d %H:%M:%S.%f'))
        if ms not in f.keys(): 
          f.create_group(ms)
        if node not in f[ms].keys():
          f[ms].create_group(node)

        f[ms][node].create_dataset('realsense_camera_depth', data=depth_code)

        depth_idx += 1

  f.close()
  bag.close()
  rgb_vid.release()
  depth_vid.release()
  print(f'Finished converting {realsense_bag} to HDF5')

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('node', type=int, help='Node id')
  parser.add_argument('aligned_csv', type=str, help='Path to aligned CSV file')
  parser.add_argument('realsense_bag', type=str, help='Path to RealSense bag file')
  parser.add_argument('output_folder', type=str, help='Path to output folder')
  parser.add_argument('-div', type=float, help='Image dimensions multiplier, default 1.0', default=1.0)
  args = parser.parse_args()
  convert(args.node, args.aligned_csv, args.realsense_bag, args.output_folder, args.div)
