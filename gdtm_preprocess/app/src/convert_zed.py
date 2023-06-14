import sys
import pyzed.sl as sl
import cv2
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import rosbag
from std_msgs.msg import Int32, String
from tqdm import tqdm
import os
from utils import datetime_to_ms
import h5py
from datetime import datetime

def parse_img(img, div_factor=1):
  img = img.get_data()
  img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
  H, W, _ = img.shape
  dsize = int(W // div_factor), int(H // div_factor)
  img = cv2.resize(img, dsize=dsize)
  code = cv2.imencode('.jpg', img)[1]
  return code

def parse_dmap(dmap, div_factor=1):
  dmap = dmap.get_data()
  dmap = dmap.astype(np.uint16)
  H, W = dmap.shape
  dsize = int(W // div_factor), int(H // div_factor)
  dmap = cv2.resize(dmap, dsize=dsize)
  return dmap

def convert(node, zed_svo, aligned_csv, output_folder, img_format, depth_mode, sensing_mode, div):

  # check if zed svo exists
  if 'svo' not in zed_svo: 
    print(f'Error: {zed_svo} is not an SVO file')
    exit(1)

  if not os.path.isfile(zed_svo): 
    print(f'Error: SVO file {zed_svo} not found')
    exit(1)

  zed_bag = zed_svo.replace('svo', 'bag')

  if not os.path.isfile(zed_bag): 
    print(f'Error: Bag file {zed_bag} not found')
    exit(1)

  # check if aligned csv exists
  if not os.path.isfile(aligned_csv): 
    print(f'Error: CSV file {aligned_csv} not found')
    exit(1)

  # Set SVO file for playback
  init_parameters = sl.InitParameters()
  init_parameters.depth_mode = {
    'ultra': sl.DEPTH_MODE.ULTRA, 
    'quality': sl.DEPTH_MODE.QUALITY, 
    'performance': sl.DEPTH_MODE.PERFORMANCE
  }[depth_mode]
  if False: 
    if args.m == 'ultra':
      init_parameters.depth_mode = sl.DEPTH_MODE.ULTRA
    elif args.m == 'quality':
      init_parameters.depth_mode = sl.DEPTH_MODE.QUALITY
    elif args.m == 'performance':
      init_parameters.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    else:
      print("Depth Mode Error! [ultra, performance, quality] only.")
      exit(1)
  
  # don't convert in realtime
  init_parameters.svo_real_time_mode = False
  # use milliliter units (for depth measurements)
  init_parameters.coordinate_units = sl.UNIT.MILLIMETER
  init_parameters.set_from_svo_file(zed_svo)

  # open zed
  zed = sl.Camera()
  status = zed.open(init_parameters)

  if status != sl.ERROR_CODE.SUCCESS:
    print(repr(status))
    zed.close()
    exit(1)

  # get image size
  image_size = zed.get_camera_information().camera_resolution
  width = image_size.width
  height = image_size.height
  width_sbs = width * 2
  
  # prepare image containers
  left_image = sl.Mat()
  right_image = sl.Mat()
  depth_measure = sl.Mat()
  
  runtime_parameter = sl.RuntimeParameters()
  runtime_parameter.sensing_mode = {
    'standard': sl.SENSING_MODE.STANDARD, 
    'fill': sl.SENSING_MODE.FILL
  }[sensing_mode]
  if False: 
    if args.s == 'standard':
      runtime_parameter.sensing_mode = sl.SENSING_MODE.STANDARD
    elif args.s == 'fill':
      runtime_parameter.sensing_mode = sl.SENSING_MODE.FILL
    else:
      print("Sensing Mode Error! [standard, fill] only.")
      exit(1)

  n_frames = zed.get_svo_number_of_frames()

  df = pd.read_csv(aligned_csv)
      
  # open bag
  bag = rosbag.Bag(zed_bag)
  msgs = bag.read_messages()

  timestamps = []
  for topic, msg, t in msgs:
    if "zed_timestamp" in topic:
      timestamps.append(msg.data)

  timestamps = np.array(timestamps).astype(np.datetime64)
  
  Path(os.path.join(output_folder, f'node_{node}')).mkdir(parents=True, exist_ok=True)
  output_file = os.path.join(output_folder, f'node_{node}', 'zed.hdf5')
  f = h5py.File(output_file, 'a')

  print(f'Converting {zed_svo} to HDF5')
  node = f'node_{node}'
  idx = 0
  for i in tqdm(range(n_frames + 1)):
    state = zed.grab(runtime_parameter)
    svo_position = zed.get_svo_position()

    if state == sl.ERROR_CODE.SUCCESS:
      zed.retrieve_image(left_image, sl.VIEW.LEFT)
      left_img = parse_img(left_image,div)

      zed.retrieve_image(right_image, sl.VIEW.RIGHT)
      right_img = parse_img(right_image,div)
      
      zed.retrieve_measure(depth_measure, sl.MEASURE.DEPTH)
      depth_map = parse_dmap(depth_measure,div)
      
      # save everything to disk
      timestamp = timestamps[svo_position]

      while idx < df['timestamp'].shape[0]: 
        curt = df.at[idx, f'{node}_zed']
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

        f[ms][node].create_dataset('zed_camera_left', data=left_img)
        f[ms][node].create_dataset('zed_camera_right', data=right_img)
        f[ms][node].create_dataset('zed_camera_depth', data=depth_map, compression='gzip')

        idx += 1

    else:
      print(f'Current frame position: {svo_position + 1}/{n_frames}')
      print(f'Frame grab error code: {str(state)}')
   
  f.close()
  zed.close()
  print(f'Finished converting {zed_svo} to HDF5')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('node', type=int, help='Node id')
  parser.add_argument('zed_svo', type=str, help='Path to ZED SVO file')
  parser.add_argument('aligned_csv', type=str, help='Path to aligned CSV file')
  parser.add_argument('output_folder', type=str, help='Path to output folder')
  parser.add_argument('-t', choices=['jpg', 'png'], default='jpg', help='Image format (jpg, png), default jpg')
  parser.add_argument('-m', choices=['ultra', 'performance', 'quality'], default='ultra', help="Depth quality mode (ultra, performance, quality), default ultra")
  parser.add_argument('-s', choices=['standard', 'fill'], default='standard', help='Depth sensing mode (standard, fill), default standard')
  parser.add_argument('-div', type=float, default=1.0, help="Image dimention multiplier, default 1.0")
  args = parser.parse_args()
  convert(args.node, args.zed_svo, args.aligned_csv, args.output_folder, args.t, args.m, args.s, args.div)
