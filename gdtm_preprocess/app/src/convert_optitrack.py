import pandas as pd
from configparser import Interpolation
import os
import pickle
from tracemalloc import start
import numpy as np
from datetime import datetime, timedelta
import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pydub
import cv2
import torch
import torchaudio
import math
from tqdm import tqdm, trange
import re
from mpl_toolkits.mplot3d import Axes3D 

import collections
from datetime import datetime

import h5py
import json
from utils import datetime_to_ms, parse_objects

def convert(aligned_csv, metadata_json, output_folder): 

  # check if aligned csv exists
  if not os.path.isfile(aligned_csv): 
    print(f'Error: Aligned CSV {aligned_csv} not found')
    exit(1)

  # create directory for output
  Path(output_folder).mkdir(parents=True, exist_ok=True)
  output_file = os.path.join(output_folder, 'mocap.hdf5')

  # load aligned csv
  df = pd.read_csv(aligned_csv)

  # load metadata json
  with open(metadata_json, 'r') as f: 
    metadata = json.load(f)

  print(f'Converting {aligned_csv} to OptiTrack HDF5')
  f = h5py.File(output_file, 'a')
  for i in trange(len(df)):
    timestamp = df.at[i, 'timestamp']
    time = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f')
    ms = datetime_to_ms(time)
    objs = parse_objects(df.iloc[i], metadata, timestamp)
    data = json.dumps(objs)

    if ms not in f.keys():
      f.create_group(ms)

    f[ms].create_dataset('mocap', data=data)
  
  f.close()
  print(f'Finished converting {aligned_csv} to OptiTrack HDF5')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('aligned_csv', type=str, help='Path to Aligned CSV')
  parser.add_argument('metadata_json', type=str, help='Path to metadata JSON')
  parser.add_argument('output_folder', type=str, help='Path to output folder')
  args = parser.parse_args()
  convert(args.aligned_csv, args.metadata_json, args.output_folder)
