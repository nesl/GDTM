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

def split(hdf5, metadata_json, output_folder): 

  # check if hdf5 file exists
  if not os.path.isfile(hdf5): 
    print(f'Error: HDF5 file {hdf5} not found')
    exit(1)

  # check if metadata JSON exists
  if not os.path.isfile(metadata_json): 
    print(f'Error: Metadata JSON {metadata_json} not found')
    exit(1)

  # create directory for output
  Path(output_folder).mkdir(parents=True, exist_ok=True)

  print(f'Splitting HDF5...')
  with open(metadata_json) as f: 
    metadata = json.load(f)
  f = h5py.File(hdf5, 'r')
  fname = os.path.splitext(os.path.basename(hdf5))[0]
  timestamps = list(f.keys())
  for i in trange(0, len(metadata['valid_ranges'])):
    chunk_f = h5py.File(os.path.join(output_folder, f'{fname}_{i}.hdf5'), 'a')
    # start = timestamps.index(metadata['valid_ranges'][i][0])
    # end = timestamps.index(metadata['valid_ranges'][i][1]) + 1
    start = metadata['valid_ranges'][i][0]
    end = metadata['valid_ranges'][i][1] + 1
    valid_range = timestamps[start : end]
    for timestamp in valid_range: 
      f.copy(f[f'/{timestamp}'], chunk_f['/'])
    chunk_f.close()
  f.close()
  print('Finished')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('hdf5', type=str, help='Path to HDF5 file')
  parser.add_argument('metadata_json', type=str, help='Path to metadata JSON')
  parser.add_argument('output_folder', type=str, help='Path to output folder')
  args = parser.parse_args()
  split(args.hdf5, args.metadata_json, args.output_folder)
