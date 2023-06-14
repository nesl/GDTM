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

def merge(output_hdf5, files): 

  # check if files exist
  for i in range(len(files)): 
    f = files[i]
    if not os.path.isfile(f): 
      print(f'Error: HDF5 file {f} not found')
      exit(1)

  Path(os.path.dirname(output_hdf5)).mkdir(parents=True, exist_ok=True)

  print(f'Merging HDF5s...')
  f = h5py.File(output_hdf5, 'a')
  for i in trange(len(files)):
    f2 = h5py.File(files[i], 'r')
    for timestamp in f2.keys(): 
      f2.copy(f2[f'/{timestamp}'], f['/'])
    f2.close()
  f.close()
  print('Finished')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('output_hdf5', type=str, help='Path to output HDF5 file')
  parser.add_argument('files', type=str, nargs='*', help='Paths to HDF5 files')
  args = parser.parse_args()
  merge(args.output_hdf5, args.files)
