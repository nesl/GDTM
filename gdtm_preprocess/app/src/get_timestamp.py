import sys
import numpy as np
import argparse
from pathlib import Path
from pandas import Timestamp
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

def get(f, frame):
  f = h5py.File(f, 'r')
  print(list(f.keys())[frame])
  f.close()

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('file', type=str, help='HDF5 file')
  parser.add_argument('frame', type=int, help='Frame for timestamp')
  args = parser.parse_args()

  get(args.file, args.frame)
