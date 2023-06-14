import numpy as np
from datetime import datetime
from scipy.spatial.transform import Rotation as R
import glob

def datetime_to_path(t):
  vals = (t.year, t.month, t.day,
      t.hour, t.minute, t.second,
      t.microsecond // 1000)
  path = '%04d/%02d/%02d/%02d/%02d/%02d/%04d' % vals
  return path

def path_to_datetime(path):
  toks = path.split('/')
  toks = [int(tok) for tok in toks]
  toks[-1] *= 1000
  return datetime(*toks)

def convert_time(t):
  vals = [t.year, t.month, t.day,
      t.hour, t.minute, t.second,
      t.microsecond // 1000]
  vals = [str(v) for v in vals]
  return vals

def datetime_to_path(t):
  vals = (t.year, t.month, t.day,
      t.hour, t.minute, t.second,
      t.microsecond // 1000)
  path = '%04d/%02d/%02d/%02d/%02d/%02d/%04d' % vals
  return path

def datetime_to_ms(t):
  ms = t.timestamp() * 1000
  ms = int(ms)
  ms = str(ms)
  return ms

def normalize(p): 
  x, y, z = p
  xmin, xmax = -1.5, 2.5
  ymin, ymax = 0, 1
  zmin, zmax = -1.5, 2.5
  x = (x - xmin) / (xmax - xmin)
  y = (y - ymin) / (ymax - ymin)
  z = (z - zmin) / (zmax - zmin)
  return [x, y, z]

# id, type, position (calibrated, optitrack coords), normalized position (from position), roll, pitch, yaw, rotation (flattened 3x3, global2local), timestamp
def parse_objects(mocap, metadata, timestamp, 
  obj_names = ['node1', 'node2', 'node3', 'red_car', 'green_car']):
  objs = []
  for on in obj_names:
    if f'optitrack.{on}.raw_x' not in mocap: continue
    if 'node' in on: 
      rot = R.from_matrix(np.array(metadata[on]['rot']).reshape((3, 3))).as_rotvec()
      obj = {
        'id': obj_names.index(on),
        'type': 'node',
        'position': [metadata[on]['x'], metadata[on]['y'], metadata[on]['z']],
        'normalized_position': normalize([metadata[on]['x'], metadata[on]['y'], metadata[on]['z']]),
        'roll': rot[0], 
        'pitch': rot[1], 
        'yaw': rot[2], 
        'rotation': metadata[on]['rot'],
        'timestamp': timestamp
      }
    else: 
      obj = {
        'id': obj_names.index(on),
        'type': 'truck',
        'position': [float(mocap[f'optitrack.{on}.x']), float(mocap[f'optitrack.{on}.y']), float(mocap[f'optitrack.{on}.z'])],
        'normalized_position': normalize([mocap[f'optitrack.{on}.x'], mocap[f'optitrack.{on}.y'], mocap[f'optitrack.{on}.z']]),
        'roll': float(mocap[f'optitrack.{on}.rx']), 
        'pitch': float(mocap[f'optitrack.{on}.ry']), 
        'yaw': float(mocap[f'optitrack.{on}.rz']), 
        'rotation': R.from_rotvec([float(mocap[f'optitrack.{on}.rx']), float(mocap[f'optitrack.{on}.ry']), float(mocap[f'optitrack.{on}.rz'])]).as_matrix().flatten().tolist(),
        'timestamp': timestamp
      }
    objs.append(obj)
  return objs

def get_group(parent, gname):
  if gname not in parent.keys():
    group = parent.create_group(gname)
  return parent[gname]

def find_file(h5_dir, time):
  fnames = glob.glob(h5_dir + '/*.hdf5')
  time = int(time)
  true_end = -1
  for fname in fnames:
    base = fname.split('/')[-1].split('.')[0]
    start, end = base.split('_')
    start, end = int(start), int(end)
    if end >= true_end:
      true_end = end
    if time >= start and time <= end:
      return fname
  
  #time is beyond all file names defined by mocap
  #this can happen because some modalities have
  #~1 min of data after mocap ends
  #these data cases will be thrown away
  #putting assert here to insure that only
  #these cases will be removed
  # return fnames[-1]
  assert time > true_end
