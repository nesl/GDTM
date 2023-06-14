import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import argparse
import os
import cv2
import collections
import matplotlib.pyplot as plt
import json
import h5py
import torch
import torchaudio
import math

from tqdm import tqdm
from pathlib import Path

def visualize(node, mocap_hdf5, realsense_hdf5, zed_hdf5, mmwave_hdf5, respeaker_hdf5, output_mp4, start, frames):
  
  if not os.path.isfile(mocap_hdf5): 
    print(f'HDF5 file {mocap_hdf5} not found')
    exit()
  
  if not os.path.isfile(realsense_hdf5): 
    print(f'HDF5 file {realsense_hdf5} not found')
    exit()
  
  if not os.path.isfile(zed_hdf5): 
    print(f'HDF5 file {zed_hdf5} not found')
    exit()
  
  if not os.path.isfile(mmwave_hdf5): 
    print(f'HDF5 file {mmwave_hdf5} not found')
    exit()
  
  if not os.path.isfile(respeaker_hdf5): 
    print(f'HDF5 file {respeaker_hdf5} not found')
    exit()

  node = f'node_{node}'
  mocap = h5py.File(mocap_hdf5, 'r')
  realsense = h5py.File(realsense_hdf5, 'r')
  zed = h5py.File(zed_hdf5, 'r')
  mmwave = h5py.File(mmwave_hdf5, 'r')
  respeaker = h5py.File(respeaker_hdf5, 'r')

  mocap_x = [collections.deque(maxlen = 8), collections.deque(maxlen = 8), collections.deque(maxlen = 8), collections.deque(maxlen = 8), collections.deque(maxlen = 8)]
  mocap_y = [collections.deque(maxlen = 8), collections.deque(maxlen = 8), collections.deque(maxlen = 8), collections.deque(maxlen = 8), collections.deque(maxlen = 8)]
  mocap_z = [collections.deque(maxlen = 8), collections.deque(maxlen = 8), collections.deque(maxlen = 8), collections.deque(maxlen = 8), collections.deque(maxlen = 8)]
  mocap_c = ['black', 'black', 'black', 'red', 'green']

  Path(os.path.dirname(output_mp4)).mkdir(parents=True, exist_ok=True)
  out = cv2.VideoWriter(output_mp4, cv2.VideoWriter_fourcc(*'mp4v'), 15, (1600, 900))

  print("Rendering...")
  timestamps = list(mocap.keys())
  if frames == 0: frames = len(timestamps) - start
  timestamps = timestamps[start : start + frames]
  curframe = start
  for t in tqdm(timestamps):
    fig = plt.figure(figsize=(16,9))

    # ZED left, right, depth
    ax1 = plt.subplot2grid((3,4), (0,0))
    ax2 = plt.subplot2grid((3,4), (0,1))
    ax3 = plt.subplot2grid((3,4), (0,2))
    
    # OptiTrack ground truth
    ax4 = plt.subplot2grid((3,4), (0,3), rowspan = 2, projection = '3d')
    
    # RealSense RGB, depth
    ax5 = plt.subplot2grid((3,4), (1,0))
    ax6 = plt.subplot2grid((3,4), (1,1))

    # mmWave azimuth, doppler, detected points
    ax7 = plt.subplot2grid((3,4), (1,2))
    ax8 = plt.subplot2grid((3,4), (2,0))
    ax9 = plt.subplot2grid((3,4), (2,1), projection='3d')

    ## ReSpeaker waveform, direction
    ax10 = plt.subplot2grid((3,4), (2,2))
    ax11 = plt.subplot2grid((3,4), (2,3), projection='polar')

    fig.suptitle(f'Frame {curframe}', fontsize = 16)

    ## Plot ZED images
    zed_camera_left = cv2.imdecode(np.array(zed[t][node]['zed_camera_left']), cv2.IMREAD_ANYCOLOR)
    ax1.imshow(cv2.cvtColor(zed_camera_left, cv2.COLOR_BGR2RGB))
    ax1.axis('off')
    ax1.set_title("ZED Left Image")

    zed_camera_right = cv2.imdecode(np.array(zed[t][node]['zed_camera_right']), cv2.IMREAD_ANYCOLOR)
    ax2.imshow(cv2.cvtColor(zed_camera_right, cv2.COLOR_BGR2RGB))
    ax2.axis('off')
    ax2.set_title("ZED Right Image")

    zed_camera_depth = np.array(zed[t][node]['zed_camera_depth'])
    ax3.imshow(zed_camera_depth, cmap='turbo', vmin=0, vmax=10000)
    ax3.axis('off')
    ax3.set_title("ZED Depth Visual")

    ## Plot OptiTrack ground truth
    for obj in json.loads(mocap[t]['mocap'][()]): 
      x, y, z = obj['position']
      if obj['type'] == 'node': 
        rot = R.from_matrix(np.array(obj['rotation']).reshape((3, 3))).inv()
        x_axis = rot.apply((0.3, 0, 0))
        y_axis = rot.apply((0, 0.3, 0))
        z_axis = rot.apply((0, 0, 0.3))
        ax4.plot([z, z + x_axis[2]], [x, x + x_axis[0]], [y, y + x_axis[1]], color = 'red')
        ax4.plot([z, z + y_axis[2]], [x, x + y_axis[0]], [y, y + y_axis[1]], color = 'green')
        ax4.plot([z, z + z_axis[2]], [x, x + z_axis[0]], [y, y + z_axis[1]], color = 'blue')
      else: 
        rx, ry, rz = obj['roll'], obj['pitch'], obj['yaw']
        dx, dy, dz = R.from_rotvec([rx, ry, rz]).inv().apply((0, 0, 0.3))
        ax4.plot([z, z + dz], [x, x + dx], [y, y + dy], color = mocap_c[obj['id']])

      mocap_x[obj['id']].append(x)
      mocap_y[obj['id']].append(y)
      mocap_z[obj['id']].append(z)

    for i in range(len(mocap_x)): 
      ax4.scatter3D(mocap_z[i], mocap_x[i], mocap_y[i], color = mocap_c[i], alpha = 0.5)

    ax4.set_xlim(-2, 3)
    ax4.set_ylim(-2, 3)
    ax4.set_zlim(0, 1)

    ax4.set_box_aspect((5, 5, 1))
    ax4.grid(True, color="gray", linestyle="--")
    ax4.set_xlabel("x [m]")
    ax4.set_ylabel("y [m]")
    ax4.set_zlabel("z [m]")
    ax4.set_title("OptiTrack Ground Truth")

    ## Plot Intel RealSense
    realsense_camera_img = cv2.imdecode(np.array(realsense[t][node]['realsense_camera_img']), cv2.IMREAD_ANYCOLOR)
    ax5.imshow(cv2.cvtColor(realsense_camera_img, cv2.COLOR_BGR2RGB))

    realsense_camera_depth = cv2.imdecode(np.array(realsense[t][node]['realsense_camera_depth']), cv2.IMREAD_ANYCOLOR)
    ax6.imshow(cv2.cvtColor(realsense_camera_depth, cv2.COLOR_BGR2RGB))
    
    ax5.axis('off')
    ax5.set_title("RealSense RGB")

    ax6.axis('off')
    ax6.set_title("RealSense Depth")

    ## Plot mmWave Radar
    if 'azimuth_static' not in mmwave[t][node]: 
      ax7.text(0.5,0.5,"N/A", fontsize=30, ha='center', va='center')
    else:
      azimuth_static = np.array(mmwave[t][node]['azimuth_static'])
      ax7.imshow(azimuth_static, cmap='turbo', aspect='auto')

    if 'range_doppler' not in mmwave[t][node]: 
      ax8.text(0.5,0.5,"N/A", fontsize=30, ha='center', va='center')
    else:
      range_doppler = np.array(mmwave[t][node]['range_doppler'])
      ax8.imshow(range_doppler.T, cmap='turbo', aspect='auto')


    x, y, z, v = [], [], [], []
    for p in json.loads(mmwave[t][node]['detected_points'][()]).values():
      x.append(p['x'])
      y.append(p['y'])
      z.append(p['z'])
      v.append(p['v'])

    ax9.scatter(x, y, z, c=v, cmap="Spectral")
      
    ax7.axis('off')
    ax7.set_title("mmWave Azimuth Heatmap")
    # ax7.set_aspect('equal', adjustable='box')

    ax8.axis('off')
    ax8.set_title("mmWave Doppler Heatmap")

    d = 11 
    ax9.set_xlabel('x [m]')
    ax9.set_ylabel('y [m]')
    ax9.set_zlabel('z [m]')

    ax9.set_xlim3d((-d / 2, +d / 2))
    ax9.set_ylim3d((0, d))
    ax9.set_zlim3d((-d / 2, +d / 2))
    ax9.set_title("mmWave Detected Points")

    ## Plot ReSpeaker
    audio_np = np.array(respeaker[t][node]['mic_waveform']).T[0]
    wave = torch.tensor(audio_np, dtype = torch.float)
    spectro = torchaudio.transforms.Spectrogram()(wave)
    ax10.imshow(spectro, aspect='auto')

    angle = np.array(respeaker[t][node]['mic_waveform'].attrs['direction'])
    ax11.scatter(math.radians(angle), 1)

    ax10.axis('off')
    ax10.set_title("ReSpeaker Spectrogram")

    ax11.set_theta_zero_location("NW")
    ax11.set_theta_direction(-1)
    ax11.set_yticklabels([])
    ax11.set_title("ReSpeaker Direction")

    fig.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    out.write(data) 
    plt.close()

    curframe += 1

  out.release()
  mocap.close()
  realsense.close()
  zed.close()
  mmwave.close()
  respeaker.close()

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('node', type=str, help='Node ID')
  parser.add_argument('mocap_hdf5', type=str, help='Path to OptiTrack HDF5')
  parser.add_argument('realsense_hdf5', type=str, help='Path to RealSense HDF5')
  parser.add_argument('zed_hdf5', type=str, help='Path to ZED HDF5')
  parser.add_argument('mmwave_hdf5', type=str, help='Path to mmWave HDF5')
  parser.add_argument('respeaker_hdf5', type=str, help='Path to ReSpeaker HDF5')
  parser.add_argument('output_mp4', type=str, help='Path to output mp4')
  parser.add_argument('-s', '--start', type=int, default=0, help='Start frame')
  parser.add_argument('-f', '--frames', type=int, default=0, help='Number of frames to render')
  args = parser.parse_args()
  visualize(args.node, args.mocap_hdf5, args.realsense_hdf5, args.zed_hdf5, args.mmwave_hdf5, args.respeaker_hdf5, args.output_mp4, args.start, args.frames)
