import rosbag
import soundfile as sf
import wave
import time
import numpy as np
import pandas as pd
from datetime import datetime
import datetime as dt
import socket
from io import BytesIO
import sys
import os

from pathlib import Path
from pydub import AudioSegment
from pydub.utils import make_chunks
import argparse
import csv
import codecs
from tqdm import tqdm
from utils import datetime_to_ms

import h5py

RESPEAKER_RATE = 16000
RESPEAKER_CHANNELS = 6 
RESPEAKER_WIDTH = 2
RESPEAKER_INDEX = 24 
CHUNK = 1024
RECORD_SECONDS = 60

def nearest(items, pivot):
  return min(items, key=lambda x: abs(x - pivot))

def convert(node, respeaker_bag, aligned_csv, output_folder, audio_length = 66):

  # check if respeaker bag exists
  if not os.path.isfile(respeaker_bag): 
    print(f'Error: Bag file {respeaker_bag} not found')
    exit(1)

  # check if aligned csv exists
  if not os.path.isfile(aligned_csv): 
    print(f'Error: CSV file {aligned_csv} not found')
    exit(1)

  # open bag
  print(f'Converting {respeaker_bag} to HDF5')
  bag = rosbag.Bag(respeaker_bag)

  # Read all the messages from the bag file into a list
  timestamps = []
  directions = []
  audio = []

  msgs = bag.read_messages()

  for topic, msg, t in msgs:
    if "direction" in topic:
      directions.append(msg.data)
    elif "timestamp" in topic:
      timestamps.append(datetime.strptime(msg.data, "%Y-%m-%d %H:%M:%S.%f"))
    elif "audio-flac" in topic:
      print("Found flac topic: %s"%topic)
      audio = msg.data
  
  # Decode the flac audio into memory
  mem_file = BytesIO(audio.encode('ISO-8859-1'))
  audio_file = AudioSegment.from_file(mem_file)

  # print(f'Audio file length (s): {audio_file.duration_seconds}')
  # print(f'# of channels: {audio_file.channels}')
  # print(f'Sample width: {audio_file.sample_width}')
  # print(f'Frame rate: {audio_file.frame_rate}')
  # print(f'# of frames: {audio_file.frame_count()}')

  df = pd.read_csv(aligned_csv)

  # Separate flac file into chunks based on audio_length time.
  chunks = make_chunks(audio_file, audio_length)

  # start_time of the audio file
  start_time = timestamps[0]

  # create directory for output
  Path(os.path.join(output_folder, f'node_{node}')).mkdir(parents=True, exist_ok=True)
  output_file = os.path.join(output_folder, f'node_{node}', 'respeaker.hdf5')
  f = h5py.File(output_file, 'a')

  node = f'node_{node}'
  index = 0
  for i, chunk in enumerate(tqdm(chunks)):
    # https://github.com/jiaaro/pydub/blob/master/API.markdown
    channels = chunk.split_to_mono()
    samples = [s.get_array_of_samples() for s in channels]
    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples[0].typecode).max

    current_time = start_time + dt.timedelta(milliseconds=audio_length * i)
    idx = timestamps.index(nearest(timestamps, current_time))
    dir_time = timestamps[idx]

    timestamp = np.array(current_time).astype(np.datetime64).item()
    while index < df['timestamp'].shape[0]: 
      curt = df.at[index, f'{node}_respeaker'] 
      if '.' not in curt: curt += '.0'
      curt = datetime.strptime(curt, '%Y-%m-%d %H:%M:%S.%f')
      if curt < timestamp: 
        index += 1
        continue
      elif curt > timestamp: 
        break

      framet = df.at[index, 'timestamp']
      if '.' not in framet: framet += '.0'
      ms = datetime_to_ms(datetime.strptime(framet, '%Y-%m-%d %H:%M:%S.%f'))

      if ms not in f.keys():
        f.create_group(ms)

      if node not in f[ms].keys():
        f[ms].create_group(node)

      f[ms][node].create_dataset('mic_waveform', data=fp_arr)
      
      f[ms][node]['mic_waveform'].attrs['direction'] = directions[idx]

      index += 1
    
  f.close()
  bag.close()
  print(f'Finished converting {respeaker_bag} to HDF5')

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('node', type=int, help='Node id')
  parser.add_argument('respeaker_bag', type=str, help="Path to ReSpeaker bag file")
  parser.add_argument('aligned_csv', type=str, help="Path to aligned CSV file")
  parser.add_argument('output_folder', type=str, help="Path to output folder")
  args = parser.parse_args()
  
  convert(args.node, args.respeaker_bag, args.aligned_csv, args.output_folder)
