import argparse
import csv
import os
import rosbag
from io import BytesIO
from pydub import AudioSegment
from pydub.utils import make_chunks

from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import datetime as dt
import numpy as np

def nearest(items, pivot):
  return min(items, key=lambda x: abs(x - pivot))

def process(respeaker_bag, output_folder, audio_length = 66):

  # check if respeaker bag exists
  if not os.path.isfile(respeaker_bag):
    print(f'Error: Bag file {respeaker_bag} not found')
    exit(1)

  # open bag
  print(f'Processing {respeaker_bag}')
  bag = rosbag.Bag(respeaker_bag)

  # create output directory
  Path(output_folder).mkdir(parents=True, exist_ok=True)

  # open output csv and videos
  output_csv = os.path.join(output_folder, 'respeaker.csv')
  csv_file = open(output_csv, 'a', newline='')
  writer = csv.writer(csv_file)

  timestamps = []

  msgs = bag.read_messages()
  for topic, msg, t in tqdm(msgs, total=bag.get_message_count()):
    if "timestamp" in topic:
      timestamps.append(datetime.strptime(msg.data, "%Y-%m-%d %H:%M:%S.%f"))
    elif 'audio-flac' in topic: 
      audio = msg.data
  
  # Decode the flac audio into memory
  mem_file = BytesIO(audio.encode('ISO-8859-1'))
  audio_file = AudioSegment.from_file(mem_file)

  if False: 
    print("Audio File length(s): " + str(audio_file.duration_seconds))
    print("# of Channels: " + str(audio_file.channels))
    print("Sample width: " + str(audio_file.sample_width))
    print("Frame Rate: " + str(audio_file.frame_rate))
    print("# of frames: " + str(audio_file.frame_count()))

  # Separate flac file into chunks based on audio_length time.
  chunks = make_chunks(audio_file, audio_length)

  # start_time of the audio file
  start_time = timestamps[0]

  for i, chunk in enumerate(tqdm(chunks)):
    current_time = start_time + dt.timedelta(milliseconds=audio_length * i)
    idx = timestamps.index(nearest(timestamps, current_time))
    dir_time = timestamps[idx]

    timestamp = np.array(current_time).astype(np.datetime64).item()
    writer.writerow([timestamp])

  csv_file.close()
  bag.close()
  print(f'Finished processing {respeaker_bag}')

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('respeaker_bag', type=str, help='Path to ReSpeaker bag file')
  parser.add_argument('output_folder', type=str, help='Path to output folder')
  args = parser.parse_args()
  process(args.respeaker_bag, args.output_folder)
