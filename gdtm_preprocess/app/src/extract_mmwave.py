import argparse
import csv
import os
import cv2
import rosbag

from pathlib import Path
from tqdm import tqdm
from datetime import datetime

def process(mmwave_bag, output_folder):

  # check if mmwave bag exists
  if not os.path.isfile(mmwave_bag):
    print(f'Error: Bag file {mmwave_bag} not found')
    exit(1)

  # open bag
  print(f'Processing {mmwave_bag}')
  bag = rosbag.Bag(mmwave_bag)

  # create output directory
  Path(output_folder).mkdir(parents=True, exist_ok=True)

  # open output csv and videos
  output_csv = os.path.join(output_folder, 'mmwave.csv')
  csv_file = open(output_csv, 'a', newline='')
  writer = csv.writer(csv_file)

  msgs = bag.read_messages()
  for topic, msg, t in tqdm(msgs, total=bag.get_message_count()):
    if 'RadarData' not in topic: continue   

    timestamp = datetime.strptime(msg.timestamp, "%Y-%m-%d %H:%M:%S.%f")
    writer.writerow([timestamp])

  csv_file.close()
  bag.close()
  print(f'Finished processing {mmwave_bag}')

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('mmwave_bag', type=str, help='Path to mmWave bag file')
  parser.add_argument('output_folder', type=str, help='Path to output folder')
  args = parser.parse_args()
  process(args.mmwave_bag, args.output_folder)
