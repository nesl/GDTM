import argparse
import csv
import os
import cv2
import rosbag

from pathlib import Path
from tqdm import tqdm

def process(realsense_bag, output_folder):

  # check if realsense bag exists
  if not os.path.isfile(realsense_bag):
    print(f'Error: Bag file {realsense_bag} not found')
    exit(1)

  # open bag
  print(f'Processing {realsense_bag}')
  bag = rosbag.Bag(realsense_bag)

  # create output directory
  Path(output_folder).mkdir(parents=True, exist_ok=True)

  # open output csvs
  rgb_csv = os.path.join(output_folder, 'realsense_rgb.csv')
  depth_csv = os.path.join(output_folder, 'realsense_depth.csv')
  rgb_csv_file = open(rgb_csv, 'a', newline='')
  depth_csv_file = open(depth_csv, 'a', newline='')
  rgb_writer = csv.writer(rgb_csv_file)
  depth_writer = csv.writer(depth_csv_file)

  # slice into frames
  rgb_list = []
  depth_list = []
  base_name = os.path.basename(realsense_bag).replace('.bag', '')
  base_name = os.path.join(output_folder, 'realsense', base_name)
  msgs = bag.read_messages()
  for topic, msg, t in tqdm(msgs, total=bag.get_message_count()):
    if 'rgb_timestamp' in topic:
      rgb_list.append([msg.data])

    if 'depth_timestamp' in topic:
      depth_list.append([msg.data])

  if len(rgb_list) != len(depth_list):
    print('Frame count mismatch detected')

  for entry in rgb_list: rgb_writer.writerow(entry)
  for entry in depth_list: depth_writer.writerow(entry)
  
  rgb_csv_file.close()
  depth_csv_file.close()
  bag.close()
  print(f'Finished processing {realsense_bag}')

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('realsense_bag', type=str, help='Path to RealSense bag file')
  parser.add_argument('output_folder', type=str, help='Path to output folder')
  args = parser.parse_args()
  process(args.realsense_bag, args.output_folder)
