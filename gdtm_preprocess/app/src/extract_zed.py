import pyzed.sl as sl
import numpy as np
import argparse
import rosbag
from std_msgs.msg import Int32, String
from tqdm import tqdm
from pathlib import Path
import csv
import os

csv_filename = 'None'

def process(zed_svo, output_folder, img_format = 'png', depth_mode = 'ultra', sensing_mode = 'standard'):

    # check if zed svo exists
    if not os.path.isfile(zed_svo): 
      print(f'Error: SVO file {zed_svo} not found')
      exit(1)

    if 'svo' not in zed_svo: 
      print(f'Error: {zed_svo} is not an SVO file')
      exit(1)

    zed_bag = zed_svo.replace('svo', 'bag')

    if not os.path.isfile(zed_bag): 
      print(f'Error: Bag file {zed_bag} not found')
      exit(1)

    print(f'Processing {zed_svo}')

    # Set SVO file for playback
    init_parameters = sl.InitParameters()
    init_parameters.depth_mode = {
      'ultra': sl.DEPTH_MODE.ULTRA, 
      'quality': sl.DEPTH_MODE.QUALITY, 
      'performance': sl.DEPTH_MODE.PERFORMANCE
    }[depth_mode]

    # don't convert in realtime
    init_parameters.svo_real_time_mode = False 
    # use milliliter units (for depth measurements)
    init_parameters.coordinate_units = sl.UNIT.MILLIMETER
    init_parameters.set_from_svo_file(zed_svo)

    # open zed
    zed = sl.Camera()
    status = zed.open(init_parameters)

    if status != sl.ERROR_CODE.SUCCESS:
      print(repr(status))
      zed.close()
      exit(1)

    # get image size
    image_size = zed.get_camera_information().camera_resolution
    width = image_size.width
    height = image_size.height
    width_sbs = width * 2
    
    # prepare single image containers
    left_image = sl.Mat()
    right_image = sl.Mat()
    depth_view_image = sl.Mat()
    sidebyside_image = sl.Mat()

    depth_measure = sl.Mat()
    
    runtime_parameter = sl.RuntimeParameters()
    runtime_parameter.sensing_mode = {
      'standard': sl.SENSING_MODE.STANDARD, 
      'fill': sl.SENSING_MODE.FILL
    }[sensing_mode]
    if False: 
      if sensing_mode == 'standard':
        runtime_parameter.sensing_mode = sl.SENSING_MODE.STANDARD
      elif sensing_mode == 'fill':
        runtime_parameter.sensing_mode = sl.SENSING_MODE.FILL
      else:
        print("Sensing Mode Error! [standard, fill] only.")
        exit(1)

    n_frames = zed.get_svo_number_of_frames()
        
    # open bag
    bag = rosbag.Bag(zed_bag)
    msgs = bag.read_messages()

    # create output directory
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    timestamps = []
    for topic, msg, t in msgs:
      if "zed_timestamp" in topic:
        timestamps.append(msg.data)
        
    csv_file = []
    output_csv = os.path.join(output_folder, 'zed.csv')
    csv_file = open(output_csv, 'a', newline='')
    writer = csv.writer(csv_file)
    
    for i in tqdm(range(n_frames+1)):
        state = zed.grab(runtime_parameter)
        svo_position = zed.get_svo_position()
        if state == sl.ERROR_CODE.SUCCESS:
            timestamp = timestamps[svo_position]
            line = [timestamp]
            writer.writerow(line)

        else:
            # get current frame count
            print(f'Current frame position: {svo_position + 1}/{n_frames}')
            print(f"Frame grab error code: {state}")

    csv_file.close()
    zed.close()
    print(f'Finished processing {zed_svo}')

def save_img(output_folder, svo_fname, f_num, ext, mat, img_format):
    svo_basename = os.path.basename(svo_fname).replace('.svo', '')
    img_file_name = os.path.join(output_folder, f'{svo_basename}_{f_num}_{ext}.{img_format}')
    status = mat.write(img_file_name)
    if status != sl.ERROR_CODE.SUCCESS:
      print("Error saving image")

    return img_file_name

def save_depth(output_folder, svo_fname, f_num, ext, mat):
    svo_basename = os.path.basename(svo_fname).replace('.svo', '')
    depth_file_name = os.path.join(output_folder, f'{svo_basename}_{f_num}_{ext}.npz')
    depth_np_array = mat.get_data()
    depth_np_array = depth_np_array.astype(np.int16)

    np.savez_compressed(depth_file_name, depth_np_array)
    return depth_file_name

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('zed_svo', type=str, help='Path to ZED SVO file')
    parser.add_argument('output_folder', type=str, help='Path to output folder')
    parser.add_argument('-t', action="store", choices=['jpg','png'], default='png', help='Image format (jpg, png), default png')
    parser.add_argument('-m', action="store", choices=['ultra', 'performance', 'quality'], default='ultra', help="Depth quality mode (ultra, performance, quality), default ultra")
    parser.add_argument('-s', action="store", choices=['standard', 'fill'], default='standard', help="Depth sensing mode (standard, fill), default standard")
    # parser.add_argument('-c', action="store", default='None', help="Specify timestamp csv output file. Default file name be svo file name.")
    args = parser.parse_args()
    process(args.zed_svo, args.output_folder, args.t, args.m, args.s)

