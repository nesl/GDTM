# Data Processing for IoBT node data

## Pre-requisite

### ZED Stereo Camera:

ZED Stereo camera is recording SVO files during data collection, such files require ZED SDK to process as it's proprietary data format with lossy video encoding. Thus, to extract images and depth data from SVO files requires NVIDIA GPU, which is CUDA enabled (As of ZED SDK 3.7.2). 

### Other Sensors:

Other sensors are mostly stored in ROS bag files, to extract data, some requires compile the custom ROS message.

---
## Build

Run `bash build.sh` to build the data-processing docker image. Lunch docker image with `./run_dp.sh /path/to/data/`. The script will automatically attach the data folder (e.g. iobt_data_202205) to `data` folder inside the docker image. The docker will lunch bash program which files and codes can be executed.

---

### Process SVO files

```
# Process single svo file, the output will be stored in the zed_imgs folder with left and right png and npz for depth.
python3 svo_to_img.py -f svo_file.svo 

usage: svo_to_img.py [-h] -f FILENAME [-nv] [-t {jpg,png}] [-m {ultra,performance,quality}] [-s {standard,fill}]
                     [-o OUTPUT_FOLDER] [-c C]

optional arguments:
  -h, --help            show this help message and exit
  -f FILENAME           Required argument, specify SVO file.
  -nv                   Specify -nv to disable live view of the image. Default shows view.
  -t {jpg,png}          Specify image format. [jpg, png], default png
  -m {ultra,performance,quality}
                        Specify depth quality mode. [ultra, performance, quality], default ultra
  -s {standard,fill}    Specify depth sensing mode. [standard, fill], default standard
  -o OUTPUT_FOLDER      Specify output folder. Default zed_imgs
  -c C                  Specify timestamp csv output file. Default file name be svo file name.

# For example, the timestamps and corresponding png and npz files can be saved to the same csv file for different svo files
python3 svo_to_img.py -f svo_file_1.svo -c node_1_zed.csv
python3 svo_to_img.py -f svo_file_2.svo -c node_1_zed.csv
python3 svo_to_img.py -f svo_file_3.svo -c node_1_zed.csv


# Process all files in a folder can modify the run_svo_to_img_folder.sh. Check the script for more details.

```
---
### Process Motion Capture data files

This is dependent on the motion capture data collected at UMass Amherst, an example data can be found in `example_data` folder.

```
# To process all the motion capture data stored in the data/motion_capture_data folder to create a single csv file sorted based on timestamp:
python3 motioncap_data_process.py -f data/motion_capture_data/ -c motion_cap.csv

# for more
python3 motioncap_data_process.py -h
```
---
### Process mmWave Radar data files

Create individual pickle files that stores each mmWave data frame. 

```
python3 mmwave_to_frames.py -f data/iobt_node_1_mmwave_20220511_000000.bag 

# for more
python3 mmwave_to_frames.py -h
```

Check `run_mmwave_to_frames_folder.sh` for batch processing.

---
### Process Intel RealSense data file

Extract frames from Intel RealSense recorded files. Specify the bag file name, and the code will extract the frames from the mp4 files. Each frame has timestamp which will be stored in the csv file.

```
python3 realsense_to_frames.py -f data/iobt_node_1_realsense_20220511_000000.bag

# for more
python3 realsense_to_frames.py -h
```

Check `run_realsense_to_frames_folder.sh` for batch processing.

---
### Process ReSpeaker data files

Extract ReSpeaker flac audio file and slice into segments. Default audio slice length is 1/15s (66ms).

```
python3 respeaker_to_slice.py -f data/iobt_node_1_respeaker_20220511_000000.bag

# for more
python3 respeaker_to_slice.py -h
```

Check `run_respeaker_to_slice_folder.sh` for batch processing.

---
### Time alignment based on ZED images

Create individual python pickle files from ZED left, right, depth, motion capture data, mmWave data, Intel RealSense, and ReSpeaker data.

```
# To align motion capture data to ZED images.
python3 zed_timealign.py -z data/iobt_node_1_zed.csv -motion data/motion_capture_data/motion_cap.csv -wave data/iobt_node_1_mmwave.csv -real data/iobe_node_1_realsense.csv -speaker data/iobt_node_1_respeaker.csv -node iobt_node_1

# for more
python3 zed_timealign.py -h 
```

If the nearest timestamp for current data frame is more than 100 ms apart, this data frame will be discarded.

The output file name will be the node name + time. For example: `-node iobt_node_1` will result in `iobt_node_1_YearMonthDay_msSinceepoch.pickle` (`iobt_node_1_20220511_1652301633074.pickle`).


---
### Generate Visualization from time-synced pickle files

```
python3 visualize_synced.py -f data/dataframes/iobt_node_1_20220511_1652301633943.pickle -n 1800

# for more
python3 visualize_synced.py -h
```

