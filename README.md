
# GDTM
Dataset Repository of NeurIPS 2023 Track on Datasets and Benchmarks Paper #207

## Overview


### Abstract
One of the critical sensing tasks in indoor environments is geospatial tracking, i.e., constantly detecting and locating objects moving across a scene. Successful development of multimodal sensor fusion tracking algorithms relies on large multimodal datasets where common modalities exist and are time-aligned, and such datasets are not readily available. Moreover, some existing datasets either employs a single centralized sensor node or a set of sensors whose positions and orientations are fixed. Models developed on such datasets have difficulties generalizing to different sensor placements To fill these gaps, we propose GDTM, a nine-hour dataset for multimodal object tracking with distributed multimodal sensors and reconfigurable sensor node placements. This GTDM dataset enables the exploration of several research problems, including creating multimodal sensor fusion architectures robust to adverse sensing conditions and creating distributed object tracking systems robust to sensor placement variances.

### External Links

We will be hosting the dataset on IEEE Dataport under a CC-BY-4.0 license for public access. The public dataset repository will be ready before cameraready. We hereby provide **a Google Drive link to part of the dataset available for the reviewers** before the terabyte full dataset is available online.
https://drive.google.com/drive/folders/1N0b8-o9iipR7m3sq7EnTHrk_fRR5eFug?usp=sharing

We provide the dataset documentation and intended uses using the **dataset nutrition labels** framework in the following link:
https://datanutrition.org/labels/v3/?id=0c819282-b39c-451f-aa8c-f2044bfacf21

To ensure the reproducibility of the results shown in this paper, we summarized the setup instructions, the link to pre-processed sample data, and the **code used to generate our results** in:
https://github.com/nesl/GDTM-tracking

A **demo video** of our baseline results is available at
https://youtu.be/4EO5Z2IxO0o


### Internal Links
[Further Information on the Raw Dataset](#"Dataset-(Raw)")
[How to Pre-process GDTM Dataset](#Pre-processing-GDTM-Dataset)
[How to Load Pre-processed GDTM for Your Own Work](#How-to-Use-Pre-processed-GDTM)

## Dataset (Raw) 

### Raw Dataset

### Dataset Metadata

## Pre-processing GDTM Dataset



### Overview


#### Expected Dataset Structure after Pre-processing

### Installation Instructions

### Raw Data -> HDF5 Files

### Merging

### Rendering and Visualization (Optional)

## How to Use Pre-processed GDTM

In this section, we showcase how the proGDTM dataset can be loaded and used. The sample dataset loading script is based on [mmtracking](https://mmtracking.readthedocs.io/en/latest/install.html) library developed by OpenMMLab.  However, you can use any libraries for downstream processing as the dataset has been read into dictionaries and numpy arrays.

### Installation
Firstly, open a terminal and create a conda vertual environment using python 3.9:
```
conda create -n gdtm python=3.9
conda activate gdtm
```
Install dependencies:
```
sudo apt-get install gcc

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html

pip install "mmdet<3" "mmtrack==0.14" "h5py==3.8" "ipdb"

```

### Usage
We put the sample usage of GDTM in USING_GDTM/ folder of this repository.    The main file is _data_test.py_, with a few other supporting python scripts.

Firstly we enter the USING_GDTM/ folder
```
cd GDTM/USING_GDTM
```
The data pre-processed in the previous section should be copied to a /data folder under the current directory. The hierachy is shown below:
```
─── GDTM/
    └── USING_GDTM/
        ├── data_test.py
        ├── ...
        └── data/
            ├── node1
            ├── node2
            ├── node3
            └── mocap.hdf5
```
Finally we need to run the _data_test.py_:
```
python data_test.py
```

The _data_test.py_ is very straightforward. The script first define a few pipelines to load data from each modality, and then configures the filepaths. The scripts then renders a video of the ground truth and the sensor data. In the end, the scripts enters a loop where we visit the dataset timestamp by timestamp. Inside each timestamp, the dataset is a dictionary containing these keys (modality, nodes):
```
[('mocap', 'mocap'), ('azimuth_static', 'node_1'), ('range_doppler', 'node_1'), ('realsense_camera_depth', 'node_1'), ('realsense_camera_img', 'node_1'), ('mic_waveform', 'node_1'), ('zed_camera_depth', 'node_1'), ('zed_camera_left', 'node_1'), ('azimuth_static', 'node_2'), ('range_doppler', 'node_2'), ('realsense_camera_depth', 'node_2'), ('realsense_camera_img', 'node_2'), ('mic_waveform', 'node_2'), ('zed_camera_depth', 'node_2'), ('zed_camera_left', 'node_2'), ('azimuth_static', 'node_3'), ('range_doppler', 'node_3'), ('realsense_camera_depth', 'node_3'), ('realsense_camera_img', 'node_3'), ('mic_waveform', 'node_3'), ('zed_camera_depth', 'node_3'), ('zed_camera_left', 'node_3')]
```
The code will step into an ipdb breakpoint where you can play with the loaded data.


**Important Troubleshooting Note**: to avoid repetitive dataset loading, the _data_test.py_ will also cache the data into /dev/shm/cache_train

Thus if you encounter any issues during executing _data_test.py_. or just want to test with some new data, make sure to do
```
rm -r /dev/shm/cache_*
```
to clear any pre-loaded data in the memory.

### See Also
For further usage such as how to train a multimodal sensor fusion models, we provide examples in https://github.com/nesl/GDTM-Tracking

## Citations

If you find this project useful in your research, please consider cite:

```
@inproceedings{wang2023gdtm,
    title={GTDM: An Indoor Geospatial Tracking Dataset with Distributed Multimodal Sensors},
    author={Jeong, Ho Lyun and Wang, Ziqi and Samplawski, Colin and Wu, Jason and Fang, Shiwei and Kaplan, Lance and Ganesan, Deepak and Marlin, Benjamin and Srivastava, Mani},
    booktitle={submission to the Thirty-seventh Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
    year={2023}
}
```

