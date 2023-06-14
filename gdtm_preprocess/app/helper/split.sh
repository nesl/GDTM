#!/bin/bash

mkdir -p data/split
mkdir -p data/split/node_1
mkdir -p data/split/node_2
mkdir -p data/split/node_3

python3 src/split.py data/processed/mocap.hdf5 data/raw/metadata.json data/split

python3 src/split.py data/processed/node_1/realsense.hdf5 data/raw/metadata.json data/split/node_1
python3 src/split.py data/processed/node_1/zed.hdf5 data/raw/metadata.json data/split/node_1
python3 src/split.py data/processed/node_1/mmwave.hdf5 data/raw/metadata.json data/split/node_1
python3 src/split.py data/processed/node_1/respeaker.hdf5 data/raw/metadata.json data/split/node_1

python3 src/split.py data/processed/node_2/realsense.hdf5 data/raw/metadata.json data/split/node_2
python3 src/split.py data/processed/node_2/zed.hdf5 data/raw/metadata.json data/split/node_2
python3 src/split.py data/processed/node_2/mmwave.hdf5 data/raw/metadata.json data/split/node_2
python3 src/split.py data/processed/node_2/respeaker.hdf5 data/raw/metadata.json data/split/node_2

python3 src/split.py data/processed/node_3/realsense.hdf5 data/raw/metadata.json data/split/node_3
python3 src/split.py data/processed/node_3/zed.hdf5 data/raw/metadata.json data/split/node_3
python3 src/split.py data/processed/node_3/mmwave.hdf5 data/raw/metadata.json data/split/node_3
python3 src/split.py data/processed/node_3/respeaker.hdf5 data/raw/metadata.json data/split/node_3
