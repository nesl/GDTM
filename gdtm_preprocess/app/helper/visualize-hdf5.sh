#!/bin/bash

python3 src/visualize_hdf5.py ${1:-1} data/processed/mocap.hdf5 data/processed/node_${1:-1}/realsense.hdf5 data/processed/node_${1:-1}/zed.hdf5 data/processed/node_${1:-1}/mmwave.hdf5 data/processed/node_${1:-1}/respeaker.hdf5 data/processed/${4:-output.mp4} -s ${2:-0} -f ${3:-0}
