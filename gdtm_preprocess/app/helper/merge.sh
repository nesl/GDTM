#!/bin/bash

mkdir -p merged
mkdir -p merged/node_1
mkdir -p merged/node_2
mkdir -p merged/node_3

for hdf5 in mocap.hdf5 node_1/realsense.hdf5 node_1/zed.hdf5 node_1/mmwave.hdf5 node_1/respeaker.hdf5 node_2/realsense.hdf5 node_2/zed.hdf5 node_2/mmwave.hdf5 node_2/respeaker.hdf5 node_3/realsense.hdf5 node_3/zed.hdf5 node_3/mmwave.hdf5 node_3/respeaker.hdf5; do 
  args=()
  for arg in "$@"; do 
    args+=("data/${arg}/${hdf5}")
  done
  python3 src/merge.py "merged/${hdf5}" "${args[@]}"
done

mv merged data
