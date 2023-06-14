#!/bin/bash

n=$(ls data/split | grep mocap | wc -l)

for ((i=0; i<n; ++i)); do 
  mkdir -p data/chunk_$i
  mkdir -p data/chunk_$i/node_1
  mkdir -p data/chunk_$i/node_2
  mkdir -p data/chunk_$i/node_3

  mv data/split/mocap_$i.hdf5 data/chunk_$i/mocap.hdf5

  mv data/split/node_1/realsense_$i.hdf5 data/chunk_$i/node_1/realsense.hdf5
  mv data/split/node_1/zed_$i.hdf5 data/chunk_$i/node_1/zed.hdf5
  mv data/split/node_1/mmwave_$i.hdf5 data/chunk_$i/node_1/mmwave.hdf5
  mv data/split/node_1/respeaker_$i.hdf5 data/chunk_$i/node_1/respeaker.hdf5

  mv data/split/node_2/realsense_$i.hdf5 data/chunk_$i/node_2/realsense.hdf5
  mv data/split/node_2/zed_$i.hdf5 data/chunk_$i/node_2/zed.hdf5
  mv data/split/node_2/mmwave_$i.hdf5 data/chunk_$i/node_2/mmwave.hdf5
  mv data/split/node_2/respeaker_$i.hdf5 data/chunk_$i/node_2/respeaker.hdf5

  mv data/split/node_3/realsense_$i.hdf5 data/chunk_$i/node_3/realsense.hdf5
  mv data/split/node_3/zed_$i.hdf5 data/chunk_$i/node_3/zed.hdf5
  mv data/split/node_3/mmwave_$i.hdf5 data/chunk_$i/node_3/mmwave.hdf5
  mv data/split/node_3/respeaker_$i.hdf5 data/chunk_$i/node_3/respeaker.hdf5
done

rmdir data/split/node_1
rmdir data/split/node_2
rmdir data/split/node_3
rmdir data/split
