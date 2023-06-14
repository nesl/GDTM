#!/bin/bash

for entry in data/raw/node1/*realsense*bag
do
  python3 src/process_realsense.py $entry data/processed/node_1
done

for entry in data/raw/node2/*realsense*bag
do
  python3 src/process_realsense.py $entry data/processed/node_2
done

for entry in data/raw/node3/*realsense*bag
do
  python3 src/process_realsense.py $entry data/processed/node_3
done
