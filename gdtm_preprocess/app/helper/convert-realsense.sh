#!/bin/bash

for entry in data/raw/node1/*realsense*bag
do
  python3 src/convert_realsense.py 1 $entry data/processed/aligned.csv data/processed -div 4
done

for entry in data/raw/node2/*realsense*bag
do
  python3 src/convert_realsense.py 2 $entry data/processed/aligned.csv data/processed -div 4
done

for entry in data/raw/node3/*realsense*bag
do
  python3 src/convert_realsense.py 3 $entry data/processed/aligned.csv data/processed -div 4
done
