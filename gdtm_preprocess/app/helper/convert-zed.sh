#!/bin/bash

for entry in data/raw/node1/*zed*svo
do
  python3 src/convert_zed.py 1 $entry data/processed/aligned.csv data/processed -div 4
done

for entry in data/raw/node2/*zed*svo
do
  python3 src/convert_zed.py 2 $entry data/processed/aligned.csv data/processed -div 4
done

for entry in data/raw/node3/*zed*svo
do
  python3 src/convert_zed.py 3 $entry data/processed/aligned.csv data/processed -div 4
done
