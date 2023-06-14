#!/bin/bash

for entry in data/raw/node1/*zed*svo
do
  python3 src/extract_zed.py $entry data/processed/node_1
done

for entry in data/raw/node2/*zed*svo
do
  python3 src/extract_zed.py $entry data/processed/node_2
done

for entry in data/raw/node3/*zed*svo
do
  python3 src/extract_zed.py $entry data/processed/node_3
done
