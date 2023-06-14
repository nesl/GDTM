#!/bin/bash

for entry in data/raw/node1/*respeaker*bag
do
  python3 src/extract_respeaker.py $entry data/processed/node_1
done

for entry in data/raw/node2/*respeaker*bag
do
  python3 src/extract_respeaker.py $entry data/processed/node_2
done

for entry in data/raw/node3/*respeaker*bag
do
  python3 src/extract_respeaker.py $entry data/processed/node_3
done
