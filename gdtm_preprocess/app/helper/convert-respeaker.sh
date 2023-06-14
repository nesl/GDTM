#!/bin/bash

for entry in data/raw/node1/*respeaker*bag
do
  python3 src/convert_respeaker.py 1 $entry data/processed/aligned.csv data/processed
done

for entry in data/raw/node2/*respeaker*bag
do
  python3 src/convert_respeaker.py 2 $entry data/processed/aligned.csv data/processed
done

for entry in data/raw/node3/*respeaker*bag
do
  python3 src/convert_respeaker.py 3 $entry data/processed/aligned.csv data/processed
done
