#!/bin/bash

for entry in data/raw/node1/*mmwave*bag
do
  python3 src/convert_mmwave.py 1 $entry data/processed/aligned.csv data/processed
done

for entry in data/raw/node2/*mmwave*bag
do
  python3 src/convert_mmwave.py 2 $entry data/processed/aligned.csv data/processed
done

for entry in data/raw/node3/*mmwave*bag
do
  python3 src/convert_mmwave.py 3 $entry data/processed/aligned.csv data/processed
done
