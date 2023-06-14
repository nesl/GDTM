#!/bin/bash

for entry in data/raw/node1/*mmwave*bag
do
  python3 src/extract_mmwave.py $entry data/processed/node_1
done

for entry in data/raw/node2/*mmwave*bag
do
  python3 src/extract_mmwave.py $entry data/processed/node_2
done

for entry in data/raw/node3/*mmwave*bag
do
  python3 src/extract_mmwave.py $entry data/processed/node_3
done
