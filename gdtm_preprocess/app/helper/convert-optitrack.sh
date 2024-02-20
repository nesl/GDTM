#!/bin/bash
cp data/raw/metadata.json data/processed
python3 src/convert_optitrack.py data/processed/aligned.csv data/processed/metadata.json data/processed
