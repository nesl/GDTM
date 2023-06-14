#!/bin/bash

python3 src/align.py --optitrack data/processed/optitrack.csv --input $(find . -wholename './data/processed/node*/*.csv') --output data/processed/aligned.csv --start_frame ${1:-0} --frames ${2:-1000000000}
