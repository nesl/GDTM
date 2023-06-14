#!/bin/bash

helper/process-optitrack.sh
helper/extract-realsense.sh
helper/extract-zed.sh
helper/extract-mmwave.sh
helper/extract-respeaker.sh
helper/align.sh ${1:-0} ${2:-1000000000}
helper/convert-optitrack.sh
helper/convert-realsense.sh
helper/convert-zed.sh
helper/convert-mmwave.sh
helper/convert-respeaker.sh
helper/split.sh
helper/reorganize-split.sh
