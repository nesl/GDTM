#!/bin/bash

#Get the IP address of this machine
thisip=`hostname -I | cut -d' ' -f1`

#Get environment varaibles
export MCP_USER=$USER
export MCP_HOME=/home/${MCP_USER}
export MCP_ROOT=${MCP_HOME}/mcp

export MCP_APP_DIR=${MCP_ROOT}/app 

data_folder=$(cd "$(dirname "$1")"; pwd)/$(basename "$1")
echo "Folder to attach: $data_folder"


xhost +si:localuser:root

#Start docker container
docker run -it --net=host \
            --gpus all \
            --privileged \
            -e DISPLAY=$DISPLAY \
            -v /tmp/.X11-unix:/tmp/.X11-unix \
            -v "$data_folder":$MCP_APP_DIR/data_processing/data \
            -v /etc/timezone:/etc/timezone:ro \
            data-processing bash
