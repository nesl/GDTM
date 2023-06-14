#!/bin/bash

#Get environment varaibles
export MCP_USER=$USER
export MCP_HOME=/home/${MCP_USER}
export MCP_ROOT=${MCP_HOME}/mcp

export MCP_APP_DIR=${MCP_ROOT}/app 

env=`printenv | grep MCP`

#Constructu build args from MCP environment variables
build_args=""
for line in $env; do
    build_args="${build_args} --build-arg $line "
done

#Run docker build
echo "running: docker build ${build_args} -t data-processing ."
docker build $build_args -t data-processing .
