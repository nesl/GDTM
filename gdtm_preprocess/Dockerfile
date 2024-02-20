FROM stereolabs/zed:3.7-gl-devel-cuda11.4-ubuntu20.04

# https://gitlab.com/nvidia/container-images/cuda/-/issues/158
RUN apt-key del "7fa2af80" \
    && export this_distro="$(cat /etc/os-release | grep '^ID=' | awk -F'=' '{print $2}')" \
    && export this_version="$(cat /etc/os-release | grep '^VERSION_ID=' | awk -F'=' '{print $2}' | sed 's/[^0-9]*//g')" \
    && apt-key adv --fetch-keys "https://developer.download.nvidia.com/compute/cuda/repos/${this_distro}${this_version}/x86_64/3bf863cc.pub"




ARG DEBIAN_FRONTEND="noninteractive"
ENV TZ="America/New_York"


#Basic install items
RUN apt-get update && apt-get install -y --no-install-recommends make g++ cmake nano gnupg2 software-properties-common python3-pip


#Install opencv library
RUN apt-get install -y libopencv-dev

# WORKDIR /workspace

# add the ROS deb repo to the apt sources list
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
		cmake \
		build-essential \
		curl \
		wget \
		gnupg2 \
		lsb-release \
    && rm -rf /var/lib/apt/lists/*

RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

RUN apt install curl
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

SHELL ["/bin/bash", "-c"]

RUN apt-get update && \
    apt-get install -y ros-noetic-desktop-full && \
    source /opt/ros/noetic/setup.bash && \
    echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc && \
    source ~/.bashrc && \
    apt-get install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential && \
    apt-get install -y python3-rosdep && \
    rosdep init && \
    rosdep update


RUN pip3 install matplotlib numpy jupyter pandas scipy


#Get passed in build args
ARG MCP_USER
ARG MCP_HOME
ARG MCP_ROOT
ARG MCP_APP_DIR

#Set environment variables from build args
ENV MCP_USER=$MCP_USER
ENV MCP_HOME=$MCP_HOME
ENV MCP_ROOT=$MCP_ROOT
ENV MCP_APP_DIR=$MCP_APP_DIR

#Create the MCP user
RUN groupadd ${MCP_USER} && useradd --create-home --shell /bin/bash -g ${MCP_USER} -G sudo,audio,dip,video,plugdev,dialout ${MCP_USER}

RUN add-apt-repository universe && \
    apt update && apt-get install -y ffmpeg python3 python-dev python3-dev build-essential python3-pip
RUN pip3 install tqdm pydub bagpy
RUN pip install pysoundfile bagpy pydub pyyaml
RUN pip3 install torchaudio
RUN pip3 install matplotlib==3.5.1
RUN pip3 install ipdb
RUN pip3 install h5py
# RUN pip install wheel
# RUN pip install pyaudio
RUN pip3 install psutil==5.9.8
## Ouster LIDAR related
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install 'ouster-sdk[examples]'


RUN ls
COPY --chown=${MCP_USER}:${MCP_USER} app $MCP_APP_DIR/data_processing
RUN cd $MCP_APP_DIR/data_processing/ros_pkgs   && \
    . /opt/ros/noetic/setup.sh &&\
    catkin_make &&\
    catkin_make install && \
    echo "source $MCP_APP_DIR/data_processing/ros_pkgs/devel/setup.bash" >> ~/.bashrc

RUN echo "source $MCP_APP_DIR/data_processing/ros_pkgs/devel/setup.bash"

RUN mkdir -p $MCP_APP_DIR/data_processing/data

CMD bash

WORKDIR $MCP_APP_DIR/data_processing
