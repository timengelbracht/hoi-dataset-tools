# Use official ROS Noetic base image
FROM ros:noetic-ros-base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-dev \
    python3-numpy \
    python3-catkin-tools \
    ros-noetic-cv-bridge \
    ros-noetic-tf \
    ros-noetic-tf-conversions \
    ros-noetic-tf2-ros \
    ros-noetic-image-transport \
    ros-noetic-message-filters \
    ros-noetic-rviz \
    libeigen3-dev \
    libboost-all-dev \
    libblas-dev \
    liblapack-dev \
    libv4l-dev \
    libsuitesparse-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python packages
RUN python3 -m pip install --upgrade pip
RUN pip3 install \
    setuptools \
    numpy \
    scipy \
    matplotlib \
    opencv-python \
    pyquaternion \
    catkin_pkg

# Set NumPy include path for C++ extensions (for numpy_eigen)
ENV CPLUS_INCLUDE_PATH=/usr/local/lib/python3.8/dist-packages/numpy/core/include

# Create and configure the catkin workspace
RUN mkdir -p /catkin_ws/src
WORKDIR /catkin_ws

# Clone Kalibr and dependencies
RUN git clone https://github.com/ethz-asl/kalibr.git src/kalibr
RUN git clone https://github.com/ori-drs/allan_variance_ros.git src/allan_variance_ros

# Build the workspace
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && cd /catkin_ws && catkin build --cmake-args -DCMAKE_BUILD_TYPE=Release"

# Auto-source the workspace in future shells
RUN echo 'source /catkin_ws/devel/setup.bash' >> ~/.bashrc

CMD ["bash"]

