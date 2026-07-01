#!/bin/bash
set -e

# Source ROS
source /opt/ros/noetic/setup.bash

# Source workspace if already built
if [ -f /catkin_ws/devel/setup.bash ]; then
    source /catkin_ws/devel/setup.bash
fi

# Clone and build zed-ros-wrapper if not already present
if [ ! -d "/catkin_ws/src/zed-ros-wrapper" ]; then
    echo "Cloning ZED ROS Wrapper..."
    git clone --recursive https://github.com/stereolabs/zed-ros-wrapper.git /catkin_ws/src/zed-ros-wrapper
    cd /catkin_ws/src/zed-ros-wrapper && git checkout v3.8.x
    echo "🔧 Building ZED ROS Wrapper..."
    cd /catkin_ws
    catkin build zed_ros zed_wrapper zed_nodelets
    source /catkin_ws/devel/setup.bash
fi

# Continue to default command (bash or roslaunch etc.)
exec "$@"
